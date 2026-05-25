import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from timm.data import resolve_data_config, create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler

# Import our shared modules
from model import create_model
from data_loader import CarDataset, gather_kfold_items

import mlflow
from mlflow_utils import load_mlflow_config, init_mlflow, get_dvc_data_rev, log_pytorch_best


def train_one_fold(fold, train_items, val_items, num_classes, args, device):
    with mlflow.start_run(nested=True, run_name=f"fold{fold}"):
        mlflow.log_params({"fold": fold, **vars(args)})

        # Initialize using the shared factory
        model = create_model(args.model_name, num_classes, drop_rate=0.1, drop_path_rate=0.1)
        model = model.to(device)

        cfg = resolve_data_config({}, model=model)
        cfg['input_size'] = (3, args.img_size, args.img_size)

        train_tf = create_transform(
            input_size=cfg['input_size'], is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1', interpolation='bicubic',
            re_prob=0.25, re_mode='pixel', re_count=1, mean=cfg['mean'], std=cfg['std'],
            crop_pct=cfg.get('crop_pct', 0.95), hflip=0.5,
        )
        val_tf = create_transform(
            input_size=cfg['input_size'], is_training=False, interpolation='bicubic',
            mean=cfg['mean'], std=cfg['std'], crop_pct=cfg.get('crop_pct', 0.95),
        )

        train_loader = DataLoader(CarDataset(train_items, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(CarDataset(val_items, val_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=num_classes)
        train_loss_fn = SoftTargetCrossEntropy()
        val_loss_fn   = LabelSmoothingCrossEntropy(smoothing=0.1)

        head_params, backbone_params = [], []
        for n, p in model.named_parameters():
            (head_params if ('head' in n or 'fc' in n or 'classifier' in n) else backbone_params).append(p)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr},
            {'params': head_params,     'lr': args.lr * 10}
        ], weight_decay=0.05)

        steps_per_epoch = len(train_loader)
        scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs*steps_per_epoch, lr_min=1e-6, warmup_t=steps_per_epoch*2, warmup_lr_init=1e-6)
        scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        best_acc, best_state = 0.0, None
        step = 0

        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f'Fold {fold} Ep {epoch+1}/{args.epochs}', leave=False)
            for imgs, labels, _ in pbar:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                imgs, labels_mix = mixup_fn(imgs, labels)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                    logits = model(imgs)
                    loss = train_loss_fn(logits, labels_mix)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(step)
                step += 1

            model.eval()
            correct, total, vloss = 0, 0, 0.0
            with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    logits = model(imgs)
                    vloss += val_loss_fn(logits, labels).item() * imgs.size(0)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += imgs.size(0)

            acc = correct/total
            print(f'  Fold {fold} Ep {epoch+1}: val_loss={vloss/total:.4f} val_acc={acc:.4f}')

            mlflow.log_metrics({
                "val_loss": vloss/total,
                "val_acc":  acc,
            }, step=epoch)

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f'fold{fold}.pt')
        torch.save({
            'state_dict': best_state, 'val_acc': best_acc, 'model_name': args.model_name,
            'img_size': args.img_size, 'mean': cfg['mean'], 'std': cfg['std'], 'crop_pct': cfg.get('crop_pct', 0.95)
        }, ckpt_path)

        mlflow.set_tag("fold_best_acc", f"{best_acc:.4f}")
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoint")

        return best_acc, best_state, cfg


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcfg = load_mlflow_config()
    init_mlflow(mcfg)

    with mlflow.start_run(run_name=f"{args.model_name}-kfold{args.folds}"):
        mlflow.log_params(vars(args))
        mlflow.set_tag("script", "train_model2.py")
        mlflow.set_tag("data_rev", get_dvc_data_rev() or "unknown")

        # Use the data loader script to gather items dynamically
        items, classes = gather_kfold_items(args.data_dir)
        print(f'Total training images for CV: {len(items)} across {len(classes)} classes.')
        labels_arr = np.array([l for _, l in items])

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_results = []  # list of (acc, best_state, cfg)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(items, labels_arr)):
            tr = [items[i] for i in tr_idx]
            va = [items[i] for i in va_idx]
            print(f"\n--- Starting Fold {fold} ---")
            acc, best_state, cfg = train_one_fold(fold, tr, va, len(classes), args, device)
            fold_results.append((acc, best_state, cfg))

        accs = [r[0] for r in fold_results]
        cv_mean = float(np.mean(accs))
        cv_std  = float(np.std(accs))
        print(f'\nCV Mean Accuracy = {cv_mean:.4f}')
        mlflow.log_metrics({"cv_mean_acc": cv_mean, "cv_std_acc": cv_std})
        mlflow.set_tag("cv_mean_acc", f"{cv_mean:.4f}")

        # Register only the best fold's model
        best_acc, best_state, best_cfg = max(fold_results, key=lambda r: r[0])
        best_model = create_model(args.model_name, len(classes), drop_rate=0.1, drop_path_rate=0.1)
        best_model.load_state_dict(best_state)
        log_pytorch_best(
            best_model,
            registered_model_name=mcfg["registered_model_name"],
            class_names=classes,
            extra_meta={
                "model_name":   args.model_name,
                "img_size":     args.img_size,
                "best_fold_acc": best_acc,
                "cv_mean_acc":   cv_mean,
                "script":        "train_model2.py",
            },
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced K-Fold Vehicle Classification")
    parser.add_argument('--data_dir', type=str, default='data/processed/train')
    parser.add_argument('--out_dir', type=str, default='models/')
    parser.add_argument('--model_name', type=str, default='convnext_base_advanced')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--folds', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
