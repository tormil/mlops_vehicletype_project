import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from model import create_model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size, args.img_size)
    num_classes = len(class_names)

    # 2. Initialize Model via Factory
    model = create_model(args.model_name, num_classes)
    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    print(f"Starting Training for {args.epochs} epochs with {args.model_name}...")

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        # --- VALIDATION ---
        model.eval()
        val_running_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val

        scheduler.step()

        # Save local checkpoint if validation accuracy improves
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = f"best_{args.model_name}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"-> Saved new best model checkpoint to {checkpoint_path}")

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vehicle Classification Model")
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--model_name', type=str, default='swin_tiny', choices=['swin_tiny', 'convnext_tiny'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=2) 
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    train(args)