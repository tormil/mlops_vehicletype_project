import os
import shutil
import random
import argparse

def split_data(raw_dir, processed_dir, val_images_per_class, n_per_class=-1, n_test=-1, seed=42):
    random.seed(seed)

    raw_train_dir = os.path.join(raw_dir, 'train')
    raw_test_dir = os.path.join(raw_dir, 'test')

    proc_train_dir = os.path.join(processed_dir, 'train')
    proc_val_dir = os.path.join(processed_dir, 'val')
    proc_test_dir = os.path.join(processed_dir, 'test')

    print("--- Creating Processed Train/Val/Test Splits ---")
    if n_per_class > 0:
        print(f"   Demo subset: sampling {n_per_class} images per class (val_per_class={val_images_per_class}).")

    # Clear old processed data if it exists to prevent mixing
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    os.makedirs(proc_train_dir, exist_ok=True)
    os.makedirs(proc_val_dir, exist_ok=True)

    # 1. Handle the Test Set (flat directory, optionally subsampled)
    if os.path.exists(raw_test_dir):
        os.makedirs(proc_test_dir, exist_ok=True)
        test_files = [f for f in os.listdir(raw_test_dir) if os.path.isfile(os.path.join(raw_test_dir, f))]
        if n_test > 0 and len(test_files) > n_test:
            test_files = random.sample(test_files, n_test)
        for fname in test_files:
            shutil.copy(os.path.join(raw_test_dir, fname), os.path.join(proc_test_dir, fname))
        print(f"✔️ Test set: copied {len(test_files)} files to processed directory.")

    # 2. Handle Train & Validation Split
    classes = [d for d in os.listdir(raw_train_dir) if os.path.isdir(os.path.join(raw_train_dir, d))]

    for class_name in classes:
        src_class_dir = os.path.join(raw_train_dir, class_name)
        dst_train_class = os.path.join(proc_train_dir, class_name)
        dst_val_class = os.path.join(proc_val_dir, class_name)

        os.makedirs(dst_train_class, exist_ok=True)
        os.makedirs(dst_val_class, exist_ok=True)

        # Get all images for this vehicle class
        images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        # Optional demo subsample before splitting
        if n_per_class > 0 and len(images) > n_per_class:
            images = random.sample(images, n_per_class)

        if len(images) > val_images_per_class:
            # Randomly pick images for validation
            val_images = set(random.sample(images, val_images_per_class))

            for img in images:
                src_path = os.path.join(src_class_dir, img)
                if img in val_images:
                    shutil.copy(src_path, os.path.join(dst_val_class, img))
                else:
                    shutil.copy(src_path, os.path.join(dst_train_class, img))

            print(f"✔️ Class '{class_name}': Copied {len(images) - val_images_per_class} to train, {val_images_per_class} to val.")
        else:
            print(f"⚠️ Warning: Not enough images in {class_name} to create split.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Data Preparation and Splitting")
    parser.add_argument('--raw_dir', type=str, default='data/raw')
    parser.add_argument('--processed_dir', type=str, default='data/processed')
    parser.add_argument('--val_per_class', type=int, default=10)
    parser.add_argument('--n_per_class', type=int, default=-1,
                        help="If >0, sample this many images per class before the train/val split (demo mode).")
    parser.add_argument('--n_test', type=int, default=-1,
                        help="If >0, sample this many files from the flat raw test directory.")

    args = parser.parse_args()
    split_data(args.raw_dir, args.processed_dir, args.val_per_class, args.n_per_class, args.n_test)