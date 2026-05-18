import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- Model 1 Standard Loader ---
def get_dataloaders(data_dir, batch_size=32, img_size=224):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes


# --- Model 2 Custom K-Fold Dataset & Helpers ---
class CarDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert('RGB')), label, str(path)


def gather_kfold_items(train_dir):
    """Gathers all images and maps them to numerical labels based on subfolders."""
    classes = sorted(os.listdir(train_dir))
    cls2idx = {c: i for i, c in enumerate(classes)}
    
    items = []
    for c in classes:
        class_path = os.path.join(train_dir, c)
        if not os.path.isdir(class_path):
            continue
        for f in os.listdir(class_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                items.append((os.path.join(class_path, f), cls2idx[c]))
    return items, classes