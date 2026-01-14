"""
Data loading and preprocessing utilities
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import config

class HerbalPlantsDataset(Dataset):
    """Custom Dataset for Indonesian Herbal Plants"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """Get data transforms for training or validation/test"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_dataset() -> Tuple[List[str], List[int], List[str]]:
    """Load all image paths and labels from dataset directory"""
    
    image_paths = []
    labels = []
    class_names = sorted([d.name for d in config.DATA_DIR.iterdir() if d.is_dir()])
    
    print(f"Found {len(class_names)} classes:")
    for idx, class_name in enumerate(class_names):
        class_dir = config.DATA_DIR / class_name
        class_images = list(class_dir.glob("*.[jJ][pP][gG]")) + \
                       list(class_dir.glob("*.[jJ][pP][eE][gG]")) + \
                       list(class_dir.glob("*.[pP][nN][gG]"))
        
        print(f"  [{idx:2d}] {class_name}: {len(class_images)} images")
        
        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(idx)
    
    # Update global class names
    config.CLASS_NAMES = class_names
    
    return image_paths, labels, class_names


def create_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test data loaders"""
    
    # Load dataset
    image_paths, labels, class_names = load_dataset()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, 
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=labels,
        random_state=config.RANDOM_SEED
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=y_temp,
        random_state=config.RANDOM_SEED
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} images")
    print(f"  Val:   {len(X_val)} images")
    print(f"  Test:  {len(X_test)} images")
    
    # Create datasets
    train_dataset = HerbalPlantsDataset(X_train, y_train, transform=get_transforms(is_training=True))
    val_dataset = HerbalPlantsDataset(X_val, y_val, transform=get_transforms(is_training=False))
    test_dataset = HerbalPlantsDataset(X_test, y_test, transform=get_transforms(is_training=False))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, class_names = create_data_loaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
