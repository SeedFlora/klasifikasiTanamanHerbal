"""
Training and Evaluation Pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from pathlib import Path

import config
from models import get_model
from dataset import create_data_loaders


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class Trainer:
    """Model trainer with mixed precision and various optimizations"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader,
        val_loader,
        num_classes: int,
        device: str = config.DEVICE
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer - AdamW with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE * 10,
            epochs=config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs: int = config.EPOCHS) -> Dict:
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f"  *** New best model! ***")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Add training time to history
        self.history['training_time'] = training_time
        self.history['best_val_acc'] = self.best_val_acc
        
        return self.history
    
    def save_model(self, path: Optional[Path] = None):
        """Save the trained model"""
        if path is None:
            path = config.MODELS_DIR / f"{self.model_name.lower().replace(' ', '_')}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, path)
        
        print(f"Model saved to {path}")
        return path


def train_all_models():
    """Train all 5 models and return results"""
    print("\n" + "="*70)
    print("TRAINING 5 MODELS FOR INDONESIAN HERBAL PLANTS CLASSIFICATION")
    print("="*70)
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders()
    num_classes = len(class_names)
    
    # Save class names
    with open(config.OUTPUT_DIR / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    
    results = {}
    
    for model_name in config.MODEL_NAMES:
        print(f"\n{'#'*70}")
        print(f"# Model: {model_name.upper()}")
        print(f"{'#'*70}")
        
        # Create model
        model = get_model(model_name, num_classes, pretrained=True)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes
        )
        
        # Train
        history = trainer.train(epochs=config.EPOCHS)
        
        # Save model
        model_path = trainer.save_model()
        
        # Store results
        results[model_name] = {
            'history': history,
            'model_path': str(model_path),
            'params': params,
            'trainable_params': trainable_params
        }
    
    # Save results summary
    with open(config.OUTPUT_DIR / "training_results.json", 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for name, data in results.items():
            serializable_results[name] = {
                'best_val_acc': data['history']['best_val_acc'],
                'training_time': data['history']['training_time'],
                'params': data['params'],
                'model_path': data['model_path']
            }
        json.dump(serializable_results, f, indent=2)
    
    return results, test_loader, class_names


if __name__ == "__main__":
    results, test_loader, class_names = train_all_models()
