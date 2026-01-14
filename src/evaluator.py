"""
Evaluation metrics and visualization
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import pandas as pd
from pathlib import Path

import config
from models import get_model

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Evaluator:
    """Model evaluator with comprehensive metrics"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        test_loader,
        class_names: List[str],
        device: str = config.DEVICE
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.test_loader = test_loader
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device
        
        self.model.eval()
    
    @torch.no_grad()
    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions, true labels, and probabilities"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(self.test_loader, desc=f"Evaluating {self.model_name}"):
            images = images.to(self.device)
            
            with autocast():
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
            
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def calculate_metrics(self) -> Dict:
        """Calculate all evaluation metrics"""
        preds, labels, probs = self.get_predictions()
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds) * 100
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0) * 100
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0) * 100
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0) * 100
        
        precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0) * 100
        recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0) * 100
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0) * 100
        
        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0) * 100
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0) * 100
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0) * 100
        
        # ROC AUC (multi-class)
        labels_bin = label_binarize(labels, classes=range(self.num_classes))
        try:
            auc_macro = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr') * 100
            auc_weighted = roc_auc_score(labels_bin, probs, average='weighted', multi_class='ovr') * 100
        except:
            auc_macro = 0.0
            auc_weighted = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        metrics = {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'auc_roc_macro': auc_macro,
            'auc_roc_weighted': auc_weighted,
            'confusion_matrix': cm,
            'predictions': preds,
            'labels': labels,
            'probabilities': probs,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS: {metrics['model_name']}")
        print(f"{'='*60}")
        print(f"Accuracy:           {metrics['accuracy']:.2f}%")
        print(f"Precision (macro):  {metrics['precision_macro']:.2f}%")
        print(f"Recall (macro):     {metrics['recall_macro']:.2f}%")
        print(f"F1-Score (macro):   {metrics['f1_macro']:.2f}%")
        print(f"AUC-ROC (macro):    {metrics['auc_roc_macro']:.2f}%")
        print(f"-" * 40)
        print(f"Precision (weighted): {metrics['precision_weighted']:.2f}%")
        print(f"Recall (weighted):    {metrics['recall_weighted']:.2f}%")
        print(f"F1-Score (weighted):  {metrics['f1_weighted']:.2f}%")
        print(f"AUC-ROC (weighted):   {metrics['auc_roc_weighted']:.2f}%")


def plot_confusion_matrix(metrics: Dict, class_names: List[str], save_path: Path):
    """Plot and save confusion matrix"""
    cm = metrics['confusion_matrix']
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 16))
    
    # Plot normalized confusion matrix
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.1%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title(f'Confusion Matrix - {metrics["model_name"]}\nAccuracy: {metrics["accuracy"]:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(metrics: Dict, class_names: List[str], save_path: Path):
    """Plot ROC curves for all classes"""
    labels = metrics['labels']
    probs = metrics['probabilities']
    num_classes = len(class_names)
    
    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(num_classes))
    
    plt.figure(figsize=(14, 10))
    
    # Plot ROC curve for each class
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=1.5, alpha=0.7,
                label=f'{class_names[i]} (AUC={roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {metrics["model_name"]}\nMacro AUC: {metrics["auc_roc_macro"]:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")


def plot_training_history(history: Dict, model_name: str, save_path: Path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Text summary
    axes[1, 1].axis('off')
    summary_text = f"""
    Model: {model_name}
    
    Training Summary:
    ─────────────────────────
    Best Val Accuracy: {history['best_val_acc']:.2f}%
    Training Time: {history['training_time']/60:.2f} min
    Total Epochs: {len(epochs)}
    Final Train Loss: {history['train_loss'][-1]:.4f}
    Final Val Loss: {history['val_loss'][-1]:.4f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f'Training History - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_model_comparison(all_metrics: List[Dict], save_path: Path):
    """Plot comparison of all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = [m['model_name'] for m in all_metrics]
    
    # Metrics for comparison
    metrics_to_compare = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_roc_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Bar chart comparison
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
        values = [m[metric] for m in all_metrics]
        axes[0, 0].bar(x + i * width, values, width, label=label)
    
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Score (%)')
    axes[0, 0].set_title('Model Comparison - All Metrics')
    axes[0, 0].set_xticks(x + width * 2)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 105])
    
    # Accuracy comparison (horizontal bar)
    accuracies = [m['accuracy'] for m in all_metrics]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    bars = axes[0, 1].barh(model_names, accuracies, color=colors)
    axes[0, 1].set_xlabel('Accuracy (%)')
    axes[0, 1].set_title('Model Accuracy Comparison')
    axes[0, 1].set_xlim([0, 105])
    for bar, acc in zip(bars, accuracies):
        axes[0, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'{acc:.2f}%', va='center', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # F1-Score comparison
    f1_scores = [m['f1_macro'] for m in all_metrics]
    bars = axes[1, 0].barh(model_names, f1_scores, color=colors)
    axes[1, 0].set_xlabel('F1-Score (%)')
    axes[1, 0].set_title('Model F1-Score Comparison (Macro)')
    axes[1, 0].set_xlim([0, 105])
    for bar, f1 in zip(bars, f1_scores):
        axes[1, 0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{f1:.2f}%', va='center', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # AUC-ROC comparison
    auc_scores = [m['auc_roc_macro'] for m in all_metrics]
    bars = axes[1, 1].barh(model_names, auc_scores, color=colors)
    axes[1, 1].set_xlabel('AUC-ROC (%)')
    axes[1, 1].set_title('Model AUC-ROC Comparison (Macro)')
    axes[1, 1].set_xlim([0, 105])
    for bar, auc_val in zip(bars, auc_scores):
        axes[1, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{auc_val:.2f}%', va='center', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Model Performance Comparison\nIndonesian Herbal Plants Classification', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison saved to {save_path}")


def plot_per_class_metrics(all_metrics: List[Dict], class_names: List[str], save_path: Path):
    """Plot per-class F1 scores for all models"""
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    
    model_names = [m['model_name'] for m in all_metrics]
    x = np.arange(len(class_names))
    width = 0.15
    
    for i, metrics in enumerate(all_metrics):
        f1_per_class = metrics['f1_per_class']
        axes.bar(x + i * width, f1_per_class, width, label=metrics['model_name'], alpha=0.8)
    
    axes.set_xlabel('Class', fontsize=12)
    axes.set_ylabel('F1-Score (%)', fontsize=12)
    axes.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    axes.set_xticks(x + width * 2)
    axes.set_xticklabels(class_names, rotation=45, ha='right')
    axes.legend()
    axes.grid(True, alpha=0.3, axis='y')
    axes.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics saved to {save_path}")


def create_results_table(all_metrics: List[Dict], save_path: Path):
    """Create and save results table"""
    data = []
    for m in all_metrics:
        data.append({
            'Model': m['model_name'],
            'Accuracy (%)': f"{m['accuracy']:.2f}",
            'Precision (%)': f"{m['precision_macro']:.2f}",
            'Recall (%)': f"{m['recall_macro']:.2f}",
            'F1-Score (%)': f"{m['f1_macro']:.2f}",
            'AUC-ROC (%)': f"{m['auc_roc_macro']:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(save_path.with_suffix('.csv'), index=False)
    
    # Create table image
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4CAF50'] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Evaluation Results Summary\nIndonesian Herbal Plants Classification',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(save_path.with_suffix('.png'), dpi=150, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Results table saved to {save_path}")
    return df


def evaluate_all_models(test_loader, class_names: List[str], training_results: Dict = None):
    """Evaluate all trained models"""
    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70)
    
    all_metrics = []
    
    for model_name in config.MODEL_NAMES:
        print(f"\nLoading {model_name}...")
        
        # Load model
        model_path = config.MODELS_DIR / f"{model_name.lower()}.pth"
        
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            continue
        
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model = get_model(model_name, len(class_names), pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = Evaluator(model, model_name, test_loader, class_names)
        metrics = evaluator.calculate_metrics()
        evaluator.print_metrics(metrics)
        
        all_metrics.append(metrics)
        
        # Plot confusion matrix
        cm_path = config.PLOTS_DIR / f"confusion_matrix_{model_name.lower()}.png"
        plot_confusion_matrix(metrics, class_names, cm_path)
        
        # Plot ROC curves
        roc_path = config.PLOTS_DIR / f"roc_curves_{model_name.lower()}.png"
        plot_roc_curves(metrics, class_names, roc_path)
        
        # Plot training history if available
        if training_results and model_name in training_results:
            history = training_results[model_name]['history']
            history_path = config.PLOTS_DIR / f"training_history_{model_name.lower()}.png"
            plot_training_history(history, model_name, history_path)
    
    if len(all_metrics) > 0:
        # Plot model comparison
        comparison_path = config.PLOTS_DIR / "model_comparison.png"
        plot_model_comparison(all_metrics, comparison_path)
        
        # Plot per-class metrics
        per_class_path = config.PLOTS_DIR / "per_class_f1_comparison.png"
        plot_per_class_metrics(all_metrics, class_names, per_class_path)
        
        # Create results table
        table_path = config.PLOTS_DIR / "results_table"
        results_df = create_results_table(all_metrics, table_path)
        
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Find best model
        best_idx = np.argmax([m['accuracy'] for m in all_metrics])
        best_model = all_metrics[best_idx]
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"   Accuracy: {best_model['accuracy']:.2f}%")
        print(f"   F1-Score: {best_model['f1_macro']:.2f}%")
        print(f"   AUC-ROC:  {best_model['auc_roc_macro']:.2f}%")
    
    return all_metrics


if __name__ == "__main__":
    from dataset import create_data_loaders
    
    _, _, test_loader, class_names = create_data_loaders()
    all_metrics = evaluate_all_models(test_loader, class_names)
