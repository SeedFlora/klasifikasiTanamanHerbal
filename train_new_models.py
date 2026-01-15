"""
Script untuk training InternImage dan ConvFormer
Tambahan model SOTA untuk klasifikasi tanaman herbal Indonesia
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import json
from src.dataset import create_data_loaders
from src.trainer import Trainer
from src.models import get_model
import src.config as config

def train_new_models():
    """Train InternImage and ConvFormer models"""

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   🌿 TRAINING NEW SOTA MODELS 🌿                                  ║
    ║                                                                  ║
    ║   - InternImage: Deformable Conv + Global Modeling              ║
    ║   - ConvFormer: Efficient CNN + Self-Attention                   ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Create data loaders
    print("📊 Loading dataset...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders()
    print(f"   Classes: {len(class_names)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Models to train
    new_models = ["internimage", "convformer"]

    # Load existing results
    results_path = config.OUTPUT_DIR / "training_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Train each model
    for model_name in new_models:
        print("\n" + "="*70)
        print(f"TRAINING: {model_name.upper()}")
        print("="*70)

        # Create model
        model = get_model(model_name, len(class_names), pretrained=True)

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
            num_classes=len(class_names)
        )

        # Train
        history = trainer.train(epochs=config.EPOCHS)

        # Save model
        model_path = trainer.save_model()

        # Save results
        all_results[model_name] = {
            'best_val_acc': history['best_val_acc'],
            'training_time': history['training_time'],
            'params': params,
            'model_path': str(model_path)
        }

        # Save updated results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ {model_name} completed!")
        print(f"   Best Val Acc: {history['best_val_acc']:.2f}%")
        print(f"   Training Time: {history['training_time']:.1f}s")
        print(f"   Parameters: {params:,}")

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED - ALL MODELS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'Val Acc':<12} {'Time (s)':<12} {'Params':<15}")
    print("-" * 70)

    # Sort by accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True)

    for model_name, metrics in sorted_results:
        print(f"{model_name:<20} {metrics['best_val_acc']:>10.2f}%  "
              f"{metrics['training_time']:>10.1f}s  {metrics['params']:>13,}")

    print("\n📊 Results saved to:", results_path)
    print("\n🎯 Next Steps:")
    print("   1. Evaluate models: python src/evaluator.py")
    print("   2. Run Gradio app: python src/app.py")

    return all_results

if __name__ == "__main__":
    results = train_new_models()
