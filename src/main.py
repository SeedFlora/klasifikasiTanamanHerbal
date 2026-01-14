"""
Main script to run the complete pipeline
Training -> Evaluation -> Visualization -> Gradio -> Hugging Face
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from dataset import create_data_loaders
from trainer import train_all_models
from evaluator import evaluate_all_models


def main():
    """Run the complete training and evaluation pipeline"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   🌿 INDONESIAN HERBAL PLANTS CLASSIFICATION 🌿                   ║
    ║                                                                  ║
    ║   5 State-of-the-Art Deep Learning Models (2025)                 ║
    ║   - YOLOv11 Classification                                       ║
    ║   - EfficientNetV2-S                                             ║
    ║   - ConvNeXt V2                                                  ║
    ║   - Vision Transformer (ViT)                                     ║
    ║   - Hybrid CNN + ViT (CoAtNet-style)                             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n📁 Base Directory: {config.BASE_DIR}")
    print(f"📁 Data Directory: {config.DATA_DIR}")
    print(f"📁 Output Directory: {config.OUTPUT_DIR}")
    print(f"🖥️  Device: {config.DEVICE}")
    
    # Step 1: Train all models
    print("\n" + "="*70)
    print("STEP 1: TRAINING ALL MODELS")
    print("="*70)
    
    training_results, test_loader, class_names = train_all_models()
    
    # Step 2: Evaluate all models
    print("\n" + "="*70)
    print("STEP 2: EVALUATING ALL MODELS")
    print("="*70)
    
    all_metrics = evaluate_all_models(test_loader, class_names, training_results)
    
    # Step 3: Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED!")
    print("="*70)
    print(f"\n📊 Results saved to: {config.OUTPUT_DIR}")
    print(f"📈 Plots saved to: {config.PLOTS_DIR}")
    print(f"🤖 Models saved to: {config.MODELS_DIR}")
    
    print("\n📝 Next Steps:")
    print("  1. Run Gradio interface:")
    print(f"     python {config.BASE_DIR}/src/app.py")
    print("\n  2. Push to Hugging Face:")
    print(f"     python {config.BASE_DIR}/src/huggingface_upload.py --username YOUR_USERNAME --token YOUR_TOKEN")
    
    return training_results, all_metrics


if __name__ == "__main__":
    training_results, all_metrics = main()
