"""
Hugging Face Hub Integration
Push models and create model cards
"""
import os
import json
import shutil
from pathlib import Path
from huggingface_hub import (
    HfApi, 
    create_repo, 
    upload_folder,
    login
)
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent))
import config


def create_model_card(model_name: str, metrics: dict, class_names: list) -> str:
    """Create a model card for Hugging Face"""
    
    card_content = f"""---
license: apache-2.0
tags:
  - image-classification
  - pytorch
  - indonesian
  - herbal-plants
  - computer-vision
datasets:
  - custom
language:
  - id
  - en
metrics:
  - accuracy
  - f1
  - precision
  - recall
pipeline_tag: image-classification
---

# Indonesian Herbal Plants Classification - {model_name}

## Model Description

This model classifies **31 types of Indonesian herbal plants and spices** commonly used in traditional medicine (Jamu) and cooking.

### Model Architecture
- **Architecture**: {model_name}
- **Framework**: PyTorch
- **Input Size**: 224x224 pixels
- **Number of Classes**: 31

### Training Details
- **Dataset**: Indonesian Spices Dataset
- **Total Images**: 6,510 (210 per class)
- **Train/Val/Test Split**: 70%/15%/15%
- **Epochs**: {config.EPOCHS}
- **Optimizer**: AdamW
- **Learning Rate**: {config.LEARNING_RATE}
- **Data Augmentation**: Yes (RandomCrop, Flip, Rotation, ColorJitter, etc.)

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 'N/A'):.2f}% |
| Precision (macro) | {metrics.get('precision_macro', 'N/A'):.2f}% |
| Recall (macro) | {metrics.get('recall_macro', 'N/A'):.2f}% |
| F1-Score (macro) | {metrics.get('f1_macro', 'N/A'):.2f}% |
| AUC-ROC (macro) | {metrics.get('auc_roc_macro', 'N/A'):.2f}% |

## Classes

The model can classify the following 31 Indonesian herbal plants:

{chr(10).join([f'{i+1}. **{name}**' for i, name in enumerate(class_names)])}

## Usage

### Using with PyTorch

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load('model.pth', map_location='cpu')
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('your_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

print(f'Predicted class: {{class_names[predicted_class]}}')
```

### Using with Gradio

```python
import gradio as gr
# See the app.py file for full Gradio implementation
```

## Intended Use

This model is intended for:
- Educational purposes
- Research on Indonesian herbal plants
- Development of herbal plant identification applications
- Supporting traditional medicine documentation

## Limitations

- Model performance may vary with images taken under different lighting conditions
- Works best with clear, focused images of the plant/spice
- May not perform well on processed or mixed spices

## Training Data

The model was trained on the [Indonesian Spices Dataset](https://www.kaggle.com/datasets/albertnathaniel12/indonesian-spices-dataset) from Kaggle.

## Citation

If you use this model, please cite:

```bibtex
@misc{{indonesian-herbal-plants-classification,
  title={{Indonesian Herbal Plants Classification}},
  author={{Your Name}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/your-username/indonesian-herbal-plants}}
}}
```

## License

This model is released under the Apache 2.0 License.
"""
    
    return card_content


def prepare_upload_folder(model_name: str) -> Path:
    """Prepare folder for upload to Hugging Face"""
    
    upload_dir = config.OUTPUT_DIR / "hf_upload" / model_name.lower()
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model file
    model_path = config.MODELS_DIR / f"{model_name.lower()}.pth"
    if model_path.exists():
        shutil.copy(model_path, upload_dir / "model.pth")
    
    # Copy class names
    class_names_path = config.OUTPUT_DIR / "class_names.json"
    if class_names_path.exists():
        shutil.copy(class_names_path, upload_dir / "class_names.json")
    
    # Copy config
    shutil.copy(config.BASE_DIR / "src" / "config.py", upload_dir / "config.py")
    
    # Copy relevant plots
    plots_to_copy = [
        f"confusion_matrix_{model_name.lower()}.png",
        f"roc_curves_{model_name.lower()}.png",
        f"training_history_{model_name.lower()}.png"
    ]
    
    for plot_name in plots_to_copy:
        plot_path = config.PLOTS_DIR / plot_name
        if plot_path.exists():
            shutil.copy(plot_path, upload_dir / plot_name)
    
    return upload_dir


def push_to_huggingface(
    model_name: str,
    repo_name: str,
    metrics: dict,
    class_names: list,
    token: str = None,
    private: bool = False
):
    """Push model to Hugging Face Hub"""
    
    print(f"\n{'='*60}")
    print(f"Pushing {model_name} to Hugging Face Hub")
    print(f"{'='*60}")
    
    # Login
    if token:
        login(token=token)
    else:
        print("Please login to Hugging Face first using `huggingface-cli login`")
    
    api = HfApi()
    
    # Create repo
    try:
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"Repository created/found: {repo_name}")
    except Exception as e:
        print(f"Error creating repo: {e}")
    
    # Prepare upload folder
    upload_dir = prepare_upload_folder(model_name)
    
    # Create model card
    model_card = create_model_card(model_name, metrics, class_names)
    with open(upload_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    # Create requirements.txt
    requirements = """torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
Pillow>=9.0.0
numpy>=1.23.0
"""
    with open(upload_dir / "requirements.txt", 'w') as f:
        f.write(requirements)
    
    # Upload
    try:
        upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_name,
            commit_message=f"Upload {model_name} model for Indonesian Herbal Plants Classification"
        )
        print(f"✅ Successfully uploaded to https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"❌ Error uploading: {e}")
    
    return repo_name


def push_all_models(username: str, token: str = None):
    """Push all trained models to Hugging Face"""
    
    # Load class names
    class_names_path = config.OUTPUT_DIR / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = []
    
    # Load evaluation results if available
    results_path = config.PLOTS_DIR / "results_table.csv"
    metrics_dict = {}
    
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path)
        for _, row in df.iterrows():
            metrics_dict[row['Model']] = {
                'accuracy': float(row['Accuracy (%)'].replace('%', '')),
                'precision_macro': float(row['Precision (%)'].replace('%', '')),
                'recall_macro': float(row['Recall (%)'].replace('%', '')),
                'f1_macro': float(row['F1-Score (%)'].replace('%', '')),
                'auc_roc_macro': float(row['AUC-ROC (%)'].replace('%', ''))
            }
    
    repos = []
    
    for model_name in config.MODEL_NAMES:
        model_path = config.MODELS_DIR / f"{model_name.lower()}.pth"
        
        if not model_path.exists():
            print(f"Skipping {model_name} - model not found")
            continue
        
        repo_name = f"{username}/indonesian-herbal-plants-{model_name.lower()}"
        metrics = metrics_dict.get(model_name, {})
        
        try:
            push_to_huggingface(
                model_name=model_name,
                repo_name=repo_name,
                metrics=metrics,
                class_names=class_names,
                token=token
            )
            repos.append(repo_name)
        except Exception as e:
            print(f"Error pushing {model_name}: {e}")
    
    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully uploaded {len(repos)} models:")
    for repo in repos:
        print(f"  - https://huggingface.co/{repo}")
    
    return repos


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push models to Hugging Face Hub")
    parser.add_argument("--username", type=str, required=True, help="Hugging Face username")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--model", type=str, default=None, help="Specific model to push (or 'all')")
    
    args = parser.parse_args()
    
    if args.model and args.model.lower() != 'all':
        # Push single model
        class_names_path = config.OUTPUT_DIR / "class_names.json"
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        push_to_huggingface(
            model_name=args.model,
            repo_name=f"{args.username}/indonesian-herbal-plants-{args.model.lower()}",
            metrics={},
            class_names=class_names,
            token=args.token
        )
    else:
        # Push all models
        push_all_models(args.username, args.token)
