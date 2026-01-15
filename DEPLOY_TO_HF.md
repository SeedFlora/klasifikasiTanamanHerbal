# 🚀 Panduan Deploy ke Hugging Face Spaces

## Prerequisites

1. **Akun Hugging Face**: Buat di [huggingface.co](https://huggingface.co/join)
2. **HF Token**: Get dari [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Opsi 1: Deploy via Web UI (Termudah)

### Step 1: Create New Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name**: `indonesian-herbal-classifier` (atau nama lain)
3. **License**: Apache 2.0
4. **SDK**: Gradio
5. **Hardware**: CPU basic (gratis) atau GPU (berbayar)
6. Click **Create Space**

### Step 2: Upload Files
Upload file-file berikut ke Space:

**Required Files:**
```
app.py                          # Entry point
requirements.txt                # Dependencies
README_HF.md → README.md        # Rename to README.md
src/app.py                      # Main Gradio app
src/config.py                   # Config
src/models.py                   # Model definitions
outputs/models/efficientnetv2.pth  # Best model (78MB)
outputs/training_results.json   # Results
outputs/class_names.json        # Class names
```

### Step 3: Upload Model Files
Model files yang perlu diupload (pilih salah satu atau semua):

1. **efficientnetv2.pth** (78MB) - **RECOMMENDED** (95.08% accuracy)
2. **yolov11.pth** (78MB) - 95.08% accuracy
3. **convformer.pth** (101MB) - 94.77% accuracy
4. **convnextv2.pth** (107MB) - 93.95% accuracy
5. **internimage.pth** (108MB) - 89.86% accuracy

💡 **Tip**: Upload minimal 1 model (efficientnetv2 atau yolov11)

---

## Opsi 2: Deploy via Git/CLI

### Step 1: Install Hugging Face CLI
```bash
pip install huggingface_hub
```

### Step 2: Login
```bash
huggingface-cli login
# Paste your HF token
```

### Step 3: Clone Space Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/indonesian-herbal-classifier
cd indonesian-herbal-classifier
```

### Step 4: Setup Git LFS for Large Files
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

### Step 5: Copy Files
Copy dari project folder ke space folder:
```bash
# Required files
cp app.py indonesian-herbal-classifier/
cp requirements.txt indonesian-herbal-classifier/
cp README_HF.md indonesian-herbal-classifier/README.md

# Source code
cp -r src indonesian-herbal-classifier/

# Outputs (model & results)
mkdir -p indonesian-herbal-classifier/outputs/models
cp outputs/models/efficientnetv2.pth indonesian-herbal-classifier/outputs/models/
cp outputs/training_results.json indonesian-herbal-classifier/outputs/
cp outputs/class_names.json indonesian-herbal-classifier/outputs/
```

### Step 6: Push to Hugging Face
```bash
cd indonesian-herbal-classifier
git add .
git commit -m "Initial commit: Indonesian Herbal Plants Classifier"
git push
```

---

## Opsi 3: Deploy via Python Script

Buat script `deploy_to_hf.py`:

```python
from huggingface_hub import HfApi, create_repo, upload_folder

# Config
HF_USERNAME = "YOUR_USERNAME"  # Ganti dengan username HF Anda
SPACE_NAME = "indonesian-herbal-classifier"
HF_TOKEN = "YOUR_TOKEN"  # Ganti dengan HF token Anda

# Create space
api = HfApi()
try:
    create_repo(
        repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
        token=HF_TOKEN,
        repo_type="space",
        space_sdk="gradio",
        private=False
    )
    print(f"✅ Space created: {HF_USERNAME}/{SPACE_NAME}")
except Exception as e:
    print(f"Space already exists or error: {e}")

# Upload files
upload_folder(
    folder_path=".",
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    token=HF_TOKEN,
    ignore_patterns=["*.git*", "__pycache__", ".venv", "dataset"],
)

print(f"✅ Deployed to: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
```

Run:
```bash
python deploy_to_hf.py
```

---

## File Structure untuk HF Space

```
indonesian-herbal-classifier/
├── app.py                      # Entry point
├── requirements.txt            # Dependencies
├── README.md                   # From README_HF.md
├── src/
│   ├── app.py                 # Gradio interface
│   ├── config.py              # Configuration
│   └── models.py              # Model architectures
└── outputs/
    ├── models/
    │   ├── efficientnetv2.pth # Best model (REQUIRED)
    │   ├── yolov11.pth        # Optional
    │   └── ...                # Other models (optional)
    ├── training_results.json  # Training results
    └── class_names.json       # Class names
```

---

## Testing Locally Before Deploy

Test app locally:
```bash
cd "d:\skripsi tanaman herbal"
python app.py
```

Open browser: http://localhost:7860

---

## Troubleshooting

### 1. Model file too large
- Upload via Git LFS
- Or use only 1 model (efficientnetv2.pth recommended)

### 2. Import errors
Make sure `requirements.txt` has all dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
gradio>=4.0.0
pillow>=10.0.0
numpy>=1.24.0
```

### 3. Path errors
Update `src/config.py` paths untuk HF Space:
```python
BASE_DIR = Path(__file__).parent.parent
```

### 4. Missing class_names.json
Create manually if needed:
```json
["adas", "andaliman", "asam jawa", ...]
```

---

## After Deployment

Your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/indonesian-herbal-classifier
```

You can:
- Share the link
- Embed in website
- Use API endpoint
- Monitor usage

---

## Recommendations

✅ **For Best Performance:**
- Use GPU hardware (T4 small - $0.60/hour)
- Upload efficientnetv2.pth (best accuracy + small size)
- Enable persistent storage for caching

🎯 **For Free Tier:**
- Use CPU basic (free)
- Upload only 1 model (efficientnetv2)
- May have startup delay

---

## Need Help?

- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://gradio.app/docs/
- Community: https://discuss.huggingface.co/

Happy Deploying! 🚀🌿
