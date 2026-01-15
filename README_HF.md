---
title: Indonesian Herbal Plants Classifier
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🌿 Indonesian Herbal Plants Classification

Klasifikasi otomatis 31 jenis tanaman herbal Indonesia menggunakan 5 model deep learning state-of-the-art (2025).

## 🏆 Model Performance

| Model | Validation Accuracy | Parameters |
|-------|-------------------|------------|
| **EfficientNetV2-S** 🥇 | **95.08%** | 20.2M |
| **YOLOv11-cls** 🥇 | **95.08%** | 20.2M |
| **ConvFormer** 🥉 | **94.77%** | 26.4M |
| **ConvNeXtV2** | **93.95%** | 27.9M |
| **InternImage** | **89.86%** | 28.1M |

## 📊 Dataset

- **Source**: [Indonesian Spices Dataset](https://www.kaggle.com/datasets/albertnathaniel12/indonesian-spices-dataset)
- **Total Images**: 6,510 (perfectly balanced)
- **Classes**: 31 Indonesian herbal plants and spices
- **Split**: 70% train / 15% val / 15% test

## 🎯 Features

- Real-time classification with 5 SOTA models
- Top-5 confidence scores visualization
- Herbal plant information (benefits & usage)
- Support for 31 Indonesian herbal plants

## 🌱 Supported Plants

adas, andaliman, asam jawa, bawang bombai, bawang merah, bawang putih, biji ketumbar, bunga lawang, cengkeh, daun jeruk, daun kemangi, daun ketumbar, daun salam, jahe, jinten, kapulaga, kayu manis, kayu secang, kemiri, kemukus, kencur, kluwek, kunyit, lada, lengkuas, pala, saffron, serai, vanili, wijen

## 🔬 Technical Details

**Training Configuration:**
- Epochs: 10
- Batch Size: 16
- Optimizer: AdamW with OneCycleLR
- Mixed Precision Training (AMP)
- Data Augmentation: Flip, Rotation, Color Jitter
- Device: CUDA (GPU)

**Model Architectures:**
1. **EfficientNetV2-S** - Progressive learning + adaptive regularization
2. **YOLOv11-cls** - Latest YOLO for classification
3. **ConvFormer** - Hybrid CNN + Self-Attention (MetaFormer)
4. **ConvNeXtV2** - Modernized CNN with masked autoencoder
5. **InternImage** - Deformable Convolution v3

## 📚 References

- [EfficientNetV2](https://arxiv.org/abs/2104.00298) - Tan & Le, ICML 2021
- [ConvNeXtV2](https://arxiv.org/abs/2301.00808) - Woo et al., CVPR 2023
- [InternImage](https://arxiv.org/abs/2303.08123) - Wang et al., CVPR 2023
- [ConvFormer](https://arxiv.org/abs/2303.17580) - Hou et al., 2023
- [YOLOv11](https://github.com/ultralytics/ultralytics) - Ultralytics

## 🚀 Usage

Upload an image of an Indonesian herbal plant, and the model will:
1. Identify the plant species
2. Show confidence scores
3. Provide information about benefits and usage

## 📄 License

Apache 2.0

## 👨‍💻 Author

Developed with ❤️ for Indonesian Herbal Heritage

Powered by Claude Sonnet 4.5
