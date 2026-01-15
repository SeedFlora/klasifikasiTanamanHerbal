"""
Configuration file for Indonesian Herbal Plants Classification
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset" / "Indonesian Spices Dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories
for dir_path in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset config
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Reduced for CPU
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Training config
EPOCHS = 10  # Reduced for faster training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 3

# Model names
MODEL_NAMES = [
    "yolov11",           # YOLOv11 Classification
    "efficientnetv2",    # EfficientNetV2-S
    "convnextv2",        # ConvNeXt V2
    "internimage",       # InternImage - SOTA with deformable conv + global modeling
    "convformer"         # ConvFormer - Efficient CNN + Self-Attention
]

# Class names (will be populated from dataset)
CLASS_NAMES = []

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
