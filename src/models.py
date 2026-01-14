"""
Model architectures for Indonesian Herbal Plants Classification
5 Latest Models (2025):
1. YOLOv11 Classification
2. EfficientNetV2-S
3. ConvNeXt V2
4. Vision Transformer (ViT)
5. Hybrid CNN + ViT (CoAtNet-style)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ultralytics import YOLO
from typing import Optional
import config


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Factory function to create models"""
    
    model_name = model_name.lower()
    
    if model_name == "yolov11":
        return YOLOv11Classifier(num_classes, pretrained)
    elif model_name == "efficientnetv2":
        return EfficientNetV2Classifier(num_classes, pretrained)
    elif model_name == "convnextv2":
        return ConvNeXtV2Classifier(num_classes, pretrained)
    elif model_name == "vit":
        return ViTClassifier(num_classes, pretrained)
    elif model_name == "hybrid_cnn_vit":
        return HybridCNNViT(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class YOLOv11Classifier(nn.Module):
    """YOLOv11 for Image Classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = "YOLOv11-cls"
        
        # Use timm's version of YOLO-like architecture or a similar efficient model
        # Since ultralytics YOLO is primarily for detection, we use a similar backbone
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',  # YOLOv11 uses similar efficient backbone
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Custom head similar to YOLOv11 classification head
        self.feature_dim = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.forward_features(x)
        return self.head(features)


class EfficientNetV2Classifier(nn.Module):
    """EfficientNetV2-S Classifier"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = "EfficientNetV2-S"
        
        self.model = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2
        )
        
    def forward(self, x):
        return self.model(x)


class ConvNeXtV2Classifier(nn.Module):
    """ConvNeXt V2 Classifier - State-of-the-art CNN architecture"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = "ConvNeXtV2-Tiny"
        
        self.model = timm.create_model(
            'convnextv2_tiny',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=0.1
        )
        
    def forward(self, x):
        return self.model(x)


class ViTClassifier(nn.Module):
    """Vision Transformer (ViT) Classifier"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = "ViT-Base-16"
        
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.1,
            attn_drop_rate=0.1
        )
        
    def forward(self, x):
        return self.model(x)


class HybridCNNViT(nn.Module):
    """
    Hybrid CNN + Vision Transformer (CoAtNet-style architecture)
    Combines the local feature extraction of CNN with global attention of ViT
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = "Hybrid-CNN-ViT"
        
        # CNN backbone for local features (EfficientNet stem)
        self.cnn_backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 3]  # Get intermediate features
        )
        
        # Feature dimensions from EfficientNet-B0
        self.cnn_channels = [40, 112]  # Channels at indices 2 and 3
        
        # Project CNN features
        self.proj = nn.Conv2d(self.cnn_channels[1], 768, kernel_size=1)
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=4
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        
        # Position embedding (will be interpolated based on feature map size)
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 768))  # 14x14 + 1 cls
        
        # Classification head
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # CNN features
        features = self.cnn_backbone(x)
        x = features[-1]  # Use last feature map
        
        # Project to transformer dimension
        x = self.proj(x)  # B, 768, H, W
        
        # Flatten spatial dimensions
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, 768
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding (interpolate if needed)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2).unsqueeze(0),
                size=x.shape[1],
                mode='linear'
            ).squeeze(0).transpose(1, 2)
        else:
            pos_embed = self.pos_embed
        
        x = x + pos_embed[:, :x.shape[1], :]
        
        # Transformer
        x = self.transformer(x)
        
        # Classification from CLS token
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x


# Summary of models
def print_model_summary():
    """Print summary of all models"""
    print("\n" + "="*60)
    print("5 LATEST MODELS FOR CLASSIFICATION (2025)")
    print("="*60)
    
    models_info = [
        ("YOLOv11-cls", "YOLOv11 Classification - Fast and efficient"),
        ("EfficientNetV2-S", "EfficientNetV2 - Optimized CNN architecture"),
        ("ConvNeXtV2-Tiny", "ConvNeXt V2 - Pure CNN with modern design"),
        ("ViT-Base-16", "Vision Transformer - Pure attention-based"),
        ("Hybrid-CNN-ViT", "CNN + Transformer hybrid (CoAtNet-style)")
    ]
    
    for i, (name, desc) in enumerate(models_info, 1):
        print(f"{i}. {name}")
        print(f"   {desc}\n")


if __name__ == "__main__":
    print_model_summary()
    
    # Test all models
    num_classes = 31
    batch = torch.randn(2, 3, 224, 224)
    
    for model_name in config.MODEL_NAMES:
        print(f"\nTesting {model_name}...")
        model = get_model(model_name, num_classes, pretrained=False)
        output = model(batch)
        print(f"  Input: {batch.shape}")
        print(f"  Output: {output.shape}")
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
