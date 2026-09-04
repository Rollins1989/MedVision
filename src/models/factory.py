"""
src/models/factory.py
----------------------
Builds any of the four compared backbones with a multi-label classification
head (num_labels sigmoid outputs, applied by the loss function -- the model
itself returns raw logits). All backbones start from ImageNet-pretrained
weights (transfer learning), per the project brief -- nothing is trained
from scratch.
"""
import torch.nn as nn
import torchvision.models as tvm

SUPPORTED_MODELS = ["resnet50", "densenet121", "efficientnet_b0", "vit_b_16", "vit_b_32"]

# CNN backbones use global-average-pool heads, so they tolerate any input
# resolution. torchvision's ViT uses a fixed patch grid sized for 224x224
# and asserts on anything else, so it needs its own image size. vit_b_32
# uses 32x32 patches (49 tokens vs vit_b_16's 196), which is ~3-4x cheaper
# on CPU -- used here as the practical stand-in for "a Vision Transformer"
# where GPU time is constrained; see README for the tradeoff.
DEFAULT_IMAGE_SIZE = {
    "resnet50": 128,
    "densenet121": 128,
    "efficientnet_b0": 128,
    "vit_b_16": 224,
    "vit_b_32": 224,
}


def build_model(name: str, num_labels: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    name = name.lower()

    if name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))

    elif name == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))

    elif name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))

    elif name == "vit_b_16":
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))

    elif name == "vit_b_32":
        weights = tvm.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vit_b_32(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))

    else:
        raise ValueError(f"Unsupported model '{name}'. Choose from {SUPPORTED_MODELS}")

    return model


def get_target_layer_for_gradcam(model: nn.Module, name: str):
    """Returns the last convolutional layer to hook for Grad-CAM. Not
    defined for ViT (no spatial conv feature map) -- Grad-CAM is a
    CNN-specific technique here; see README for the ViT explainability
    caveat."""
    name = name.lower()
    if name == "resnet50":
        return model.layer4[-1]
    elif name == "densenet121":
        return model.features.denseblock4.denselayer16.conv2 if hasattr(model.features, "denseblock4") else model.features[-1]
    elif name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Grad-CAM target layer not defined for '{name}' (non-CNN architecture)")
