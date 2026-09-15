"""
Vision Classifier Factory
============================
Creates pretrained vision models (ResNet-18, MobileNetV2) with
the final classification layer replaced for the target number of classes.

Usage:
    from src.models.vision_classifier import create_vision_model
    model = create_vision_model("resnet18", num_classes=10, pretrained=True)
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


def create_vision_model(
    name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """
    Factory function to create a vision classifier.

    Loads pretrained ImageNet weights and replaces the final FC layer
    with a new one targeting `num_classes` outputs. Adds dropout
    before the final layer to combat overfitting on small datasets.

    Args:
        name: Model architecture. Supports "resnet18" and "mobilenet_v2".
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained ImageNet weights.
        dropout: Dropout probability before the final FC layer.
        device: Target device for the model.

    Returns:
        The configured model moved to `device`.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    weights: Any  # Typing for torchvision weight enums

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

        # Replace final FC: add dropout for regularization on small data
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        logger.info(f"ResNet-18: replaced fc layer → {in_features} → {num_classes}")

    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        # MobileNetV2 uses a classifier sequential block
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        logger.info(f"MobileNetV2: replaced classifier → {in_features} → {num_classes}")

    else:
        raise ValueError(
            f"Unsupported model: '{name}'. Choose 'resnet18' or 'mobilenet_v2'."
        )

    # Move to device
    model = model.to(device)

    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model '{name}': {total_params:,} total params, "
        f"{trainable_params:,} trainable"
    )

    # Report VRAM if on GPU
    if str(device).startswith("cuda") and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"GPU memory after model load: {vram_mb:.1f} MB")

    return model
