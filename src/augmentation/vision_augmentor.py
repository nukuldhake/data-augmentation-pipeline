"""
Vision Augmentation Strategies (Albumentations)
=================================================
Defines 5 augmentation strategies + baseline for CIFAR-10 image
classification using Albumentations pipelines.

Strategies:
    V0: baseline          — Resize + Normalize (no augmentation)
    V1: basic_geometric   — Flips, rotations, shift-scale-rotate
    V2: photometric        — Brightness, contrast, hue, noise, blur
    V3: mixup             — Base transforms (Mixup applied at batch level)
    V4: cutmix            — Base transforms (CutMix applied at batch level)
    V5: combined          — Geometric + photometric + CoarseDropout

Usage:
    from src.augmentation.vision_augmentor import VisionAugmentor
    augmentor = VisionAugmentor(image_size=224)
    transform = augmentor.get_transform("basic_geometric")
"""

import logging
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VisionAugmentor:
    """
    Factory for Albumentations augmentation pipelines.

    Each strategy returns an `A.Compose` pipeline that expects
    numpy arrays (HWC, uint8) and outputs normalized tensors.
    """

    def __init__(self, image_size: int = 224) -> None:
        """
        Args:
            image_size: Target image dimension (square). Default 224.
        """
        self.image_size = image_size

    def get_strategy_names(self) -> list[str]:
        """Return the list of all available augmentation strategy names."""
        return [
            "baseline",
            "basic_geometric",
            "photometric",
            "mixup",
            "cutmix",
            "combined",
        ]

    def get_transform(self, strategy_name: str) -> A.Compose:
        """
        Return the Albumentations pipeline for a given strategy.

        Args:
            strategy_name: One of the strategy names from get_strategy_names().

        Returns:
            An A.Compose pipeline.

        Raises:
            ValueError: If strategy_name is not recognized.
        """
        strategy_map = {
            "baseline": self._baseline,
            "basic_geometric": self._basic_geometric,
            "photometric": self._photometric,
            "mixup": self._mixup_base,
            "cutmix": self._cutmix_base,
            "combined": self._combined,
        }

        if strategy_name not in strategy_map:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(strategy_map.keys())}"
            )

        transform = strategy_map[strategy_name]()
        logger.info(f"Created transform pipeline: {strategy_name}")
        return transform

    def _baseline(self) -> A.Compose:
        """V0: Resize + Normalize. No augmentation."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _basic_geometric(self) -> A.Compose:
        """V1: Spatial augmentations — flips, rotations, shift-scale-rotate."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.5,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _photometric(self) -> A.Compose:
        """V2: Color/intensity augmentations — brightness, hue, noise, blur."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _mixup_base(self) -> A.Compose:
        """
        V3: Base transforms for Mixup.
        Mixup itself is applied at batch level by MixupAugmentation.
        """
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _cutmix_base(self) -> A.Compose:
        """
        V4: Base transforms for CutMix.
        CutMix itself is applied at batch level by CutMixAugmentation.
        """
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _combined(self) -> A.Compose:
        """V5: Kitchen sink — geometric + photometric + CoarseDropout."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                max_holes=4,
                max_height=24,
                max_width=24,
                p=0.2,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def visualize_samples(
        self,
        image: np.ndarray,
        strategy: str,
        n: int = 5,
        save_path: str | None = None,
    ) -> None:
        """
        Visualize N augmented versions of a single image for sanity checking.

        Args:
            image: Input image as numpy array (HWC, uint8).
            strategy: Strategy name to apply.
            n: Number of augmented versions to show.
            save_path: Optional path to save the figure.
        """
        transform = self.get_transform(strategy)

        fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))

        # Show original
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Show augmented versions (apply transform but undo normalization for display)
        display_transform = A.Compose([
            t for t in transform.transforms
            if not isinstance(t, (A.Normalize, ToTensorV2))
        ])

        for i in range(n):
            augmented = display_transform(image=image)["image"]
            axes[i + 1].imshow(augmented)
            axes[i + 1].set_title(f"Aug {i + 1}")
            axes[i + 1].axis("off")

        plt.suptitle(f"Strategy: {strategy}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")

        plt.close()
