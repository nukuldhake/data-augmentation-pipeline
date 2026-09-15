"""
CutMix Augmentation (Batch-Level)
====================================
Implements CutMix from Yun et al. (2019).
Cuts a random rectangular patch from one image and pastes it
onto another. Labels are mixed proportionally to the patch area.

Usage:
    from src.augmentation.cutmix import CutMixAugmentation
    cutmix = CutMixAugmentation(alpha=1.0)
    mixed_images, mixed_labels = cutmix(images, labels)
"""

import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class CutMixAugmentation:
    """
    Batch-level CutMix augmentation.

    Given a batch of images and labels:
    1. Sample λ ~ Beta(alpha, alpha)
    2. Compute random bounding box with area ratio (1-λ)
    3. Paste region from shuffled image onto original
    4. Adjust λ to reflect actual patch area
    5. Mix labels: y = λ_adj · y_original + (1-λ_adj) · y_shuffled
    """

    def __init__(self, alpha: float = 1.0, num_classes: int = 10) -> None:
        """
        Args:
            alpha: Beta distribution parameter. alpha=1.0 gives uniform λ.
            num_classes: Number of classes for one-hot encoding.
        """
        self.alpha = alpha
        self.num_classes = num_classes
        logger.info(f"CutMixAugmentation initialized: alpha={alpha}")

    @staticmethod
    def _rand_bbox(
        height: int, width: int, lam: float
    ) -> tuple[int, int, int, int]:
        """
        Generate a random bounding box.

        The box has an area ratio approximately equal to (1-lam)
        relative to the full image area.

        Args:
            height: Image height.
            width: Image width.
            lam: Mixing coefficient.

        Returns:
            (y1, x1, y2, x2) bounding box coordinates.
        """
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        # Random center
        cy = np.random.randint(height)
        cx = np.random.randint(width)

        # Clip to image bounds
        y1 = max(0, cy - cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        y2 = min(height, cy + cut_h // 2)
        x2 = min(width, cx + cut_w // 2)

        return y1, x1, y2, x2

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix to a batch.

        Args:
            images: Batch tensor of shape [B, C, H, W].
            labels: Label tensor of shape [B] (integer class indices).

        Returns:
            Tuple of (mixed_images, mixed_soft_labels).
            mixed_images: [B, C, H, W] tensor with pasted patches.
            mixed_soft_labels: [B, num_classes] soft label tensor.
        """
        batch_size = images.size(0)
        _, _, height, width = images.shape
        device = images.device

        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation for pairing
        indices = torch.randperm(batch_size, device=device)

        # Generate random bounding box
        y1, x1, y2, x2 = self._rand_bbox(height, width, lam)

        # Paste patch: copy region from shuffled images onto originals
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

        # Adjust λ by actual patch area (may differ from target due to clipping)
        patch_area = (y2 - y1) * (x2 - x1)
        total_area = height * width
        lam_adj = 1.0 - (patch_area / total_area)

        # Convert labels to one-hot, then mix
        labels_onehot = torch.zeros(
            batch_size, self.num_classes, device=device, dtype=torch.float32
        )
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

        labels_shuffled = torch.zeros(
            batch_size, self.num_classes, device=device, dtype=torch.float32
        )
        labels_shuffled.scatter_(1, labels[indices].unsqueeze(1), 1.0)

        mixed_labels = lam_adj * labels_onehot + (1 - lam_adj) * labels_shuffled

        return mixed_images, mixed_labels
