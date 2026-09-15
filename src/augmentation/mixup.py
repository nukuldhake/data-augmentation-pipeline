"""
Mixup Augmentation (Batch-Level)
==================================
Implements the Mixup strategy from Zhang et al. (2018).
Operates at the batch level: creates convex combinations of
image pairs and their labels using a Beta distribution.

The mixed labels are soft targets, so the Trainer must use
soft cross-entropy loss when this is active.

Usage:
    from src.augmentation.mixup import MixupAugmentation
    mixup = MixupAugmentation(alpha=0.2)
    mixed_images, mixed_labels = mixup(images, labels)
"""

import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class MixupAugmentation:
    """
    Batch-level Mixup augmentation.

    Given a batch of images and labels:
        mixed_x = λ · x_i + (1-λ) · x_j
        mixed_y = λ · y_i + (1-λ) · y_j
    where λ ~ Beta(alpha, alpha) and j is a random permutation of i.
    """

    def __init__(self, alpha: float = 0.2, num_classes: int = 10) -> None:
        """
        Args:
            alpha: Beta distribution parameter. Smaller → less mixing.
                   alpha=0.2 gives λ close to 0 or 1 (mild mixing).
            num_classes: Number of classes for one-hot encoding labels.
        """
        self.alpha = alpha
        self.num_classes = num_classes
        logger.info(f"MixupAugmentation initialized: alpha={alpha}")

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixup to a batch.

        Args:
            images: Batch tensor of shape [B, C, H, W].
            labels: Label tensor of shape [B] (integer class indices).

        Returns:
            Tuple of (mixed_images, mixed_soft_labels).
            mixed_images: [B, C, H, W] tensor.
            mixed_soft_labels: [B, num_classes] soft label tensor.
        """
        batch_size = images.size(0)
        device = images.device

        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation for pairing
        indices = torch.randperm(batch_size, device=device)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]

        # Convert labels to one-hot, then mix
        labels_onehot = torch.zeros(
            batch_size, self.num_classes, device=device, dtype=torch.float32
        )
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

        labels_shuffled = torch.zeros(
            batch_size, self.num_classes, device=device, dtype=torch.float32
        )
        labels_shuffled.scatter_(1, labels[indices].unsqueeze(1), 1.0)

        mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled

        return mixed_images, mixed_labels
