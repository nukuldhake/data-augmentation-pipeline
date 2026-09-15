"""
Vision Data Loader — CIFAR-10
===============================
Loads CIFAR-10, subsamples to N per class for low-resource simulation,
splits into train/val/test, and wraps in DataLoaders.

The train set supports an optional Albumentations transform; val/test
always use the baseline (resize + normalize) transform.

Usage:
    from src.utils.data_loader_vision import get_vision_dataloaders
    train_dl, val_dl, test_dl = get_vision_dataloaders(config)
"""

import logging
from typing import Any

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AlbumentationsTransform:
    """
    Wrapper that bridges PIL Images (from torchvision datasets) to
    Albumentations pipelines, which expect numpy arrays in HWC uint8 format.
    """

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, img: Image.Image) -> Any:
        # PIL → numpy HWC uint8
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented["image"]


class SubsampledCIFAR10(Dataset):
    """
    A subset of CIFAR-10 with a configurable number of samples per class.
    Wraps indices into the original torchvision CIFAR10 dataset.
    """

    def __init__(
        self,
        cifar_dataset: torchvision.datasets.CIFAR10,
        indices: list[int],
        transform: AlbumentationsTransform | None = None,
    ) -> None:
        """
        Args:
            cifar_dataset: The full torchvision CIFAR10 dataset.
            indices: List of indices into cifar_dataset to include.
            transform: Optional AlbumentationsTransform to apply.
        """
        self.dataset = cifar_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]  # img is PIL Image (no transform on base)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_baseline_transform(image_size: int = 224) -> A.Compose:
    """
    Baseline transform: resize + normalize. No augmentation.
    Used for val/test sets and as the baseline training strategy.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def subsample_indices(
    targets: list[int],
    samples_per_class: int,
    num_classes: int,
    seed: int = 42,
) -> list[int]:
    """
    Stratified subsampling: select exactly N samples per class.

    Args:
        targets: Full list of class labels.
        samples_per_class: Number of samples to keep per class.
        num_classes: Total number of classes.
        seed: Random seed for reproducibility.

    Returns:
        List of selected indices.
    """
    rng = np.random.RandomState(seed)
    targets_arr = np.array(targets)
    selected: list[int] = []

    for cls in range(num_classes):
        cls_indices = np.where(targets_arr == cls)[0]
        if len(cls_indices) < samples_per_class:
            logger.warning(
                f"Class {cls} has only {len(cls_indices)} samples, "
                f"requested {samples_per_class}"
            )
            chosen = cls_indices
        else:
            chosen = rng.choice(cls_indices, size=samples_per_class, replace=False)
        selected.extend(chosen.tolist())

    logger.info(
        f"Subsampled {len(selected)} total samples "
        f"({samples_per_class}/class × {num_classes} classes)"
    )
    return selected


def split_indices(
    indices: list[int],
    targets: list[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split indices into train/val/test using stratified sampling.

    Args:
        indices: Indices to split.
        targets: Corresponding labels (for stratification).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    labels = [targets[i] for i in indices]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, valtest_idx, train_labels, valtest_labels = train_test_split(
        indices, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=relative_test_ratio,
        stratify=valtest_labels,
        random_state=seed,
    )

    logger.info(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


def get_vision_dataloaders(
    config: dict[str, Any],
    train_transform: A.Compose | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build CIFAR-10 DataLoaders with subsampling and optional augmentation.

    Args:
        config: Configuration dictionary (from vision_config.yaml).
        train_transform: Optional Albumentations Compose for the training set.
                         If None, uses baseline (resize + normalize).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    seed = config["experiment"]["seed"]

    # Load full CIFAR-10 (no torchvision transforms — we use Albumentations)
    full_dataset = torchvision.datasets.CIFAR10(
        root=dataset_cfg["root"],
        train=True,
        download=True,
        transform=None,  # Raw PIL Images
    )

    # Subsample to N per class
    selected = subsample_indices(
        targets=full_dataset.targets,
        samples_per_class=dataset_cfg["samples_per_class"],
        num_classes=dataset_cfg["num_classes"],
        seed=seed,
    )

    # Split into train/val/test
    train_idx, val_idx, test_idx = split_indices(
        indices=selected,
        targets=full_dataset.targets,
        train_ratio=dataset_cfg["train_split"],
        val_ratio=dataset_cfg["val_split"],
        test_ratio=dataset_cfg["test_split"],
        seed=seed,
    )

    # Build transforms
    image_size = dataset_cfg["image_size"]
    baseline = get_baseline_transform(image_size)

    if train_transform is None:
        train_transform = baseline

    train_albu = AlbumentationsTransform(train_transform)
    eval_albu = AlbumentationsTransform(baseline)

    # Build datasets
    train_ds = SubsampledCIFAR10(full_dataset, train_idx, transform=train_albu)
    val_ds = SubsampledCIFAR10(full_dataset, val_idx, transform=eval_albu)
    test_ds = SubsampledCIFAR10(full_dataset, test_idx, transform=eval_albu)

    # Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    logger.info(
        f"DataLoaders ready: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}, "
        f"batch_size={training_cfg['batch_size']}"
    )

    return train_loader, val_loader, test_loader
