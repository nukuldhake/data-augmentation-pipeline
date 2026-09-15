"""
Vision Baseline Training Script
==================================
Trains ResNet-18 on subsampled CIFAR-10 WITHOUT any augmentation.
This establishes the reference accuracy that all augmentation
strategies are benchmarked against.

Expected result: ~50-55% accuracy on 100 samples/class.

Run: python scripts/train_vision_baseline.py
"""

import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger
from src.utils.data_loader_vision import get_vision_dataloaders
from src.models.vision_classifier import create_vision_model
from src.training.trainer import Trainer
from src.evaluation.metrics import get_classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main() -> None:
    """Train vision baseline and save results."""

    # Load config
    config_path = PROJECT_ROOT / "config" / "vision_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  VISION BASELINE — ResNet-18 on CIFAR-10 (100/class)")
    print("  No augmentation (resize + normalize only)")
    print("=" * 60)

    # Set seed for reproducibility
    seed = config["experiment"]["seed"]
    set_seed(seed)

    # Determine device
    if config["experiment"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["experiment"]["device"])
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"VRAM: {vram:.1f} GB")

    # Initialize experiment logger
    exp_logger = ExperimentLogger(
        experiment_name="vision_baseline",
        results_dir=config["benchmark"]["results_dir"],
    )
    exp_logger.log_config(config)
    exp_logger.log_metadata(
        strategy="baseline",
        seed=seed,
        device=str(device),
    )

    # Load data (no augmentation → uses baseline transform)
    print("\n📦 Loading CIFAR-10 (subsampled)...")
    train_loader, val_loader, test_loader = get_vision_dataloaders(config)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    # Create model
    print("\n🏗️  Creating ResNet-18...")
    model = create_vision_model(
        name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        device=device,
    )

    # Train
    print("\n🚀 Starting training...\n")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_logger=exp_logger,
    )

    history = trainer.train()

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    # Log and save results
    exp_logger.log_results(test_metrics)
    results_path = exp_logger.save(filename="baseline_results.json")

    # Print final results
    print("\n" + "=" * 60)
    print("  VISION BASELINE RESULTS")
    print("=" * 60)
    for metric, value in test_metrics.items():
        print(f"  {metric:>20s}: {value:.4f}")
    print("=" * 60)
    print(f"\n  ★ Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Results saved to: {results_path}")
    print("=" * 60)

    # Cleanup GPU memory
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
