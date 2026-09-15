"""
Vision Augmented Training Script
====================================
Trains ResNet-18 on CIFAR-10 with a specified augmentation strategy.
Compares results against the baseline.

Usage:
    python scripts/train_vision_augmented.py --strategy basic_geometric
    python scripts/train_vision_augmented.py --strategy mixup
    python scripts/train_vision_augmented.py --strategy combined
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.cutmix import CutMixAugmentation
from src.augmentation.mixup import MixupAugmentation
from src.augmentation.vision_augmentor import VisionAugmentor
from src.models.vision_classifier import create_vision_model
from src.training.trainer import Trainer
from src.utils.data_loader_vision import get_vision_dataloaders
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train vision model with augmentation"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=[
            "basic_geometric",
            "photometric",
            "mixup",
            "cutmix",
            "combined",
        ],
        help="Augmentation strategy to apply",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config (for multi-run benchmarks)",
    )
    return parser.parse_args()


def load_baseline_accuracy(results_dir: Path) -> float | None:
    """Load baseline accuracy from saved results for comparison."""
    baseline_path = results_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            data = json.load(f)
        return data.get("test_results", {}).get("accuracy", None)
    return None


def main() -> None:
    """Train with augmentation and compare to baseline."""
    args = parse_args()
    strategy = args.strategy

    # Load config
    config_path = PROJECT_ROOT / "config" / "vision_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override seed if provided
    seed = args.seed if args.seed is not None else config["experiment"]["seed"]

    print("=" * 60)
    print(f"  VISION AUGMENTED — Strategy: {strategy.upper()}")
    print(f"  ResNet-18 on CIFAR-10 (100/class) | Seed: {seed}")
    print("=" * 60)

    set_seed(seed)

    # Device
    if config["experiment"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["experiment"]["device"])
    print(f"\nDevice: {device}")

    # Create augmentation transform
    augmentor = VisionAugmentor(image_size=config["dataset"]["image_size"])
    train_transform = augmentor.get_transform(strategy)

    # Setup mixup/cutmix functions (applied at batch level, not transform level)
    mixup_fn = None
    cutmix_fn = None
    num_classes = config["dataset"]["num_classes"]

    if strategy == "mixup":
        mixup_fn = MixupAugmentation(
            alpha=config["augmentation"]["mixup_alpha"],
            num_classes=num_classes,
        )
    elif strategy == "cutmix":
        cutmix_fn = CutMixAugmentation(
            alpha=config["augmentation"]["cutmix_alpha"],
            num_classes=num_classes,
        )

    # Load data with augmentation transform
    print(f"\n📦 Loading CIFAR-10 with '{strategy}' augmentation...")
    train_loader, val_loader, test_loader = get_vision_dataloaders(
        config, train_transform=train_transform
    )

    # Create model
    print("🏗️  Creating ResNet-18...")
    model = create_vision_model(
        name=config["model"]["name"],
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        device=device,
    )

    # Initialize experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=f"vision_{strategy}",
        results_dir=config["benchmark"]["results_dir"],
    )
    exp_logger.log_config(config)
    exp_logger.log_metadata(
        strategy=strategy,
        seed=seed,
        device=str(device),
    )

    # Train
    print(f"\n🚀 Training with {strategy}...\n")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_logger=exp_logger,
        mixup_fn=mixup_fn,
        cutmix_fn=cutmix_fn,
    )

    history = trainer.train()

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    # Log and save
    exp_logger.log_results(test_metrics)
    results_path = exp_logger.save(filename=f"{strategy}_results.json")

    # Compare to baseline
    results_dir = Path(config["benchmark"]["results_dir"])
    baseline_acc = load_baseline_accuracy(results_dir)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {strategy.upper()}")
    print("=" * 60)
    for metric, value in test_metrics.items():
        print(f"  {metric:>20s}: {value:.4f}")

    if baseline_acc is not None:
        improvement = test_metrics["accuracy"] - baseline_acc
        pct = improvement / baseline_acc * 100
        symbol = "↑" if improvement > 0 else "↓"
        print(f"\n  Baseline accuracy:   {baseline_acc:.4f}")
        print(f"  This strategy:       {test_metrics['accuracy']:.4f}")
        print(f"  Improvement:         {symbol} {abs(improvement):.4f} ({abs(pct):.1f}%)")
    else:
        print("\n  ⚠ No baseline results found. Run train_vision_baseline.py first.")

    print(f"\n  Results saved to: {results_path}")
    print("=" * 60)

    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
