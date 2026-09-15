"""
JSON Experiment Logger
========================
Structured logging for experiment tracking. Captures configuration,
per-epoch metrics, and final results in a single JSON file.

Usage:
    from src.utils.logger import ExperimentLogger
    logger = ExperimentLogger(
        experiment_name="vision_baseline",
        results_dir="experiments/results/vision"
    )
    logger.log_config(config_dict)
    logger.log_epoch(epoch=1, train_loss=0.5, val_acc=0.65, ...)
    logger.log_results(test_metrics)
    logger.save()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    JSON-based experiment logger that saves structured records
    of configuration, training history, and final evaluation results.
    """

    def __init__(self, experiment_name: str, results_dir: str | Path) -> None:
        """
        Initialize the experiment logger.

        Args:
            experiment_name: Human-readable name for this experiment run.
            results_dir: Directory to save the JSON log file.
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.log_data: dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": {},
            "epoch_history": [],
            "test_results": {},
            "metadata": {},
        }

        logger.info(f"ExperimentLogger initialized: {experiment_name}")

    def log_config(self, config: dict[str, Any]) -> None:
        """
        Record the experiment configuration.

        Args:
            config: Full configuration dictionary (typically from YAML).
        """
        self.log_data["config"] = config
        logger.info("Configuration logged")

    def log_epoch(self, epoch: int, **metrics: float) -> None:
        """
        Record metrics for a single training epoch.

        Args:
            epoch: Epoch number (1-indexed).
            **metrics: Arbitrary metric key-value pairs, e.g.
                       train_loss=0.5, val_acc=0.65, val_loss=0.4, lr=1e-3.
        """
        entry = {"epoch": epoch, **metrics}
        self.log_data["epoch_history"].append(entry)
        logger.debug(f"Epoch {epoch} metrics: {metrics}")

    def log_results(self, results: dict[str, float]) -> None:
        """
        Record final test set evaluation results.

        Args:
            results: Dictionary of metric name → value, e.g.
                     {"accuracy": 0.85, "f1_weighted": 0.84}.
        """
        self.log_data["test_results"] = results
        self.log_data["end_time"] = datetime.now().isoformat()
        logger.info(f"Test results logged: {results}")

    def log_metadata(self, **kwargs: Any) -> None:
        """
        Record arbitrary metadata (model name, strategy, seed, etc.).

        Args:
            **kwargs: Key-value pairs to store.
        """
        self.log_data["metadata"].update(kwargs)

    def save(self, filename: str | None = None) -> Path:
        """
        Save the log to a JSON file.

        Args:
            filename: Optional override filename. If None, uses
                      '{experiment_name}_results.json'.

        Returns:
            Path to the saved JSON file.
        """
        if filename is None:
            # Sanitize experiment name for use as filename
            safe_name = self.experiment_name.replace(" ", "_").lower()
            filename = f"{safe_name}_results.json"

        filepath = self.results_dir / filename
        self.log_data["end_time"] = datetime.now().isoformat()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2, default=str)

        logger.info(f"Experiment log saved to {filepath}")
        return filepath

    def get_best_epoch(self, metric: str = "val_acc", mode: str = "max") -> dict[str, Any]:
        """
        Find the epoch with the best value for a given metric.

        Args:
            metric: Name of the metric to optimize.
            mode: 'max' for accuracy-like metrics, 'min' for loss-like.

        Returns:
            Dictionary of the best epoch's metrics.
        """
        if not self.log_data["epoch_history"]:
            return {}

        compare = max if mode == "max" else min
        best = compare(
            self.log_data["epoch_history"],
            key=lambda e: e.get(metric, float("-inf") if mode == "max" else float("inf")),
        )
        return best
