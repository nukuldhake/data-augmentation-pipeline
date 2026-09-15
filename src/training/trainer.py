"""
Shared Training Loop
======================
A unified Trainer class for both vision and NLP experiments.
Handles: training, validation, early stopping, checkpointing,
mixup/cutmix, gradient accumulation, gradient clipping, and
warmup schedulers.

Usage:
    from src.training.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()
    test_metrics = trainer.evaluate(test_loader)
"""

import gc
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.logger import ExperimentLogger

logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified training loop supporting both vision and NLP workloads.

    Features:
        - Standard training + validation with early stopping
        - Mixup / CutMix batch-level augmentation (vision)
        - Gradient accumulation (NLP: effective batch = batch_size × grad_accum)
        - Gradient clipping (NLP: max_grad_norm)
        - LR scheduling (ReduceOnPlateau for vision, linear warmup for NLP)
        - Best-model checkpointing
        - JSON experiment logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device,
        experiment_logger: ExperimentLogger | None = None,
        mixup_fn: Callable | None = None,
        cutmix_fn: Callable | None = None,
    ) -> None:
        """
        Args:
            model: The neural network to train.
            train_loader: Training data DataLoader.
            val_loader: Validation data DataLoader.
            config: Full config dict (from YAML).
            device: Target device (cpu or cuda).
            experiment_logger: Optional ExperimentLogger instance.
            mixup_fn: Optional Mixup callable(images, labels) → (mixed, mixed_labels).
            cutmix_fn: Optional CutMix callable(images, labels) → (mixed, mixed_labels).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_logger = experiment_logger
        self.mixup_fn = mixup_fn
        self.cutmix_fn = cutmix_fn

        training_cfg = config["training"]
        self.epochs = training_cfg["epochs"]
        self.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = training_cfg.get("max_grad_norm", None)
        self.early_stopping_patience = training_cfg.get("early_stopping_patience", 10)

        # Build optimizer
        self.optimizer = self._build_optimizer(training_cfg)

        # Build scheduler
        self.scheduler = self._build_scheduler(training_cfg)

        # Loss function: CrossEntropyLoss for hard labels
        self.criterion = nn.CrossEntropyLoss()

        # Checkpoint directory
        results_dir = config.get("benchmark", {}).get(
            "results_dir", "./experiments/results"
        )
        self.checkpoint_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def _build_optimizer(self, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_name = training_cfg["optimizer"].lower()
        lr = training_cfg["learning_rate"]
        wd = training_cfg.get("weight_decay", 0.0)

        if opt_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif opt_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _build_scheduler(
        self, training_cfg: dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create LR scheduler from config."""
        sched_name = training_cfg.get("scheduler", "none").lower()

        if sched_name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",  # Maximize val accuracy
                patience=training_cfg.get("scheduler_patience", 5),
                factor=training_cfg.get("scheduler_factor", 0.5),
                verbose=True,
            )
        elif sched_name == "linear":
            # Linear warmup then linear decay — used for NLP
            # Requires transformers library
            from transformers import get_linear_schedule_with_warmup

            total_steps = (
                len(self.train_loader)
                // self.gradient_accumulation_steps
                * self.epochs
            )
            warmup_ratio = training_cfg.get("warmup_ratio", 0.1)
            warmup_steps = int(total_steps * warmup_ratio)

            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif sched_name == "none":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

    def _soft_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy loss for soft labels (mixup/cutmix).
        targets shape: [batch_size, num_classes] with soft probabilities.
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss

    def train(self) -> dict[str, list[float]]:
        """
        Run the full training loop.

        Returns:
            History dict with train_loss, val_loss, val_acc, lr per epoch.
        """
        logger.info(
            f"Starting training: {self.epochs} epochs, "
            f"grad_accum={self.gradient_accumulation_steps}, "
            f"patience={self.early_stopping_patience}"
        )

        for epoch in range(1, self.epochs + 1):
            # Training phase
            train_loss = self._train_one_epoch(epoch)

            # Validation phase
            val_loss, val_acc, val_metrics = self._validate(epoch)

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Log to experiment logger
            if self.experiment_logger:
                self.experiment_logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=current_lr,
                    **val_metrics,
                )

            # LR scheduling
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_acc)
                # Linear scheduler is stepped per batch (handled in _train_one_epoch)

            # Checkpointing: save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
                logger.info(f"  ★ New best: val_acc={val_acc:.4f}")
            else:
                self.patience_counter += 1

            # Epoch summary
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e} | "
                f"patience={self.patience_counter}/{self.early_stopping_patience}"
            )

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best val_acc={self.best_val_acc:.4f})"
                )
                print(f"\n⚡ Early stopping at epoch {epoch}")
                break

        # Load best checkpoint for final evaluation
        self._load_best_checkpoint()

        return self.history

    def _train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:3d} [train]",
            leave=False,
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)

            # Determine if we use soft labels (mixup/cutmix)
            use_soft_labels = False

            if self.mixup_fn is not None:
                targets_device = targets.to(self.device, non_blocking=True)
                inputs, targets_soft = self.mixup_fn(inputs, targets_device)
                use_soft_labels = True
            elif self.cutmix_fn is not None:
                targets_device = targets.to(self.device, non_blocking=True)
                inputs, targets_soft = self.cutmix_fn(inputs, targets_device)
                use_soft_labels = True
            else:
                targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(inputs)

            # Loss: soft vs hard labels
            if use_soft_labels:
                loss = self._soft_cross_entropy(outputs, targets_soft)
            else:
                loss = self.criterion(outputs, targets)

            # Scale loss by gradient accumulation steps
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Step optimizer every gradient_accumulation_steps batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (primarily for NLP)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Step linear scheduler per optimization step (not per batch)
                if (
                    self.scheduler is not None
                    and not isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    )
                ):
                    self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        # Handle remaining gradients if total batches not divisible by accum steps
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def _validate(
        self, epoch: int
    ) -> tuple[float, float, dict[str, float]]:
        """
        Run validation. Returns (avg_loss, accuracy, full_metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []
        num_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:3d} [val]  ",
            leave=False,
        )

        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().tolist()
            labels = targets.cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.append(probs)

        avg_loss = total_loss / max(num_batches, 1)
        all_probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.array([])

        metrics = compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_probs_np if all_probs_np.size > 0 else None,
        )

        accuracy = metrics["accuracy"]
        return avg_loss, accuracy, metrics

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model on a test set.

        Args:
            test_loader: DataLoader for the test set.

        Returns:
            Dictionary of test metrics.
        """
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []

        pbar = tqdm(test_loader, desc="Evaluating", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(inputs)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().tolist()
            labels = targets.cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.append(probs)

        all_probs_np = np.concatenate(all_probs, axis=0)

        metrics = compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_probs_np,
        )

        return metrics

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        """Save best model checkpoint."""
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
            },
            checkpoint_path,
        )
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _load_best_checkpoint(self) -> None:
        """Load the best model checkpoint."""
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded best checkpoint (epoch {checkpoint['epoch']}, "
                f"val_acc={checkpoint['val_acc']:.4f})"
            )
        else:
            logger.warning("No checkpoint found, using final model state")
