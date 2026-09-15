"""
Shared Evaluation Metrics
===========================
Computes classification metrics from ground truth and predictions.
Supports both multi-class (vision) and binary (NLP) settings.

Usage:
    from src.evaluation.metrics import compute_metrics
    results = compute_metrics(y_true, y_pred, y_prob=softmax_probs)
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Optional softmax probabilities for ROC-AUC (shape: [N, C]).
        num_classes: Number of classes. If None, inferred from y_true.

    Returns:
        Dictionary with:
            - accuracy
            - f1_weighted
            - f1_macro
            - precision_weighted
            - recall_weighted
            - roc_auc (binary only, if y_prob provided)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if num_classes is None:
        num_classes = len(np.unique(y_true))

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # Binary ROC-AUC (NLP: 2 classes)
    if num_classes == 2 and y_prob is not None:
        try:
            # Use probability of the positive class
            if y_prob.ndim == 2:
                pos_probs = y_prob[:, 1]
            else:
                pos_probs = y_prob
            metrics["roc_auc"] = float(roc_auc_score(y_true, pos_probs))
        except ValueError as e:
            logger.warning(f"ROC-AUC computation failed: {e}")
            metrics["roc_auc"] = 0.0

    return metrics


def get_classification_report(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """
    Generate a formatted classification report string.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of human-readable class names.

    Returns:
        Formatted classification report string.
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )


def get_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix as numpy array.
    """
    return confusion_matrix(y_true, y_pred)
