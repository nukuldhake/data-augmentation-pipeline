"""
Reproducibility Seed Manager
==============================
Sets random seeds across all sources of randomness to ensure
deterministic, reproducible experiments.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - PYTHONHASHSEED environment variable
    - cuDNN deterministic mode

    Args:
        seed: Integer seed value. Default 42.
    """
    # Python builtins
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN: deterministic mode trades speed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed} (deterministic mode enabled)")
