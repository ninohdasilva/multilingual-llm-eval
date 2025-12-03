"""Global configuration and settings for the evaluation pipeline."""

import random

import numpy as np
import torch

# Global random seed for reproducibility
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device(preferred: str = None) -> str:
    """
    Detect the best available device for model inference.

    Priority: CUDA > MPS > CPU
    Falls back to CPU if preferred device is not available.

    Args:
        preferred: Preferred device ('cuda', 'mps', 'cpu'). If None, auto-detect.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if preferred:
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif preferred == "cpu":
            return "cpu"
        else:
            # Preferred device not available, fall back to auto-detect
            pass

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Initialize seeds on import
set_seeds(RANDOM_SEED)

