# src/random_utils.py
"""Utility function to set random seeds for reproducibility."""

def seed_everything(seed: int) -> None:
    """Set the random seed for reproducibility."""
    import random

    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False