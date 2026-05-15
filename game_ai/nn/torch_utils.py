from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install it on the training machine with: "
            "pip install torch torchvision"
        ) from exc

    return torch, nn, F
