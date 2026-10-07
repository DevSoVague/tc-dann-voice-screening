"""
Custom layers for TC-DANN.

Contains:
- GradientReversal: the core of domain-adversarial training (Ganin & Lempitsky 2015).
  Forward pass is identity. Backward pass multiplies the gradient by -lambda.
  This makes the encoder actively UNLEARN features predictive of the adversary's
  target (site / age / sex), which is how we defeat the confounder shortcuts
  identified in the audit.

- MaskedStatsPooling: mean + std pooling with a length mask, used on 1D-conv
  branches so that variable-length SPARC / prosodic sequences are summarised
  without leaking padded positions.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Multiply gradient by -lambda on the backward pass
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer.

    lambda_ is typically ramped from 0 -> 1 over training (see train.py schedule).
    Starting at 0 lets the encoder learn useful disease features first, then
    gradually forces it to strip demographic / site information.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambda_)


class MaskedStatsPooling(nn.Module):
    """
    Masked mean + std pooling along the time axis.

    Args:
        x: [B, T, C] tensor
        mask: [B, T] binary mask, 1 = valid, 0 = padded. If None, treat all as valid.

    Returns:
        [B, 2*C] tensor (concatenated mean and std).
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mean = x.mean(dim=1)
            std = x.std(dim=1, unbiased=False)
        else:
            mask = mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
            denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
            mean = (x * mask).sum(dim=1) / denom
            var = ((x - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / denom
            std = var.clamp_min(1e-8).sqrt()
        return torch.cat([mean, std], dim=-1)
