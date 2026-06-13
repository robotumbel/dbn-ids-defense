"""Fast Gradient Sign Method (FGSM) evasion attack.

Implements Goodfellow et al. (2015) directly with autograd. For a model
``f`` and input ``x`` with true label ``y``, FGSM perturbs the input one
step along the sign of the loss gradient and clips back to the valid
``[0, 1]`` feature range:

    x_adv = clip(x + eps * sign(grad_x L(f(x), y)), 0, 1).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def fgsm(model, X: torch.Tensor, y: torch.Tensor, eps: float = 0.1,
         clip=(0.0, 1.0)) -> torch.Tensor:
    """Generate FGSM adversarial examples for a batch of inputs."""
    model.eval()
    X = X.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(X), y)
    model.zero_grad()
    loss.backward()
    X_adv = X + eps * X.grad.sign()
    if clip is not None:
        X_adv = X_adv.clamp(clip[0], clip[1])
    return X_adv.detach()
