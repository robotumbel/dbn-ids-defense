"""Bernoulli Restricted Boltzmann Machine (RBM) trained with one-step
contrastive divergence (CD-1). RBMs are stacked to form the Deep Belief
Network in :mod:`dbn_ids_defense.dbn`."""
from __future__ import annotations
import torch
import torch.nn as nn


class RBM(nn.Module):
    """A Bernoulli-Bernoulli RBM.

    Parameters
    ----------
    n_visible : int
        Number of visible units (input dimension).
    n_hidden : int
        Number of hidden units.
    """

    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_hidden(self, v: torch.Tensor):
        p_h = torch.sigmoid(v @ self.W + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_visible(self, h: torch.Tensor):
        p_v = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return p_v, torch.bernoulli(p_v)

    @torch.no_grad()
    def cd1_update(self, v0: torch.Tensor, lr: float):
        """One contrastive-divergence (CD-1) parameter update for a batch."""
        p_h0, h0 = self.sample_hidden(v0)
        p_v1, _ = self.sample_visible(h0)
        p_h1, _ = self.sample_hidden(p_v1)
        n = v0.size(0)
        self.W.add_(lr * (v0.t() @ p_h0 - p_v1.t() @ p_h1) / n)
        self.v_bias.add_(lr * (v0 - p_v1).mean(0))
        self.h_bias.add_(lr * (p_h0 - p_h1).mean(0))
        return torch.mean((v0 - p_v1) ** 2).item()      # reconstruction error

    def fit(self, X: torch.Tensor, epochs: int = 15, lr: float = 0.05,
            batch_size: int = 64, verbose: bool = False):
        """Greedily pretrain the RBM on data in ``[0, 1]`` with CD-1."""
        for ep in range(epochs):
            perm = torch.randperm(X.size(0))
            err = 0.0
            for i in range(0, X.size(0), batch_size):
                v = X[perm[i:i + batch_size]]
                err += self.cd1_update(v, lr)
            if verbose:
                print(f"    RBM epoch {ep + 1}/{epochs}  recon-err={err:.4f}")
        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Map data to the hidden-unit activation probabilities."""
        return torch.sigmoid(X @ self.W + self.h_bias)
