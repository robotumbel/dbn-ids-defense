"""Deep Belief Network (DBN) for IoT intrusion detection.

A DBN is built by greedily pretraining a stack of RBMs
(:class:`~dbn_ids_defense.rbm.RBM`) on unlabelled features, then unrolling
the learned weights into a feed-forward classifier that is fine-tuned with
supervised cross-entropy. The fine-tuned classifier is a standard
``torch.nn.Module`` whose gradients w.r.t. the input are used both for the
FGSM attack and for adversarial-training defence.
"""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rbm import RBM


class DBNClassifier(nn.Module):
    """DBN-initialised feed-forward classifier.

    Parameters
    ----------
    input_dim : int
        Number of (selected) input features.
    hidden_dims : sequence of int
        Hidden-layer widths, one RBM per entry (e.g. ``(100, 80)``).
    n_classes : int
        Number of output classes.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], n_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        dims = [input_dim] + self.hidden_dims
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(self.hidden_dims))])
        self.out = nn.Linear(dims[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = torch.sigmoid(layer(x))
        return self.out(x)

    # -- DBN pretraining -----------------------------------------------------
    def pretrain(self, X: torch.Tensor, epochs: int = 15, lr: float = 0.05,
                 batch_size: int = 64, verbose: bool = False):
        """Greedy layer-wise RBM pretraining; copies weights into the MLP."""
        inp = X
        for li, h in enumerate(self.hidden):
            n_vis = inp.size(1)
            rbm = RBM(n_vis, h.out_features)
            if verbose:
                print(f"  pretraining RBM {li + 1}/{len(self.hidden)} "
                      f"({n_vis}->{h.out_features})")
            rbm.fit(inp, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose)
            with torch.no_grad():               # unroll RBM weights into the layer
                h.weight.copy_(rbm.W.t())
                h.bias.copy_(rbm.h_bias)
            inp = rbm.transform(inp)            # propagate to next layer
        return self

    # -- supervised fine-tuning ---------------------------------------------
    def finetune(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 50,
                 lr: float = 1e-3, batch_size: int = 32, class_weight=None,
                 verbose: bool = False):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for ep in range(epochs):
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = F.cross_entropy(self(X[idx]), y[idx], weight=class_weight)
                loss.backward(); opt.step()
            if verbose:
                with torch.no_grad():
                    acc = (self(X).argmax(1) == y).float().mean().item()
                print(f"  finetune epoch {ep + 1}/{epochs}  train-acc={acc:.4f}")
        self.eval(); return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval(); return self(X).argmax(1)


def build_dbn(input_dim, hidden_dims, n_classes, X_pretrain, y, *,
              pre_epochs=15, ft_epochs=50, lr=1e-3, class_weight=None, verbose=False):
    """Convenience: build, pretrain, and fine-tune a DBN classifier."""
    model = DBNClassifier(input_dim, hidden_dims, n_classes)
    model.pretrain(X_pretrain, epochs=pre_epochs, verbose=verbose)
    model.finetune(X_pretrain, y, epochs=ft_epochs, lr=lr,
                   class_weight=class_weight, verbose=verbose)
    return model
