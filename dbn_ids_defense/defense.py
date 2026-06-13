"""Adversarial-training defence.

The detector is hardened by augmenting the training set with FGSM
adversarial examples and re-fine-tuning, so that the decision boundary is
robust to small protocol-level perturbations.
"""
from __future__ import annotations
import torch
from .attacks import fgsm


def adversarial_train(model, X_train, y_train, eps=0.1, epochs=50, lr=1e-3,
                      class_weight=None, verbose=False):
    """Augment training data with FGSM examples and re-fine-tune the model."""
    X_adv = fgsm(model, X_train, y_train, eps=eps)
    X_aug = torch.cat([X_train, X_adv], dim=0)
    y_aug = torch.cat([y_train, y_train], dim=0)
    model.finetune(X_aug, y_aug, epochs=epochs, lr=lr,
                   class_weight=class_weight, verbose=verbose)
    return model
