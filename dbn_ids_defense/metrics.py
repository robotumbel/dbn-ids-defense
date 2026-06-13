"""Evaluation metrics (imbalance-aware)."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             classification_report, confusion_matrix)


@torch.no_grad()
def evaluate(model, X, y):
    """Return accuracy, macro-F1, and MCC for a model on (X, y)."""
    yp = model.predict(X).cpu().numpy()
    yt = y.cpu().numpy()
    return dict(
        accuracy=round(float(accuracy_score(yt, yp)), 4),
        f1_macro=round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        mcc=round(float(matthews_corrcoef(yt, yp)), 4))


@torch.no_grad()
def report(model, X, y, class_names=None):
    yp = model.predict(X).cpu().numpy(); yt = y.cpu().numpy()
    return classification_report(yt, yp, zero_division=0), confusion_matrix(yt, yp)
