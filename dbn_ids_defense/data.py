"""Data loading, Information-Gain feature selection, scaling, and splitting
for IoT intrusion-detection CSVs (e.g. CIC-IoT2023). The last column is the
label; all preceding columns are numeric features."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


def load_csv(path: str):
    """Read a CSV; return (X, y_int, class_names). Last column = label."""
    df = pd.read_csv(path)
    X = np.nan_to_num(df.iloc[:, :-1].values.astype(np.float32), posinf=0, neginf=0)
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1].astype(str).values)
    return X, y.astype(np.int64), list(le.classes_)


def select_information_gain(X_tr, y_tr, X_te, top_k, seed=42):
    """Rank features by mutual information (information gain) on the training
    split and keep the ``top_k`` most informative. Returns (Xtr_sel, Xte_sel,
    kept_indices)."""
    ig = mutual_info_classif(X_tr, y_tr, random_state=seed)
    order = np.argsort(ig)[::-1][:top_k]
    return X_tr[:, order], X_te[:, order], order


def prepare(path: str, top_k: int = 15, test_size: float = 0.2, seed: int = 42):
    """Full pipeline: load -> split -> IG-select -> MinMax scale to [0,1].

    Returns a dict of torch tensors and metadata. Inputs are scaled to
    ``[0, 1]`` because the RBM stack expects Bernoulli-distributed visibles.
    """
    X, y, class_names = load_csv(path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr, X_te, kept = select_information_gain(X_tr, y_tr, X_te, top_k, seed)
    scaler = MinMaxScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    return dict(
        X_train=torch.tensor(X_tr), y_train=torch.tensor(y_tr),
        X_test=torch.tensor(X_te), y_test=torch.tensor(y_te),
        n_features=top_k, n_classes=len(class_names),
        class_names=class_names, kept_indices=kept, scaler=scaler)


def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced training."""
    counts = torch.bincount(y, minlength=n_classes).float()
    w = counts.sum() / (n_classes * counts.clamp(min=1))
    return w
