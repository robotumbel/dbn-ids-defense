"""Minimal Python-API demo of dbn_ids_defense on the bundled CIC-IoT sample.

Run from the repository root:
    python examples/run_demo.py
"""
import torch
from dbn_ids_defense import (prepare, DBNClassifier, fgsm,
                             adversarial_train, evaluate, class_weights)

torch.manual_seed(42)

d = prepare("data/ciciot_sample.csv", top_k=15)
cw = class_weights(d["y_train"], d["n_classes"])

# 1) Build a DBN (RBM pretraining + supervised fine-tuning)
model = DBNClassifier(d["n_features"], [100, 80], d["n_classes"])
model.pretrain(d["X_train"], epochs=10)
model.finetune(d["X_train"], d["y_train"], epochs=30, class_weight=cw)

# 2) Evaluate clean vs FGSM
print("clean :", evaluate(model, d["X_test"], d["y_test"]))
X_adv = fgsm(model, d["X_test"], d["y_test"], eps=0.1)
print("fgsm  :", evaluate(model, X_adv, d["y_test"]))

# 3) Defend with adversarial training, then re-evaluate
adversarial_train(model, d["X_train"], d["y_train"], eps=0.1, epochs=30, class_weight=cw)
X_adv2 = fgsm(model, d["X_test"], d["y_test"], eps=0.1)
print("fgsm (defended):", evaluate(model, X_adv2, d["y_test"]))
