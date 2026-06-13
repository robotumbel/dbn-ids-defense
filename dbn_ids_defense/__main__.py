"""End-to-end command-line interface.

Pipeline: load CSV -> Information-Gain feature selection -> DBN pretraining
(RBM stack) + supervised fine-tuning -> evaluate on clean, FGSM-attacked,
and (optionally) adversarially-defended test data.

Example
-------
    python -m dbn_ids_defense --data data/ciciot_sample.csv \
        --top-k 15 --hidden 100 80 --eps 0.1 --defense
"""
from __future__ import annotations
import argparse
import torch

from .data import prepare, class_weights
from .dbn import DBNClassifier
from .attacks import fgsm
from .defense import adversarial_train
from .metrics import evaluate


def main():
    ap = argparse.ArgumentParser(description="DBN defence against adversarial "
                                             "attacks on IoT intrusion detection.")
    ap.add_argument("--data", required=True, help="path to the IoT IDS CSV")
    ap.add_argument("--top-k", type=int, default=15, help="features kept by Information Gain")
    ap.add_argument("--hidden", type=int, nargs="+", default=[100, 80], help="RBM/hidden widths")
    ap.add_argument("--pre-epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.1, help="FGSM perturbation budget")
    ap.add_argument("--defense", action="store_true", help="apply adversarial training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balanced", action="store_true", help="use inverse-frequency class weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = prepare(args.data, top_k=args.top_k, seed=args.seed)
    cw = class_weights(d["y_train"], d["n_classes"]) if args.balanced else None
    print(f"data: {d['n_features']} features (IG top-{args.top_k}), "
          f"{d['n_classes']} classes, "
          f"{len(d['y_train'])} train / {len(d['y_test'])} test")

    model = DBNClassifier(d["n_features"], args.hidden, d["n_classes"])
    print("pretraining DBN (RBM stack) ...")
    model.pretrain(d["X_train"], epochs=args.pre_epochs)
    print("fine-tuning ...")
    model.finetune(d["X_train"], d["y_train"], epochs=args.ft_epochs, class_weight=cw)

    clean = evaluate(model, d["X_test"], d["y_test"])
    X_adv = fgsm(model, d["X_test"], d["y_test"], eps=args.eps)
    adv = evaluate(model, X_adv, d["y_test"])
    print(f"\nClean test         : {clean}")
    print(f"FGSM (eps={args.eps}) test : {adv}")

    if args.defense:
        print("\napplying adversarial-training defence ...")
        adversarial_train(model, d["X_train"], d["y_train"], eps=args.eps,
                          epochs=args.ft_epochs, class_weight=cw)
        clean2 = evaluate(model, d["X_test"], d["y_test"])
        X_adv2 = fgsm(model, d["X_test"], d["y_test"], eps=args.eps)
        adv2 = evaluate(model, X_adv2, d["y_test"])
        print(f"Clean test (defended)         : {clean2}")
        print(f"FGSM test  (defended)         : {adv2}")
        print(f"\nRobustness gain under FGSM: "
              f"accuracy {adv['accuracy']:.3f} -> {adv2['accuracy']:.3f}")


if __name__ == "__main__":
    main()
