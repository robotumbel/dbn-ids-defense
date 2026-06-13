# dbn-ids-defense

A Deep Belief Network (DBN) toolkit for defending IoT intrusion detection
against adversarial (FGSM) attacks.

This is the reference implementation of the method in:

> Defence against adversarial attacks on IoT detection systems using deep
> belief network. *Indonesian Journal of Electrical Engineering and Computer
> Science (IJEECS)*, 35(2), 1073–1081, 2024.
> DOI: [10.11591/ijeecs.v35.i2.pp1073-1081](https://doi.org/10.11591/ijeecs.v35.i2.pp1073-1081)

## What it does
End-to-end pipeline for robust IoT intrusion detection:

1. **Information-Gain feature selection** — rank features by mutual
   information and keep the top *k*.
2. **Deep Belief Network** — greedy layer-wise RBM pretraining
   (contrastive divergence, CD-1) followed by supervised fine-tuning.
3. **FGSM attack** — generate adversarial flows that evade the detector.
4. **Adversarial-training defence** — re-train on clean + adversarial data
   to recover robustness.

## Install
```bash
pip install -r requirements.txt
```

## Quick start (CLI)
```bash
python -m dbn_ids_defense \
    --data data/ciciot_sample.csv \
    --top-k 15 --hidden 100 80 --eps 0.1 --defense --balanced
```
A small CIC-IoT2023-derived sample (`data/ciciot_sample.csv`) is bundled so
the pipeline runs out of the box. Replace `--data` with your full CSV (last
column = label) to reproduce the published results.

## Quick start (Python API)
```python
from dbn_ids_defense import prepare, DBNClassifier, fgsm, adversarial_train, evaluate
d = prepare("data/ciciot_sample.csv", top_k=15)
model = DBNClassifier(d["n_features"], [100, 80], d["n_classes"])
model.pretrain(d["X_train"]); model.finetune(d["X_train"], d["y_train"])
print(evaluate(model, d["X_test"], d["y_test"]))            # clean
X_adv = fgsm(model, d["X_test"], d["y_test"], eps=0.1)
print(evaluate(model, X_adv, d["y_test"]))                  # under attack
adversarial_train(model, d["X_train"], d["y_train"], eps=0.1)
```
See `examples/run_demo.py`.

## Repository layout
```
dbn_ids_defense/
  rbm.py        Restricted Boltzmann Machine (CD-1)
  dbn.py        DBN classifier: RBM pretraining + fine-tuning
  data.py       CSV loading, Information-Gain selection, scaling, split
  attacks.py    FGSM evasion attack
  defense.py    adversarial-training defence
  metrics.py    accuracy / macro-F1 / MCC
  __main__.py   command-line interface
data/ciciot_sample.csv     bundled sample
examples/run_demo.py       Python-API example
run.sh                     reproducible entry point
```

## Requirements
Python 3.9+, PyTorch, scikit-learn, pandas, numpy (see `requirements.txt`).

## License
MIT — see [LICENSE](LICENSE).

## Citation
If you use this software, please cite the article above (DOI
`10.11591/ijeecs.v35.i2.pp1073-1081`).
