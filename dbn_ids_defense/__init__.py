"""dbn_ids_defense: a Deep Belief Network toolkit for defending IoT
intrusion detection against adversarial (FGSM) attacks.

Reference implementation of:
  "Defence against adversarial attacks on IoT detection systems using deep
  belief network," IJEECS, vol. 35, no. 2, pp. 1073-1081, 2024.
  DOI: 10.11591/ijeecs.v35.i2.pp1073-1081
"""
from .rbm import RBM
from .dbn import DBNClassifier, build_dbn
from .data import prepare, load_csv, class_weights
from .attacks import fgsm
from .defense import adversarial_train
from .metrics import evaluate, report

__version__ = "1.0.0"
__all__ = ["RBM", "DBNClassifier", "build_dbn", "prepare", "load_csv",
           "class_weights", "fgsm", "adversarial_train", "evaluate", "report"]
