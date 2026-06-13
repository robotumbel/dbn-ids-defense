#!/usr/bin/env bash
# Reproducible entry point (e.g. for CodeOcean). Runs the full pipeline on the
# bundled CIC-IoT sample: IG feature selection -> DBN -> FGSM -> defence.
set -e
python -m dbn_ids_defense \
    --data data/ciciot_sample.csv \
    --top-k 15 --hidden 100 80 \
    --pre-epochs 15 --ft-epochs 50 \
    --eps 0.1 --defense --balanced
