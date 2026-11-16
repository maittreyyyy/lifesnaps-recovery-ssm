#!/bin/bash
# Reproduce every stage 1 result from the raw CSVs. Run from the repository root.
set -e
python src/prepare.py
python src/lssm.py
python src/eval_stage1.py
python src/stage1_checks.py
python src/stage1_stats.py
