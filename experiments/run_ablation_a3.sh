#!/bin/bash
# Ablation A3: No orthogonal initialization
python train.py --config configs/default.yaml --no-orthogonal-init --run-name ppo_no-orth-init_seed42
