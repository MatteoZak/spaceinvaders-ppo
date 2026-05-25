#!/bin/bash
# Ablation A1: No reward normalization
python train.py --config configs/default.yaml --no-reward-norm --run-name ppo_no-reward-norm_seed42
