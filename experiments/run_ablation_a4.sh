#!/bin/bash
# Ablation A4: Constant learning rate (no decay)
python train.py --config configs/default.yaml --no-lr-decay --run-name ppo_no-lr-decay_seed42
