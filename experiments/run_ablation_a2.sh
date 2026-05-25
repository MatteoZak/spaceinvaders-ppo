#!/bin/bash
# Ablation A2: No frame stacking (1 frame instead of 4)
python train.py --config configs/default.yaml --no-frame-stack --run-name ppo_no-frame-stack_seed42
