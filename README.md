# PPO State-of-the-Art on SpaceInvaders

DLAI 2025/2026 BYOP Exam Project — Gaming and AI / Training Dynamics

Study of training dynamics of modern PPO on `ALE/SpaceInvaders-v5`, with systematic ablation study of 4 key best practices.

## Setup

```bash
pip install -r requirements.txt
```

## Training

**Baseline (all best practices enabled):**
```bash
bash experiments/run_baseline.sh
```

**Ablation A1 — No Reward Normalization:**
```bash
bash experiments/run_ablation_a1.sh
```

**Ablation A2 — No Frame Stacking:**
```bash
bash experiments/run_ablation_a2.sh
```

**Ablation A3 — No Orthogonal Initialization:**
```bash
bash experiments/run_ablation_a3.sh
```

**Ablation A4 — Constant Learning Rate:**
```bash
bash experiments/run_ablation_a4.sh
```

Each run: ~10M environment steps (~2-3h on Colab GPU or Apple M4).

## Evaluation

```bash
python eval.py --checkpoint checkpoints/ppo_baseline_seed42.pt
python eval.py --checkpoint checkpoints/ppo_baseline_seed42.pt --no-video  # skip video
```

## Analysis

After all runs complete:
```bash
jupyter notebook notebooks/analysis.ipynb
```

Produces: reward curves, value loss curves, entropy plots, ablation comparison table.

## Project Structure

```
spaceinvaders-ppo/
├── ppo.py              # AgentCNN (Nature DQN), RewardNormalizer, compute_gae
├── train.py            # PPO training loop with argparse flags for ablations
├── eval.py             # Evaluation + video recording
├── utils/
│   ├── env_wrappers.py # Atari preprocessing (NoopReset, MaxAndSkip, FrameStack, ...)
│   └── logger.py       # W&B / TensorBoard unified logger
├── configs/
│   └── default.yaml    # Hyperparameters (baseline)
├── experiments/        # Shell scripts to reproduce all runs
├── notebooks/
│   └── analysis.ipynb  # Training dynamics analysis and plots
└── report/figures/     # Exported figures for the report
```

## PPO Best Practices (Baseline)

| Component | Value |
|---|---|
| Vectorized envs | 8 parallel |
| Frame stacking | 4 frames (84×84 grayscale) |
| Reward normalization | Running mean/std of discounted returns |
| Advantage normalization | Per minibatch |
| Orthogonal initialization | CNN + FC layers |
| Linear LR decay | 2.5e-4 → 0 |
| GAE | γ=0.99, λ=0.95 |
| PPO clip | ε=0.1 |
| Entropy bonus | c=0.01 |

## Ablation Study

| Ablation | Removed | Hypothesis |
|---|---|---|
| A1 | Reward normalization | Higher reward variance, unstable training |
| A2 | Frame stacking (1 frame) | Agent loses motion perception |
| A3 | Orthogonal init | Noisier loss curves early on |
| A4 | LR decay (constant) | Instability in late training |

## References

- Schulman et al. 2017 — Proximal Policy Optimization Algorithms
- Huang et al. 2022 — CleanRL: High-quality Single-file Implementations of Deep RL
- Andrychowicz et al. 2021 — What Matters in On-Policy Reinforcement Learning?
- Mnih et al. 2015 — Human-level control through deep reinforcement learning
