import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, orthogonal=True):
    """Inizializzazione ortogonale (best practice PPO) o random standard."""
    if orthogonal:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    else:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentCNN(nn.Module):
    """
    Nature DQN CNN con actor e critic heads condivisi.
    n_channels: 4 per frame stack, 1 per ablation A2 (no frame stack).
    """
    def __init__(self, n_actions, n_channels=4, orthogonal_init=True):
        super().__init__()
        init = lambda layer, std=np.sqrt(2): layer_init(layer, std=std, orthogonal=orthogonal_init)

        self.network = nn.Sequential(
            init(nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init(nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.actor = init(nn.Linear(512, n_actions), std=0.01)
        self.critic = init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)


class RunningMeanStd:
    """Traccia running mean e variance per normalizzare reward/osservazioni."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = m_2 / tot_count
        self.count = tot_count


class RewardNormalizer:
    """
    Normalizza i reward tramite running mean/std dei discounted return.
    Disabilitato per ablation A1 (reward_norm=False).
    """
    def __init__(self, gamma=0.99):
        self.rms = RunningMeanStd()
        self.gamma = gamma
        self.returns = None

    def normalize(self, rewards, dones):
        if self.returns is None:
            self.returns = np.zeros(rewards.shape[0])
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        self.returns[dones.astype(bool)] = 0.0
        return rewards / np.sqrt(self.rms.var + 1e-8)


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """
    Calcola Generalized Advantage Estimation.
    rewards, values, dones: array di shape (num_steps, num_envs)
    next_value: array di shape (num_envs,)
    Restituisce advantages e returns di shape (num_steps, num_envs).
    """
    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns
