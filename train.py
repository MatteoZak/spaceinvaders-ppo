import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from utils.env_wrappers import make_env
from utils.logger import Logger
from ppo import AgentCNN, RewardNormalizer, compute_gae


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-reward-norm", action="store_true", help="Ablation A1")
    parser.add_argument("--no-frame-stack", action="store_true", help="Ablation A2")
    parser.add_argument("--no-orthogonal-init", action="store_true", help="Ablation A3")
    parser.add_argument("--no-lr-decay", action="store_true", help="Ablation A4")
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_run_name(config, args):
    name = "ppo_spaceinvaders"
    if args.no_reward_norm:
        name += "_no-reward-norm"
    elif args.no_frame_stack:
        name += "_no-frame-stack"
    elif args.no_orthogonal_init:
        name += "_no-orth-init"
    elif args.no_lr_decay:
        name += "_no-lr-decay"
    else:
        name += "_baseline"
    name += f"_seed{config['seed']}"
    return name


def train():
    args = parse_args()
    config = load_config(args.config)

    if args.no_reward_norm:
        config["reward_norm"] = False
    if args.no_frame_stack:
        config["frame_stack"] = False
    if args.no_orthogonal_init:
        config["orthogonal_init"] = False
    if args.no_lr_decay:
        config["lr_decay"] = False

    run_name = args.run_name or make_run_name(config, args)
    logger = Logger(config, run_name)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    frame_stack = config.get("frame_stack", True)
    envs = SyncVectorEnv([
        make_env(config["env_id"], config["seed"], i, frame_stack=frame_stack)
        for i in range(config["num_envs"])
    ])

    n_channels = 4 if frame_stack else 1
    obs_shape = (n_channels, 84, 84)
    n_actions = envs.single_action_space.n

    orthogonal_init = config.get("orthogonal_init", True)
    agent = AgentCNN(n_actions=n_actions, n_channels=n_channels, orthogonal_init=orthogonal_init).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

    reward_norm = config.get("reward_norm", True)
    reward_normalizer = RewardNormalizer(gamma=config["gamma"]) if reward_norm else None

    num_steps = config["num_steps"]
    num_envs = config["num_envs"]
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // config["num_minibatches"]
    total_timesteps = config["total_timesteps"]
    num_updates = total_timesteps // batch_size

    obs_buf = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.uint8)
    actions_buf = np.zeros((num_steps, num_envs), dtype=np.int64)
    logprobs_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    rewards_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    dones_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    values_buf = np.zeros((num_steps, num_envs), dtype=np.float32)

    next_obs, _ = envs.reset(seed=config["seed"])
    next_obs = np.transpose(next_obs, (0, 3, 1, 2))
    next_done = np.zeros(num_envs, dtype=np.float32)

    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        if config.get("lr_decay", True):
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * config["learning_rate"]

        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
                action, logprob, _, value = agent.get_action_and_value(obs_t)

            actions_buf[step] = action.cpu().numpy()
            logprobs_buf[step] = logprob.cpu().numpy()
            values_buf[step] = value.cpu().numpy().flatten()

            next_obs_raw, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done = (terminated | truncated).astype(np.float32)
            next_obs = np.transpose(next_obs_raw, (0, 3, 1, 2))
            rewards_buf[step] = reward

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_ret = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        logger.log({
                            "charts/episodic_return": ep_ret,
                            "charts/episodic_length": ep_len,
                        }, step=global_step)

        if reward_normalizer is not None:
            rewards_buf_norm = reward_normalizer.normalize(rewards_buf.copy(), dones_buf.copy())
        else:
            rewards_buf_norm = rewards_buf

        with torch.no_grad():
            next_val = agent.get_value(
                torch.tensor(next_obs, dtype=torch.float32).to(device)
            ).cpu().numpy().flatten()

        advantages, returns = compute_gae(
            rewards_buf_norm, values_buf, dones_buf, next_val,
            gamma=config["gamma"], gae_lambda=config["gae_lambda"]
        )

        b_obs = torch.tensor(obs_buf.reshape((-1,) + obs_shape), dtype=torch.float32).to(device)
        b_actions = torch.tensor(actions_buf.reshape(-1), dtype=torch.long).to(device)
        b_logprobs = torch.tensor(logprobs_buf.reshape(-1), dtype=torch.float32).to(device)
        b_advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32).to(device)
        b_returns = torch.tensor(returns.reshape(-1), dtype=torch.float32).to(device)
        b_values = torch.tensor(values_buf.reshape(-1), dtype=torch.float32).to(device)

        clipfracs = []
        ratio = None
        log_ratio = None
        for epoch in range(config["update_epochs"]):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                mb_adv = b_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                log_ratio = new_logprob - b_logprobs[mb_idx]
                ratio = log_ratio.exp()
                clipfracs.append(((ratio - 1.0).abs() > config["clip_coef"]).float().mean().item())

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()

        sps = int(global_step / (time.time() - start_time))
        logger.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": float(explained_var),
            "charts/SPS": sps,
        }, step=global_step)

        if update % 50 == 0:
            print(f"Update {update}/{num_updates} | steps={global_step:,} | SPS={sps}")

    envs.close()
    logger.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.state_dict(), f"checkpoints/{run_name}.pt")
    print(f"Modello salvato: checkpoints/{run_name}.pt")


if __name__ == "__main__":
    train()
