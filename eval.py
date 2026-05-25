import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import imageio
import ale_py
from utils.env_wrappers import make_env
from ppo import AgentCNN


def evaluate(checkpoint_path, env_id, n_episodes=10, frame_stack=True, record_video=True, video_path="report/figures/agent.mp4"):
    gym.register_envs(ale_py)

    env_fn = make_env(env_id, seed=0, rank=0, frame_stack=frame_stack)
    env = env_fn()

    n_actions = env.action_space.n
    n_channels = 4 if frame_stack else 1
    agent = AgentCNN(n_actions=n_actions, n_channels=n_channels, orthogonal_init=True)
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    agent.eval()

    returns = []
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs = np.transpose(obs, (2, 0, 1))
        done = False
        ep_return = 0.0
        record_this = record_video and ep == 0

        if record_this:
            env_vis = gym.make(env_id, render_mode="rgb_array")
            obs_vis, _ = env_vis.reset()
            frames.append(env_vis.render())

        while not done:
            obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            action = action.item()

            obs_raw, reward, terminated, truncated, _ = env.step(action)
            obs = np.transpose(obs_raw, (2, 0, 1))
            ep_return += reward
            done = terminated or truncated

            if record_this:
                env_vis.step(action)
                frames.append(env_vis.render())

        returns.append(ep_return)
        print(f"Episode {ep+1}: return = {ep_return:.1f}")

        if record_this:
            env_vis.close()

    env.close()
    print(f"\nMean return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    if record_video and frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video salvato: {video_path}")

    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--no-frame-stack", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    evaluate(
        args.checkpoint,
        args.env_id,
        n_episodes=args.n_episodes,
        frame_stack=not args.no_frame_stack,
        record_video=not args.no_video,
    )
