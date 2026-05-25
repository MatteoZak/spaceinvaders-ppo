import gymnasium as gym
import numpy as np
from collections import deque
import cv2


class NoopResetEnv(gym.Wrapper):
    """Esegue un numero casuale di azioni NOOP all'inizio di ogni episodio."""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Ripete ogni azione per `skip` frame e restituisce il max dei 2 frame finali."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(list(self._obs_buffer)), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Tratta la perdita di una vita come fine episodio (ma non reset dell'ambiente)."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Preme FIRE per iniziare il gioco in ambienti che lo richiedono."""
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class GrayscaleAndResizeEnv(gym.ObservationWrapper):
    """Converte in grayscale e ridimensiona a 84x84."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class FrameStackEnv(gym.ObservationWrapper):
    """Stacca n_frames frame consecutivi lungo la dimensione canali."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames = deque(maxlen=n_frames)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, n_frames * c), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self._frames), axis=-1)


class ClipRewardEnv(gym.RewardWrapper):
    """Clippa il reward a {-1, 0, +1}."""
    def reward(self, reward):
        return np.sign(reward)


def make_env(env_id, seed, rank, frame_stack=True):
    """
    Crea un singolo env Atari con preprocessing standard.
    frame_stack=False per l'ablation A2.
    """
    def _init():
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make(env_id, render_mode=None)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = GrayscaleAndResizeEnv(env)
        if frame_stack:
            env = FrameStackEnv(env, n_frames=4)
        env = ClipRewardEnv(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + rank)
        return env
    return _init
