from stable_baselines3 import PPO
from backend.control.rl_env import ThermalEnv
import numpy as np
import os


class RLAgent:
    """RL Agent wrapper for thermal rod control."""

    def __init__(self, model_path="ppo_thermal_rod"):
        self.model_path = model_path
        self.env = ThermalEnv(curriculum_phase=2)  # Full objective for inference
        self.model = None
        self.ambient_temp = 25.0
        self.target_temp = 50.0
        self.current_step = 0
        self.max_steps = 1000

    def _normalize_obs(self, raw_temps):
        """
        Convert raw temperature readings to normalized observation format.
        Must match the format expected by the trained model.
        """
        # Normalize temperatures: ambient=0, target=1
        normalized_temps = (raw_temps - self.ambient_temp) / (self.target_temp - self.ambient_temp)

        mean_temp = np.mean(raw_temps)
        mean_normalized = (mean_temp - self.ambient_temp) / (self.target_temp - self.ambient_temp)
        time_fraction = self.current_step / self.max_steps

        obs = np.concatenate([
            normalized_temps,
            [mean_normalized, time_fraction]
        ]).astype(np.float32)

        return obs

    def reset(self):
        """Reset agent state for new episode."""
        self.current_step = 0

    def train(self, total_timesteps=10000):
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=1)

        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)

    def predict(self, observation, deterministic: bool = True):
        """
        Return an action for the given observation.

        Args:
            observation: Raw temperature readings (10 values) from sensors.
            deterministic: Whether to use deterministic policy output.

        Returns:
            action: Heat input in Watts [0, 50]
            state: Internal state (None for PPO)
        """
        if self.model is None:
            if os.path.exists(self.model_path + ".zip"):
                print(f"Loading RL model from {self.model_path}...")
                self.model = PPO.load(self.model_path, env=self.env)
                print("RL Model loaded successfully.")
            else:
                print(f"RL Model not found at {self.model_path}.zip. Using zero-action fallback.")
                return np.array([0.0], dtype=np.float32), None

        # Check if observation needs normalization (raw temps vs already normalized)
        if len(observation) == 10:
            # Raw temperature readings - need to normalize
            obs = self._normalize_obs(observation)
        elif len(observation) == 12:
            # Already normalized observation
            obs = observation
        else:
            print(f"Warning: Unexpected observation shape {observation.shape}")
            obs = observation

        self.current_step += 1

        action, state = self.model.predict(obs, deterministic=deterministic)
        return action, state
