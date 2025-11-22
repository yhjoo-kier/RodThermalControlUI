from stable_baselines3 import PPO
from backend.control.rl_env import ThermalEnv
import os

class RLAgent:
    def __init__(self, model_path="ppo_thermal_rod"):
        self.model_path = model_path
        self.env = ThermalEnv()
        self.model = None
        
    def train(self, total_timesteps=10000):
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        
    def predict(self, observation, deterministic: bool = False):
        """Return an action for the given observation.

        Args:
            observation: Current observation from the environment.
            deterministic: Whether to use a deterministic policy output. Defaults
                to ``False`` so stochastic actions can be used to avoid a degenerate
                zero-control policy when the learned mean is near zero.
        """

        if self.model is None:
            if os.path.exists(self.model_path + ".zip"):
                # Load with environment to ensure correct action scaling
                print(f"Loading RL model from {self.model_path}...")
                self.model = PPO.load(self.model_path, env=self.env)
                print("RL Model loaded successfully.")
            else:
                # Fallback if not trained
                print(f"RL Model not found at {self.model_path}.zip. Using zero-action fallback.")
                import numpy as np
                return np.array([0.0], dtype=np.float32), None

        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
