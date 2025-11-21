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
        
    def predict(self, observation):
        if self.model is None:
            if os.path.exists(self.model_path + ".zip"):
                self.model = PPO.load(self.model_path)
            else:
                # Fallback if not trained
                import numpy as np
                return np.array([0.0], dtype=np.float32), None
        
        action, state = self.model.predict(observation, deterministic=True)
        return action, state
