import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.physics.heat_equation import ThermalRod

class ThermalEnv(gym.Env):
    def __init__(self, target_temp=50.0):
        super(ThermalEnv, self).__init__()
        self.rod = ThermalRod()
        self.target_temp = target_temp
        self.dt = 0.05
        self.max_steps = 1000  # 50 seconds per episode (was 500)
        self.current_step = 0
        
        # Action space: Heat input [0, 50] Watts
        self.action_space = spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32)
        
        # Observation space: 10 sensor readings
        # Temp range: [0, 200] C (safe bounds)
        self.observation_space = spaces.Box(low=0.0, high=200.0, shape=(10,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rod = ThermalRod() # Reset physics
        self.current_step = 0
        return self.rod.get_sensor_readings().astype(np.float32), {}
        
    def step(self, action):
        q_in = float(action[0])
        self.rod.step(self.dt, q_in)
        self.current_step += 1
        
        obs = self.rod.get_sensor_readings().astype(np.float32)
        
        # Reward function
        # Goal: Minimize variance AND keep mean close to target
        mean_temp = np.mean(obs)
        variance = np.var(obs)
        
        # Penalty for deviation from target mean
        mean_error = (mean_temp - self.target_temp)**2
        
        # Penalty for variance (non-uniformity)
        uniformity_error = variance
        
        # Combined reward (negative cost)
        # Weighting: We want uniformity, but mean must be correct.
        reward = -(1.0 * mean_error + 5.0 * uniformity_error)
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {"mean_temp": mean_temp, "variance": variance}
