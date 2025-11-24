import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.physics.heat_equation import ThermalRod

class ThermalEnv(gym.Env):
    def __init__(self, target_temp=50.0, reward_type="shaped"):
        super(ThermalEnv, self).__init__()
        self.rod = ThermalRod()
        self.target_temp = target_temp
        self.dt = 0.05
        self.max_steps = 1000  # 50 seconds per episode (was 500)
        self.current_step = 0
        self.reward_type = reward_type  # "simple", "shaped", or "dense"

        # Track previous state for reward shaping
        self.prev_mean_error = None
        self.prev_variance = None

        # Action space: Heat input [0, 50] Watts
        self.action_space = spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32)

        # Observation space: 10 sensor readings
        # Temp range: [0, 200] C (safe bounds)
        self.observation_space = spaces.Box(low=0.0, high=200.0, shape=(10,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rod = ThermalRod() # Reset physics
        self.current_step = 0
        self.prev_mean_error = None
        self.prev_variance = None
        return self.rod.get_sensor_readings().astype(np.float32), {}
        
    def step(self, action):
        q_in = float(action[0])
        self.rod.step(self.dt, q_in)
        self.current_step += 1

        obs = self.rod.get_sensor_readings().astype(np.float32)

        # Calculate error metrics
        mean_temp = np.mean(obs)
        variance = np.var(obs)
        mean_error = abs(mean_temp - self.target_temp)
        std_dev = np.sqrt(variance)

        # Calculate reward based on selected type
        if self.reward_type == "simple":
            # Improved simple reward: normalized and scaled
            # Use exponential decay for errors to encourage getting close
            temp_reward = -mean_error / 10.0  # Scale down
            uniformity_reward = -std_dev / 5.0  # Scale down
            energy_penalty = -0.001 * q_in / 50.0  # Small penalty for energy use
            reward = temp_reward + uniformity_reward + energy_penalty

        elif self.reward_type == "shaped":
            # Reward shaping: bonus for improvement
            if self.prev_mean_error is not None:
                mean_improvement = self.prev_mean_error - mean_error
                variance_improvement = self.prev_variance - variance

                # Base reward
                temp_reward = -mean_error / 10.0
                uniformity_reward = -std_dev / 5.0

                # Bonus for improvement (positive reinforcement)
                improvement_bonus = 0.5 * mean_improvement + 0.2 * variance_improvement

                # Energy penalty
                energy_penalty = -0.001 * q_in / 50.0

                reward = temp_reward + uniformity_reward + improvement_bonus + energy_penalty
            else:
                # First step: use simple reward
                reward = -mean_error / 10.0 - std_dev / 5.0 - 0.001 * q_in / 50.0

            self.prev_mean_error = mean_error
            self.prev_variance = variance

        else:  # "dense"
            # Dense reward: more frequent positive signals
            # Zone-based reward: different rewards for different proximity to goal
            if mean_error < 1.0 and std_dev < 0.5:
                # Excellent: within 1°C and very uniform
                zone_reward = 10.0
            elif mean_error < 5.0 and std_dev < 2.0:
                # Good: within 5°C and fairly uniform
                zone_reward = 5.0
            elif mean_error < 10.0:
                # Acceptable: within 10°C
                zone_reward = 1.0
            else:
                # Far from target
                zone_reward = -mean_error / 20.0

            # Additional continuous component
            temp_component = -0.1 * mean_error
            uniformity_component = -0.5 * std_dev
            energy_penalty = -0.001 * q_in / 50.0

            reward = zone_reward + temp_component + uniformity_component + energy_penalty

        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "mean_temp": mean_temp,
            "variance": variance,
            "std_dev": std_dev,
            "mean_error": mean_error,
            "q_in": q_in
        }

        return obs, reward, terminated, truncated, info
