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
        self.prev_mean_temp = self.target_temp  # initialized in reset
        self.prev_action = 0.0
        
        # Action space: Heat input [0, 50] Watts
        self.action_space = spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32)
        
        # Observation space: 10 sensor readings
        # Temp range: [0, 200] C (safe bounds)
        self.observation_space = spaces.Box(low=0.0, high=200.0, shape=(10,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rod = ThermalRod() # Reset physics
        self.current_step = 0
        obs = self.rod.get_sensor_readings().astype(np.float32)
        self.prev_mean_temp = float(np.mean(obs))
        self.prev_action = 0.0
        return obs, {}
        
    def step(self, action):
        q_in = float(action[0])
        self.rod.step(self.dt, q_in)
        self.current_step += 1
        
        obs = self.rod.get_sensor_readings().astype(np.float32)
        
        # Reward function
        # Goal: Minimize variance AND keep mean close to target while rewarding
        # temperature climb toward the setpoint. The previous reward was purely
        # quadratic in the temperature error, which produced large negative
        # values early in an episode and encouraged the policy to sit at zero
        # output. The new shaping gives a dense gradient as the rod heats up
        # and gently penalizes aggressive action changes.
        mean_temp = np.mean(obs)
        variance = np.var(obs)

        # Dense tracking penalty using absolute error to avoid huge early costs
        tracking_error = abs(mean_temp - self.target_temp)

        # Encourage heating progress toward the target
        progress = max(mean_temp - self.prev_mean_temp, 0.0)

        # Penalize variance and action dithering to keep the profile smooth
        uniformity_penalty = variance
        action_rate_penalty = abs(q_in - self.prev_action) / self.action_space.high[0]

        reward = (
            2.0 * progress
            - 1.0 * tracking_error
            - 5.0 * uniformity_penalty
            - 0.05 * action_rate_penalty
        )

        self.prev_mean_temp = mean_temp
        self.prev_action = q_in

        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "mean_temp": mean_temp,
            "variance": variance,
            "progress": progress,
            "tracking_error": tracking_error,
        }

        return obs, reward, terminated, truncated, info
