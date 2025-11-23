import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.physics.heat_equation import ThermalRod

class ThermalEnv(gym.Env):
    def __init__(
        self,
        target_temp=50.0,
        mean_error_weight: float = 2.0,
        variance_weight: float = 0.5,
        progress_weight: float = 0.1,
        steady_tolerance: float = 1.0,
        steady_bonus: float = 2.0,
        energy_penalty: float = 1e-3,
    ):
        super(ThermalEnv, self).__init__()
        self.rod = ThermalRod()
        self.target_temp = target_temp
        self.dt = 0.05
        self.max_steps = 1000  # 50 seconds per episode (was 500)
        self.current_step = 0

        # Reward shaping parameters
        self.mean_error_weight = mean_error_weight
        self.variance_weight = variance_weight
        self.progress_weight = progress_weight
        self.steady_tolerance = steady_tolerance
        self.steady_bonus = steady_bonus
        self.energy_penalty = energy_penalty
        self.prev_mean_temp = self.rod.props.ambient_temp
        
        # Action space: Heat input [0, 50] Watts
        self.action_space = spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32)
        
        # Observation space: 10 sensor readings
        # Temp range: [0, 200] C (safe bounds)
        self.observation_space = spaces.Box(low=0.0, high=200.0, shape=(10,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rod = ThermalRod() # Reset physics
        self.current_step = 0
        # Track the last mean temperature for progress-based reward shaping
        self.prev_mean_temp = np.mean(self.rod.get_sensor_readings())
        return self.rod.get_sensor_readings().astype(np.float32), {}
        
    def step(self, action):
        q_in = float(action[0])
        self.rod.step(self.dt, q_in)
        self.current_step += 1
        
        obs = self.rod.get_sensor_readings().astype(np.float32)
        
        # Reward function
        mean_temp = np.mean(obs)
        variance = np.var(obs)
        mean_error = mean_temp - self.target_temp

        # Balance tracking the mean temperature and improving uniformity.
        reward = -(
            self.mean_error_weight * mean_error**2
            + self.variance_weight * variance
        )

        # Encourage moving the temperature upward when starting cold
        # and reward progress toward the target to avoid a zero-action policy.
        reward += self.progress_weight * (mean_temp - self.prev_mean_temp)

        # Provide a small bonus for staying near the target to stabilize control.
        if abs(mean_error) < self.steady_tolerance:
            reward += self.steady_bonus

        # Discourage unnecessarily large heat input without overwhelming the
        # incentive to reach the target temperature.
        reward -= self.energy_penalty * (q_in ** 2)

        # Update for next step progress calculation
        self.prev_mean_temp = mean_temp
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {"mean_temp": mean_temp, "variance": variance}
