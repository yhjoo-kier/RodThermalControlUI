import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.physics.heat_equation import ThermalRod


class ThermalEnv(gym.Env):
    """
    Thermal Rod Control Environment - Version 3

    Key design principles:
    1. Temperature achievement is PRIMARY goal
    2. Uniformity only matters once temperature is reached
    3. Strong incentive to use heat (action reward in early phase)
    4. Potential-based shaping for smooth gradients
    """

    def __init__(self, target_temp=50.0, curriculum_phase=2):
        super(ThermalEnv, self).__init__()
        self.rod = ThermalRod()
        self.target_temp = target_temp
        self.ambient_temp = 25.0
        self.dt = 0.05
        self.max_steps = 1000
        self.current_step = 0
        self.curriculum_phase = curriculum_phase

        # For tracking improvement
        self.prev_mean_temp = None

        # Action space: Heat input [0, 50] Watts
        self.action_space = spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32)

        # Observation space: Normalized sensor readings + meta info
        self.observation_space = spaces.Box(
            low=-1.0, high=3.0, shape=(12,), dtype=np.float32
        )

    def _normalize_temps(self, temps):
        """Normalize temperatures: ambient=0, target=1"""
        return (temps - self.ambient_temp) / (self.target_temp - self.ambient_temp)

    def _get_obs(self):
        raw_temps = self.rod.get_sensor_readings()
        normalized_temps = self._normalize_temps(raw_temps)
        mean_temp = np.mean(raw_temps)
        mean_normalized = (mean_temp - self.ambient_temp) / (self.target_temp - self.ambient_temp)
        time_fraction = self.current_step / self.max_steps

        return np.concatenate([
            normalized_temps,
            [mean_normalized, time_fraction]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rod = ThermalRod()
        self.current_step = 0
        self.prev_mean_temp = self.ambient_temp
        return self._get_obs(), {}

    def step(self, action):
        q_in = float(np.clip(action[0], 0, 50))
        self.rod.step(self.dt, q_in)
        self.current_step += 1

        raw_temps = self.rod.get_sensor_readings()
        obs = self._get_obs()

        mean_temp = np.mean(raw_temps)
        std_temp = np.std(raw_temps)

        # ====== REWARD COMPUTATION ======

        # 1. Temperature improvement reward (main driver)
        # Reward for getting closer to target
        temp_improvement = mean_temp - self.prev_mean_temp
        self.prev_mean_temp = mean_temp

        # Scale improvement reward - stronger when far from target
        distance_to_target = self.target_temp - mean_temp
        if distance_to_target > 0:
            # Not yet at target - reward improvement heavily
            improvement_reward = temp_improvement * 2.0
        else:
            # At or above target - penalize overshooting
            improvement_reward = -abs(temp_improvement) * 1.0

        # 2. Proximity bonus - continuous reward for being close to target
        temp_error = abs(mean_temp - self.target_temp)
        if temp_error < 25.0:
            # Scale from 0 to 1 as we approach target
            proximity = 1.0 - temp_error / 25.0
            proximity_bonus = 0.2 * proximity ** 2  # Quadratic scaling
        else:
            proximity_bonus = 0.0

        # 3. Uniformity penalty - MINIMAL, only at very close range
        # Key insight: With single-point heating, variance is UNAVOIDABLE
        # Don't let uniformity penalty discourage heating!
        if temp_error < 3.0:  # Only care about uniformity when VERY close to target
            closeness = 1.0 - temp_error / 3.0
            uniformity_penalty = 0.002 * std_temp * closeness  # Very small penalty
        else:
            uniformity_penalty = 0.0  # No penalty when far from target

        # 4. Action incentive - encourage using more heat when far from target
        # Scale with distance to target: more heat needed when cold
        if distance_to_target > 10.0:
            # Far from target - strongly encourage max heat
            action_reward = 0.02 * (q_in / 50.0)
        elif distance_to_target > 5.0:
            # Getting closer - moderate encouragement
            action_reward = 0.01 * (q_in / 50.0)
        else:
            # Near target - no artificial incentive
            action_reward = 0.0

        # 5. Small step penalty to encourage efficiency
        step_penalty = 0.001

        # Total reward
        reward = improvement_reward + proximity_bonus - uniformity_penalty + action_reward - step_penalty

        # 6. Terminal bonus for achieving goal
        terminated = False
        if temp_error < 2.0 and std_temp < 2.0:
            reward += 20.0  # Big bonus for success
            terminated = True

        truncated = self.current_step >= self.max_steps

        info = {
            "mean_temp": mean_temp,
            "std_temp": std_temp,
            "variance": std_temp ** 2,
            "heat_input": q_in,
            "improvement_reward": improvement_reward,
            "proximity_bonus": proximity_bonus
        }

        return obs, reward, terminated, truncated, info


class ThermalEnvCurriculum(ThermalEnv):
    """Environment with automatic curriculum progression."""

    def __init__(self, target_temp=50.0):
        super().__init__(target_temp=target_temp, curriculum_phase=0)
        self.episode_count = 0
        self.phase_thresholds = [20, 50]

    def reset(self, seed=None, options=None):
        self.episode_count += 1

        if self.episode_count > self.phase_thresholds[1]:
            self.curriculum_phase = 2
        elif self.episode_count > self.phase_thresholds[0]:
            self.curriculum_phase = 1
        else:
            self.curriculum_phase = 0

        return super().reset(seed=seed, options=options)
