"""Quick RL controller verification.

Loads the pretrained PPO thermal rod model and runs a short rollout
inside the ThermalEnv to confirm that the RL policy produces non-zero
heat inputs. Prints summary statistics of the sampled actions.
"""
from __future__ import annotations

import numpy as np

from backend.control.rl_agent import RLAgent


def verify_rl_actions(num_steps: int = 20) -> float:
    """Run the pretrained RL policy for ``num_steps`` and report mean action.

    Returns the mean heat input so callers/tests can assert non-zero control.
    """

    agent = RLAgent(model_path="ppo_thermal_rod")

    # Use the agent's environment for consistency with training.
    obs, _ = agent.env.reset()
    actions = []

    for _ in range(num_steps):
        action, _ = agent.predict(obs)
        heat_input = float(action[0])
        actions.append(heat_input)

        obs, _, terminated, truncated, _ = agent.env.step(np.array([heat_input], dtype=np.float32))
        if terminated or truncated:
            obs, _ = agent.env.reset()

    actions_array = np.array(actions, dtype=float)
    mean_action = float(actions_array.mean())

    print(f"Collected {len(actions)} actions. Min: {actions_array.min():.3f}, "
          f"Max: {actions_array.max():.3f}, Mean: {mean_action:.3f}")
    return mean_action


if __name__ == "__main__":
    mean_heat = verify_rl_actions()
    if np.isclose(mean_heat, 0.0):
        raise SystemExit("RL verification failed: mean action is zero.")
    print("RL verification passed: policy outputs non-zero control.")
