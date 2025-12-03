#!/usr/bin/env python3
"""
Test script for RL model performance analysis.
Plots temperature trajectory over a full episode.
"""

import numpy as np
from backend.control.rl_env import ThermalEnv
from stable_baselines3 import PPO


def test_rl_model(model_path="ppo_thermal_rod", steps=1000):
    """Run full episode and collect data."""
    print("=" * 60)
    print("RL Model Performance Test")
    print("=" * 60)

    # Load model
    env = ThermalEnv(curriculum_phase=2)
    model = PPO.load(model_path, env=env)

    # Reset environment
    obs, _ = env.reset()

    # Data collection
    times = []
    mean_temps = []
    std_temps = []
    heat_inputs = []
    rewards = []

    total_reward = 0
    dt = 0.05

    print(f"\nRunning {steps} steps ({steps * dt:.1f} seconds)...")
    print("-" * 60)

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        times.append(step * dt)
        mean_temps.append(info['mean_temp'])
        std_temps.append(info['std_temp'])
        heat_inputs.append(info['heat_input'])
        rewards.append(reward)
        total_reward += reward

        if step % 100 == 0:
            print(f"t={step*dt:5.1f}s | Mean: {info['mean_temp']:5.1f}°C | "
                  f"Std: {info['std_temp']:4.2f}°C | Heat: {info['heat_input']:5.1f}W")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break

    print("-" * 60)

    # Summary statistics
    print("\n📊 Performance Summary:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final mean temperature: {mean_temps[-1]:.2f}°C (target: 50°C)")
    print(f"  Final std temperature: {std_temps[-1]:.2f}°C")
    print(f"  Average heat input: {np.mean(heat_inputs):.2f} W")
    print(f"  Max heat input: {np.max(heat_inputs):.2f} W")
    print(f"  Min heat input: {np.min(heat_inputs):.2f} W")

    # Check if goal was reached
    if mean_temps[-1] > 45:
        print("\n✓ Temperature reached near-target range!")
    else:
        print(f"\n✗ Temperature still {50 - mean_temps[-1]:.1f}°C below target")

    # Create simple ASCII plot
    print("\n📈 Temperature Trajectory (ASCII):")
    print_ascii_plot(times, mean_temps, "Mean Temp (°C)", target=50.0)

    print("\n🔥 Heat Input Trajectory (ASCII):")
    print_ascii_plot(times, heat_inputs, "Heat (W)", target=None)

    return {
        'times': times,
        'mean_temps': mean_temps,
        'std_temps': std_temps,
        'heat_inputs': heat_inputs,
        'total_reward': total_reward
    }


def print_ascii_plot(times, values, ylabel, target=None, width=60, height=15):
    """Print simple ASCII plot."""
    min_val = min(values)
    max_val = max(values)

    if target is not None:
        max_val = max(max_val, target + 5)

    val_range = max_val - min_val if max_val > min_val else 1

    # Sample data points
    n_points = min(len(values), width)
    indices = np.linspace(0, len(values) - 1, n_points, dtype=int)
    sampled_values = [values[i] for i in indices]
    sampled_times = [times[i] for i in indices]

    # Create plot matrix
    plot = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot values
    for i, val in enumerate(sampled_values):
        row = int((max_val - val) / val_range * (height - 1))
        row = max(0, min(height - 1, row))
        col = int(i * (width - 1) / (n_points - 1)) if n_points > 1 else 0
        plot[row][col] = '●'

    # Plot target line if provided
    if target is not None and min_val <= target <= max_val:
        target_row = int((max_val - target) / val_range * (height - 1))
        target_row = max(0, min(height - 1, target_row))
        for col in range(width):
            if plot[target_row][col] == ' ':
                plot[target_row][col] = '-'

    # Print plot
    print(f"  {max_val:6.1f} ┤", end='')
    print(''.join(plot[0]))
    for row in range(1, height - 1):
        print(f"         │", end='')
        print(''.join(plot[row]))
    print(f"  {min_val:6.1f} ┤", end='')
    print(''.join(plot[-1]))
    print(f"         └" + "─" * width)
    print(f"         0s{' ' * (width // 2 - 3)}Time{' ' * (width // 2 - 5)}{times[-1]:.0f}s")
    print(f"         {ylabel}" + (f" (target: {target})" if target else ""))


if __name__ == "__main__":
    test_rl_model()
