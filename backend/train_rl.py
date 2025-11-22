#!/usr/bin/env python3
"""
Reinforcement Learning Model Training Script for Thermal Rod Control

This script trains a PPO (Proximal Policy Optimization) agent to control
the thermal rod to achieve uniform temperature distribution.
"""

import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from backend.control.rl_env import ThermalEnv

def train_rl_model(
    total_timesteps=100000,
    n_envs=4,
    model_save_path="ppo_thermal_rod",
    checkpoint_freq=10000,
    log_dir="./logs/"
):
    """
    Train the RL agent for thermal rod control.

    Args:
        total_timesteps: Total number of training timesteps
        n_envs: Number of parallel environments for training
        model_save_path: Path to save the trained model
        checkpoint_freq: Frequency to save checkpoints
        log_dir: Directory for tensorboard logs
    """

    print("=" * 60)
    print("Thermal Rod Control - RL Agent Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Parallel environments: {n_envs}")
    print(f"Model save path: {model_save_path}")
    print("=" * 60)

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment for parallel training
    print("\n[1/4] Creating training environment...")
    env = make_vec_env(ThermalEnv, n_envs=n_envs)

    # Create evaluation environment
    print("[2/4] Creating evaluation environment...")
    eval_env = ThermalEnv()

    # Check if pre-trained model exists
    if os.path.exists(model_save_path + ".zip"):
        print(f"[3/4] Loading existing model from {model_save_path}.zip...")
        model = PPO.load(model_save_path, env=env)
        print("      Continuing training from checkpoint.")
    else:
        print("[3/4] Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="ppo_thermal"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train the model
    print(f"[4/4] Training for {total_timesteps} timesteps...")
    print("      You can monitor progress with: tensorboard --logdir=./logs/")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False
        )

        # Save final model
        print(f"\n✓ Training completed!")
        print(f"  Saving model to {model_save_path}.zip...")
        model.save(model_save_path)
        print(f"✓ Model saved successfully!")

        # Test the trained model
        print("\n" + "=" * 60)
        print("Testing trained model...")
        print("=" * 60)

        obs, _ = eval_env.reset()
        total_reward = 0
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward

            if step % 20 == 0:
                print(f"Step {step}: Mean Temp = {info['mean_temp']:.2f}°C, "
                      f"Variance = {info['variance']:.4f}, Reward = {reward:.2f}")

            if terminated or truncated:
                break

        print(f"\nTotal reward over 100 steps: {total_reward:.2f}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Saving current model to {model_save_path}_interrupted.zip...")
        model.save(model_save_path + "_interrupted")
        print("Model saved.")

    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent for thermal rod control")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps (default: 100000)")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick training mode (20000 timesteps)")

    args = parser.parse_args()

    timesteps = 20000 if args.quick else args.timesteps

    train_rl_model(
        total_timesteps=timesteps,
        n_envs=args.envs
    )
