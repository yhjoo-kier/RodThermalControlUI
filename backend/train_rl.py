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
    log_dir="./logs/",
    reward_type="shaped",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    device="auto"
):
    """
    Train the RL agent for thermal rod control.

    Args:
        total_timesteps: Total number of training timesteps
        n_envs: Number of parallel environments for training
        model_save_path: Path to save the trained model
        checkpoint_freq: Frequency to save checkpoints
        log_dir: Directory for tensorboard logs
        reward_type: Reward function type ("simple", "shaped", "dense")
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per environment per update
        batch_size: Batch size for training
        device: Device to use ("auto", "cuda", "cpu")
    """

    print("=" * 60)
    print("Thermal Rod Control - RL Agent Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Parallel environments: {n_envs}")
    print(f"Reward type: {reward_type}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Model save path: {model_save_path}")
    print("=" * 60)

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment for parallel training
    print("\n[1/4] Creating training environment...")
    env = make_vec_env(lambda: ThermalEnv(reward_type=reward_type), n_envs=n_envs)

    # Create evaluation environment
    print("[2/4] Creating evaluation environment...")
    eval_env = ThermalEnv(reward_type=reward_type)

    # Check if pre-trained model exists
    if os.path.exists(model_save_path + ".zip"):
        print(f"[3/4] Loading existing model from {model_save_path}.zip...")
        model = PPO.load(model_save_path, env=env, device=device)
        print("      Continuing training from checkpoint.")
    else:
        print("[3/4] Creating new PPO model...")
        # Configure network architecture
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger networks for complex control
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1
        )
        print(f"      Network architecture: {policy_kwargs}")
        print(f"      Device: {model.device}")

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

    parser = argparse.ArgumentParser(
        description="Train RL agent for thermal rod control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training scale presets
    parser.add_argument("--quick", action="store_true",
                        help="Quick training mode (20k timesteps, 4 envs)")
    parser.add_argument("--standard", action="store_true",
                        help="Standard training mode (500k timesteps, 8 envs)")
    parser.add_argument("--gpu", action="store_true",
                        help="GPU training mode (2M timesteps, 16 envs)")
    parser.add_argument("--intensive", action="store_true",
                        help="Intensive GPU training mode (10M timesteps, 32 envs)")

    # Fine-grained control
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (overrides presets)")
    parser.add_argument("--envs", type=int, default=None,
                        help="Number of parallel environments (overrides presets)")

    # Training configuration
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["simple", "shaped", "dense"],
                        help="Reward function type")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use for training")

    # Model management
    parser.add_argument("--model-path", type=str, default="ppo_thermal_rod",
                        help="Path to save/load the model")
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                        help="Frequency to save checkpoints")

    args = parser.parse_args()

    # Determine training configuration from presets
    if args.intensive:
        timesteps = 10_000_000
        n_envs = 32
        print("🚀 INTENSIVE GPU MODE: 10M timesteps, 32 parallel environments")
    elif args.gpu:
        timesteps = 2_000_000
        n_envs = 16
        print("🎮 GPU MODE: 2M timesteps, 16 parallel environments")
    elif args.standard:
        timesteps = 500_000
        n_envs = 8
        print("📊 STANDARD MODE: 500k timesteps, 8 parallel environments")
    elif args.quick:
        timesteps = 20_000
        n_envs = 4
        print("⚡ QUICK MODE: 20k timesteps, 4 parallel environments")
    else:
        timesteps = 100_000
        n_envs = 4
        print("📝 DEFAULT MODE: 100k timesteps, 4 parallel environments")

    # Override with explicit arguments if provided
    if args.timesteps is not None:
        timesteps = args.timesteps
    if args.envs is not None:
        n_envs = args.envs

    print(f"Reward function: {args.reward}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print()

    train_rl_model(
        total_timesteps=timesteps,
        n_envs=n_envs,
        model_save_path=args.model_path,
        checkpoint_freq=args.checkpoint_freq,
        reward_type=args.reward,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device
    )
