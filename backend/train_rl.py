#!/usr/bin/env python3
"""
Reinforcement Learning Model Training Script for Thermal Rod Control

This script trains a PPO (Proximal Policy Optimization) agent to control
the thermal rod to achieve uniform temperature distribution.

Key improvements:
- Larger network architecture for better representation
- Curriculum learning support
- Better hyperparameters for exploration
- Observation normalization via VecNormalize
"""

import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from backend.control.rl_env import ThermalEnv, ThermalEnvCurriculum
import numpy as np


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning progress tracking."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        return True


def train_rl_model(
    total_timesteps=200000,
    n_envs=8,
    model_save_path="ppo_thermal_rod",
    checkpoint_freq=10000,
    log_dir="./logs/",
    use_curriculum=True
):
    """
    Train the RL agent for thermal rod control with improved settings.

    Args:
        total_timesteps: Total number of training timesteps
        n_envs: Number of parallel environments for training
        model_save_path: Path to save the trained model
        checkpoint_freq: Frequency to save checkpoints
        log_dir: Directory for tensorboard logs
        use_curriculum: Whether to use curriculum learning
    """

    print("=" * 60)
    print("Thermal Rod Control - RL Agent Training (Improved)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Parallel environments: {n_envs}")
    print(f"Model save path: {model_save_path}")
    print(f"Curriculum learning: {use_curriculum}")
    print("=" * 60)

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./checkpoints/", exist_ok=True)

    # Create vectorized environment for parallel training
    print("\n[1/4] Creating training environment...")

    if use_curriculum:
        # Use curriculum learning environment
        env = make_vec_env(
            lambda: ThermalEnv(curriculum_phase=0),  # Start easy
            n_envs=n_envs
        )
    else:
        env = make_vec_env(ThermalEnv, n_envs=n_envs)

    # Wrap with VecNormalize for additional normalization (reward normalization)
    env = VecNormalize(
        env,
        norm_obs=False,  # We already normalize observations manually
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )

    # Create evaluation environment
    print("[2/4] Creating evaluation environment...")
    eval_env = DummyVecEnv([lambda: ThermalEnv(curriculum_phase=2)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=False,
        training=False
    )

    # Network architecture: larger network for better representation
    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128, 64],  # Policy network
            vf=[128, 128, 64]   # Value network
        )
    )

    # Check if pre-trained model exists
    vecnorm_path = model_save_path + "_vecnorm.pkl"
    if os.path.exists(model_save_path + ".zip"):
        print(f"[3/4] Loading existing model from {model_save_path}.zip...")
        # Load VecNormalize stats if available
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
        model = PPO.load(model_save_path, env=env)
        print("      Continuing training from checkpoint.")
    else:
        print("[3/4] Creating new PPO model with improved architecture...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # Higher entropy for more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
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

    curriculum_callback = CurriculumCallback()

    # Curriculum learning: gradually increase difficulty
    if use_curriculum:
        print("\n[*] Curriculum Learning Schedule:")
        print("    Phase 0 (0-30%):   Temperature tracking only")
        print("    Phase 1 (30-60%):  Temperature + light uniformity")
        print("    Phase 2 (60-100%): Full objective")

        phase_steps = [
            int(total_timesteps * 0.3),
            int(total_timesteps * 0.3),
            int(total_timesteps * 0.4)
        ]

        print(f"\n[4/4] Training with curriculum ({total_timesteps} total timesteps)...")

        for phase, steps in enumerate(phase_steps):
            print(f"\n--- Phase {phase}: Training for {steps} steps ---")

            # Update environment phase
            for i in range(n_envs):
                env.envs[i].curriculum_phase = phase

            try:
                model.learn(
                    total_timesteps=steps,
                    callback=[checkpoint_callback, eval_callback, curriculum_callback],
                    progress_bar=False,
                    reset_num_timesteps=False
                )
            except KeyboardInterrupt:
                print("\nTraining interrupted!")
                break

    else:
        # Train without curriculum
        print(f"[4/4] Training for {total_timesteps} timesteps...")
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=False
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted!")

    # Save final model
    print(f"\n✓ Training completed!")
    print(f"  Saving model to {model_save_path}.zip...")
    model.save(model_save_path)
    env.save(vecnorm_path)
    print(f"✓ Model saved successfully!")

    # Test the trained model
    print("\n" + "=" * 60)
    print("Testing trained model...")
    print("=" * 60)

    test_env = ThermalEnv(curriculum_phase=2)
    obs, _ = test_env.reset()
    total_reward = 0
    temps_history = []
    actions_history = []

    print(f"\n{'Step':>5} | {'Mean Temp':>10} | {'Std':>8} | {'Heat In':>8} | {'Reward':>8}")
    print("-" * 55)

    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward

        temps_history.append(info['mean_temp'])
        actions_history.append(info['heat_input'])

        if step % 20 == 0:
            print(f"{step:>5} | {info['mean_temp']:>10.2f} | {info['std_temp']:>8.3f} | "
                  f"{info['heat_input']:>8.2f} | {reward:>8.3f}")

        if terminated or truncated:
            break

    print("-" * 55)
    print(f"\nTotal reward over {step+1} steps: {total_reward:.2f}")
    print(f"Final mean temperature: {temps_history[-1]:.2f}°C (target: 50°C)")
    print(f"Average heat input: {np.mean(actions_history):.2f} W")
    print(f"Max heat input: {np.max(actions_history):.2f} W")
    print("=" * 60)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent for thermal rod control")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps (default: 200000)")
    parser.add_argument("--envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick training mode (50000 timesteps)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")

    args = parser.parse_args()

    timesteps = 50000 if args.quick else args.timesteps

    train_rl_model(
        total_timesteps=timesteps,
        n_envs=args.envs,
        use_curriculum=not args.no_curriculum
    )
