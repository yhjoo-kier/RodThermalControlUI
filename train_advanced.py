#!/usr/bin/env python3
"""
Combined training script for SAC and Imitation Learning approaches.
"""

import os
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from backend.control.rl_env import ThermalEnv
from backend.control.mpc_controller import MPCController
from backend.physics.heat_equation import ThermalRod
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# PART 1: SAC Training
# ============================================================================

def train_sac(total_timesteps=300000, save_path="sac_thermal_rod"):
    """Train SAC agent - better for continuous control."""
    print("=" * 60)
    print("Training SAC Agent")
    print("=" * 60)

    # Create environments
    env = make_vec_env(lambda: ThermalEnv(curriculum_phase=2), n_envs=4)
    eval_env = DummyVecEnv([lambda: ThermalEnv(curriculum_phase=2)])

    # SAC with tuned hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",  # Automatic entropy tuning
        policy_kwargs=dict(
            net_arch=[256, 256]  # Larger network
        ),
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_sac/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(save_path)
    print(f"SAC model saved to {save_path}.zip")

    env.close()
    eval_env.close()
    return model


# ============================================================================
# PART 2: Collect MPC Demonstrations
# ============================================================================

def collect_mpc_demonstrations(n_episodes=50, max_steps=1000):
    """Collect expert demonstrations from MPC controller."""
    print("=" * 60)
    print("Collecting MPC Demonstrations")
    print("=" * 60)

    observations = []
    actions = []

    for ep in range(n_episodes):
        rod = ThermalRod()
        mpc = MPCController(rod)
        env = ThermalEnv(curriculum_phase=2)
        obs, _ = env.reset()

        episode_obs = []
        episode_actions = []

        for step in range(max_steps):
            # Get MPC action
            current_temp = rod.temperature
            mpc_action, _ = mpc.predict(current_temp)
            mpc_action = np.clip(mpc_action, 0, 50)

            episode_obs.append(obs.copy())
            episode_actions.append([mpc_action])

            # Step environment
            obs, reward, terminated, truncated, info = env.step(np.array([mpc_action]))

            # Sync rod state with environment
            rod.temperature = env.rod.temperature.copy()

            if terminated or truncated:
                break

        observations.extend(episode_obs)
        actions.extend(episode_actions)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} - Mean temp: {info['mean_temp']:.1f}°C")

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    print(f"\nCollected {len(observations)} demonstration samples")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Action range: [{actions.min():.2f}, {actions.max():.2f}]")

    return observations, actions


# ============================================================================
# PART 3: Behavior Cloning
# ============================================================================

class BCPolicy(nn.Module):
    """Simple MLP policy for Behavior Cloning."""
    def __init__(self, obs_dim=12, action_dim=1, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output in [0, 1], scale to [0, 50]
        )

    def forward(self, x):
        return self.net(x) * 50.0  # Scale to action range [0, 50]


def train_behavior_cloning(observations, actions, epochs=100, batch_size=256, save_path="bc_thermal_rod.pt"):
    """Train policy via behavior cloning."""
    print("=" * 60)
    print("Training Behavior Cloning Policy")
    print("=" * 60)

    # Convert to tensors
    obs_tensor = torch.FloatTensor(observations)
    act_tensor = torch.FloatTensor(actions)

    # Create model
    policy = BCPolicy(obs_dim=observations.shape[1], action_dim=actions.shape[1])
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    n_samples = len(observations)
    best_loss = float('inf')

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            obs_batch = obs_tensor[batch_idx]
            act_batch = act_tensor[batch_idx]

            # Forward pass
            pred_actions = policy(obs_batch)
            loss = criterion(pred_actions, act_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), save_path)

    print(f"\nBC policy saved to {save_path}")
    print(f"Best loss: {best_loss:.4f}")

    return policy


# ============================================================================
# PART 4: Evaluation
# ============================================================================

def evaluate_model(model, model_type="ppo", n_episodes=5, max_steps=1000):
    """Evaluate a trained model."""
    env = ThermalEnv(curriculum_phase=2)

    all_mean_temps = []
    all_actions = []
    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if model_type == "bc":
            # Reset BC agent step counter if needed
            pass

        episode_temps = []
        episode_actions = []
        episode_reward = 0

        for step in range(max_steps):
            if model_type == "bc":
                with torch.no_grad():
                    action = model(torch.FloatTensor(obs).unsqueeze(0)).numpy()[0]
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_temps.append(info['mean_temp'])
            episode_actions.append(info['heat_input'])
            episode_reward += reward

            if terminated or truncated:
                break

        all_mean_temps.append(episode_temps[-1])
        all_actions.append(np.mean(episode_actions))
        all_rewards.append(episode_reward)

    return {
        'final_temp_mean': np.mean(all_mean_temps),
        'final_temp_std': np.std(all_mean_temps),
        'avg_action': np.mean(all_actions),
        'avg_reward': np.mean(all_rewards)
    }


def run_full_evaluation():
    """Run evaluation on all trained models."""
    print("\n" + "=" * 60)
    print("FULL EVALUATION")
    print("=" * 60)

    results = {}

    # 1. PPO (existing)
    if os.path.exists("ppo_thermal_rod.zip"):
        print("\n[1] Evaluating PPO...")
        from stable_baselines3 import PPO
        ppo_model = PPO.load("ppo_thermal_rod")
        results['PPO'] = evaluate_model(ppo_model, "ppo")

    # 2. SAC
    if os.path.exists("sac_thermal_rod.zip"):
        print("\n[2] Evaluating SAC...")
        sac_model = SAC.load("sac_thermal_rod")
        results['SAC'] = evaluate_model(sac_model, "sac")

    # 3. Behavior Cloning
    if os.path.exists("bc_thermal_rod.pt"):
        print("\n[3] Evaluating Behavior Cloning...")
        bc_policy = BCPolicy()
        bc_policy.load_state_dict(torch.load("bc_thermal_rod.pt"))
        bc_policy.eval()
        results['BC'] = evaluate_model(bc_policy, "bc")

    # 4. MPC (baseline)
    print("\n[4] Evaluating MPC (baseline)...")
    results['MPC'] = evaluate_mpc()

    return results


def evaluate_mpc(n_episodes=5, max_steps=1000):
    """Evaluate MPC controller as baseline."""
    all_mean_temps = []
    all_actions = []
    all_rewards = []

    for ep in range(n_episodes):
        rod = ThermalRod()
        mpc = MPCController(rod)
        env = ThermalEnv(curriculum_phase=2)
        obs, _ = env.reset()

        episode_temps = []
        episode_actions = []
        episode_reward = 0

        for step in range(max_steps):
            current_temp = rod.temperature
            mpc_action, _ = mpc.predict(current_temp)
            mpc_action = np.clip(mpc_action, 0, 50)

            obs, reward, terminated, truncated, info = env.step(np.array([mpc_action]))
            rod.temperature = env.rod.temperature.copy()

            episode_temps.append(info['mean_temp'])
            episode_actions.append(info['heat_input'])
            episode_reward += reward

            if terminated or truncated:
                break

        all_mean_temps.append(episode_temps[-1])
        all_actions.append(np.mean(episode_actions))
        all_rewards.append(episode_reward)

    return {
        'final_temp_mean': np.mean(all_mean_temps),
        'final_temp_std': np.std(all_mean_temps),
        'avg_action': np.mean(all_actions),
        'avg_reward': np.mean(all_rewards)
    }


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<15} {'Final Temp (°C)':<20} {'Avg Heat (W)':<15} {'Reward':<15}")
    print("-" * 70)

    for method, metrics in results.items():
        temp_str = f"{metrics['final_temp_mean']:.1f} ± {metrics['final_temp_std']:.1f}"
        print(f"{method:<15} {temp_str:<20} {metrics['avg_action']:<15.1f} {metrics['avg_reward']:<15.1f}")

    print("=" * 70)
    print("Target: 50.0°C")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sac-only", action="store_true", help="Train SAC only")
    parser.add_argument("--bc-only", action="store_true", help="Train BC only")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate only")
    parser.add_argument("--timesteps", type=int, default=300000, help="SAC training timesteps")
    args = parser.parse_args()

    if args.eval_only:
        results = run_full_evaluation()
        print_comparison_table(results)
    elif args.sac_only:
        train_sac(total_timesteps=args.timesteps)
    elif args.bc_only:
        obs, acts = collect_mpc_demonstrations(n_episodes=50)
        train_behavior_cloning(obs, acts)
    else:
        # Full pipeline
        print("\n" + "=" * 70)
        print("FULL TRAINING PIPELINE")
        print("=" * 70)

        # 1. Train SAC
        print("\n[Step 1/3] Training SAC...")
        train_sac(total_timesteps=args.timesteps)

        # 2. Collect MPC demonstrations and train BC
        print("\n[Step 2/3] Collecting MPC demonstrations...")
        obs, acts = collect_mpc_demonstrations(n_episodes=50)

        print("\n[Step 3/3] Training Behavior Cloning...")
        train_behavior_cloning(obs, acts)

        # 4. Evaluate all
        results = run_full_evaluation()
        print_comparison_table(results)
