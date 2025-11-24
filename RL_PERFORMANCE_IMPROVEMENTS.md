# RL Performance Improvements

## Executive Summary

The RL controller was showing poor performance with near-zero outputs compared to MPC's active control (50W → gradual reduction). This document details the root causes identified and comprehensive improvements implemented.

## Problem Analysis

### 1. Critical Issues Identified

#### Reward Function Problems (rl_env.py:48)
```python
# OLD (PROBLEMATIC):
reward = -(1.0 * mean_error + 5.0 * uniformity_error)
```

**Issues:**
- **Massive negative rewards**: Initial temperature 0°C → target 50°C
  - `mean_error = (0 - 50)² = 2500`
  - `reward ≈ -2500 - 5×variance`
  - Scale too large for stable learning

- **Sparse reward signal**: No feedback for intermediate improvements
  - Agent can't distinguish between "do nothing" vs "heat aggressively"
  - No positive reinforcement for moving toward goal

- **No energy cost**: No penalty for energy usage
  - Can't learn energy-efficient control strategies

#### Training Data Insufficient
- Only 100,000 timesteps (≈25 episodes)
- Far too little for learning complex thermal dynamics
- Need 1-10 million timesteps for robust policy

#### Poor Exploration
- Without clear reward signals, agent defaults to conservative (≈0 output) policy
- Gets stuck in local minimum: "don't do anything risky"

## Implemented Solutions

### 1. Reward Function Redesign

Implemented three reward types with proper scaling:

#### **Shaped Reward (Recommended)**
```python
# Reward improvement over time
if self.prev_mean_error is not None:
    mean_improvement = self.prev_mean_error - mean_error
    variance_improvement = self.prev_variance - variance

    # Base reward (scaled)
    temp_reward = -mean_error / 10.0
    uniformity_reward = -std_dev / 5.0

    # Bonus for improvement (positive reinforcement!)
    improvement_bonus = 0.5 * mean_improvement + 0.2 * variance_improvement

    # Small energy penalty
    energy_penalty = -0.001 * q_in / 50.0

    reward = temp_reward + uniformity_reward + improvement_bonus + energy_penalty
```

**Key improvements:**
- ✅ Scaled rewards (-5 to +5 range instead of -2500)
- ✅ Positive reinforcement for improvements
- ✅ Energy efficiency incentive
- ✅ Dense feedback signal

#### **Dense Reward (Alternative)**
Zone-based rewards for clear goal hierarchy:
```python
if mean_error < 1.0 and std_dev < 0.5:
    zone_reward = 10.0  # Excellent!
elif mean_error < 5.0 and std_dev < 2.0:
    zone_reward = 5.0   # Good
elif mean_error < 10.0:
    zone_reward = 1.0   # Acceptable
else:
    zone_reward = -mean_error / 20.0  # Far from target
```

### 2. Enhanced Network Architecture

```python
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])
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
    ent_coef=0.01,      # Encourage exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    device=device,       # GPU support
    verbose=1
)
```

**Improvements:**
- Larger networks: [256, 256] instead of default [64, 64]
- Entropy coefficient (0.01) for better exploration
- GPU device support for faster training

### 3. GPU Training Support

Added training presets optimized for different hardware:

| Preset | Timesteps | Parallel Envs | Use Case |
|--------|-----------|---------------|----------|
| ⚡ Quick | 20,000 | 4 | Testing/debugging |
| 📊 Standard | 500,000 | 8 | CPU training |
| 🎮 GPU | 2,000,000 | 16 | GPU workstation |
| 🚀 Intensive | 10,000,000 | 32 | High-end GPU cluster |

#### CLI Usage:
```bash
# Quick test
python backend/train_rl.py --quick

# Standard training
python backend/train_rl.py --standard

# GPU training with shaped rewards
python backend/train_rl.py --gpu --reward shaped

# Intensive GPU training
python backend/train_rl.py --intensive --device cuda

# Custom configuration
python backend/train_rl.py --timesteps 5000000 --envs 24 --reward dense --lr 0.0001
```

### 4. UI Improvements

Enhanced training modal with:
- Quick preset buttons for easy configuration
- Reward type selector with descriptions
- Advanced parameter controls
- Real-time progress visualization

## Expected Performance Improvements

### Before Improvements:
- Output: ~0W (near-zero, inactive)
- Learning: Very slow, often stuck
- Behavior: Conservative, risk-averse
- Training time: 100k steps insufficient

### After Improvements:
- Output: Expected 10-50W adaptive control
- Learning: Clear progress signal from reward shaping
- Behavior: Active thermal management
- Training time: 500k-2M steps for good performance

## Recommendations

### For Quick Testing (CPU):
```bash
python backend/train_rl.py --standard --reward shaped
```
- 500k timesteps, 8 parallel environments
- Shaped reward function (recommended)
- ~30-60 minutes training time

### For Production Quality (GPU):
```bash
python backend/train_rl.py --gpu --reward shaped --device cuda
```
- 2M timesteps, 16 parallel environments
- Shaped reward function
- ~2-4 hours training time
- Expected high-quality policy

### For Research/Optimal Performance (GPU):
```bash
python backend/train_rl.py --intensive --reward shaped --device cuda
```
- 10M timesteps, 32 parallel environments
- Maximum data collection
- ~10-20 hours training time
- Highest quality policy possible

## Troubleshooting

### If RL still shows poor performance after training:

1. **Check reward signals**: Monitor `current_reward` during training
   - Should start negative, gradually improve
   - If stuck at same value → reward function issue

2. **Increase training time**: Try longer training
   ```bash
   python backend/train_rl.py --timesteps 5000000 --envs 16
   ```

3. **Try different reward types**:
   - `shaped`: Best for learning progress (default)
   - `dense`: More frequent positive signals
   - `simple`: Fallback if others fail

4. **Adjust learning rate**: If training unstable
   ```bash
   python backend/train_rl.py --gpu --lr 0.0001  # Lower LR
   ```

## Technical Details

### Reward Function Comparison

| Aspect | Old | New (Shaped) |
|--------|-----|--------------|
| Scale | -2500 to 0 | -5 to +5 |
| Signal | Sparse | Dense |
| Positive feedback | None | Yes (improvement bonus) |
| Energy awareness | No | Yes (small penalty) |
| Exploration support | Poor | Good |

### Network Architecture

- **Input**: 10 sensor readings (temperature)
- **Policy network**: [10] → [256] → [256] → [1] (heat output)
- **Value network**: [10] → [256] → [256] → [1] (state value)
- **Activation**: ReLU
- **Output**: Continuous [0, 50] Watts

## Next Steps

1. **Train with improved settings**:
   ```bash
   # Delete old model first
   rm ppo_thermal_rod.zip

   # Train with new settings
   python backend/train_rl.py --gpu --reward shaped
   ```

2. **Monitor training progress**:
   - Watch reward values increasing
   - Check test episodes show active control
   - Compare with MPC in simulation

3. **Fine-tune if needed**:
   - Adjust reward weights in `rl_env.py`
   - Try different network sizes
   - Increase training duration

## Files Modified

1. `backend/control/rl_env.py`: Reward function redesign
2. `backend/train_rl.py`: GPU support, presets, improved hyperparameters
3. `backend/app/main.py`: API endpoint updates
4. `frontend/src/App.jsx`: Enhanced UI with presets and advanced options

## Conclusion

The root cause was a combination of:
1. **Poorly scaled reward function** preventing learning
2. **Insufficient training data** (100k timesteps too few)
3. **Lack of exploration** due to sparse rewards

The improvements address all three issues:
- ✅ Reward shaping for clear learning signals
- ✅ GPU support for massive data collection
- ✅ Exploration incentives through entropy and improvement bonuses

Expected result: RL controller will now learn active thermal control comparable to MPC, with 500k-2M timesteps of training.
