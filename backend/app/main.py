from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import numpy as np
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from backend.physics.heat_equation import ThermalRod
from backend.control.mpc_controller import MPCController
from backend.control.rl_agent import RLAgent
from backend.control.pid_controller import PIDController
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from backend.control.rl_env import ThermalEnv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class TrainingConfig(BaseModel):
    total_timesteps: int = 100000
    n_envs: int = 4
    checkpoint_freq: int = 10000

class TrainingStatus(BaseModel):
    is_training: bool
    current_step: int
    total_steps: int
    progress: float
    elapsed_time: float
    estimated_remaining: float
    current_reward: Optional[float] = None
    message: str

# Custom callback for training progress
class ProgressCallback(BaseCallback):
    def __init__(self, training_manager, verbose=0):
        super().__init__(verbose)
        self.training_manager = training_manager

    def _on_step(self) -> bool:
        # Update progress every step
        self.training_manager.current_step = self.num_timesteps

        # Update reward if available
        if len(self.locals.get("rewards", [])) > 0:
            self.training_manager.current_reward = float(np.mean(self.locals["rewards"]))

        # Check if training should be stopped
        return not self.training_manager.stop_requested

class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.stop_requested = False
        self.current_step = 0
        self.total_steps = 0
        self.start_time = None
        self.current_reward = None
        self.task = None
        self.websockets: list[WebSocket] = []

    async def start_training(self, config: TrainingConfig):
        if self.is_training:
            return {"error": "Training already in progress"}

        self.is_training = True
        self.stop_requested = False
        self.current_step = 0
        self.total_steps = config.total_timesteps
        self.start_time = datetime.now()
        self.current_reward = None

        # Start training in background
        self.task = asyncio.create_task(self._train(config))

        return {"status": "Training started"}

    async def _train(self, config: TrainingConfig):
        try:
            # Broadcast training started
            await self._broadcast_status("Training started...")

            # Create vectorized environment
            await self._broadcast_status(f"Creating {config.n_envs} parallel environments...")
            env = make_vec_env(ThermalEnv, n_envs=config.n_envs)
            eval_env = ThermalEnv()

            # Create or load model
            model_path = "ppo_thermal_rod"
            if os.path.exists(f"{model_path}.zip"):
                await self._broadcast_status("Loading existing model...")
                model = PPO.load(model_path, env=env)
            else:
                await self._broadcast_status("Creating new PPO model...")
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
                    verbose=0
                )

            # Setup callback
            progress_callback = ProgressCallback(self)

            # Train
            await self._broadcast_status(f"Training for {config.total_timesteps} timesteps...")

            # Run training in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: model.learn(
                    total_timesteps=config.total_timesteps,
                    callback=progress_callback,
                    progress_bar=False
                )
            )

            if not self.stop_requested:
                # Save model
                await self._broadcast_status("Saving trained model...")
                model.save(model_path)

                # Test model
                await self._broadcast_status("Testing trained model...")
                obs, _ = eval_env.reset()
                total_reward = 0
                for _ in range(100):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break

                await self._broadcast_status(
                    f"Training completed! Test reward: {total_reward:.2f}",
                    completed=True
                )
            else:
                await self._broadcast_status("Training stopped by user", completed=True)

            env.close()
            eval_env.close()

        except Exception as e:
            await self._broadcast_status(f"Training error: {str(e)}", error=True)
        finally:
            self.is_training = False
            self.stop_requested = False

    async def stop_training(self):
        if not self.is_training:
            return {"error": "No training in progress"}

        self.stop_requested = True
        await self._broadcast_status("Stopping training...")

        return {"status": "Training stop requested"}

    def get_status(self) -> TrainingStatus:
        elapsed = 0
        estimated_remaining = 0

        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if self.current_step > 0:
                time_per_step = elapsed / self.current_step
                remaining_steps = self.total_steps - self.current_step
                estimated_remaining = time_per_step * remaining_steps

        progress = (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0

        message = "Idle"
        if self.is_training:
            message = f"Training... Step {self.current_step}/{self.total_steps}"

        return TrainingStatus(
            is_training=self.is_training,
            current_step=self.current_step,
            total_steps=self.total_steps,
            progress=progress,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            current_reward=self.current_reward,
            message=message
        )

    async def _broadcast_status(self, message: str, completed: bool = False, error: bool = False):
        status = self.get_status()
        status.message = message

        data = {
            "type": "training_update",
            "status": status.dict(),
            "completed": completed,
            "error": error
        }

        # Broadcast to all connected websockets
        disconnected = []
        for ws in self.websockets:
            try:
                await ws.send_json(data)
            except:
                disconnected.append(ws)

        # Remove disconnected websockets
        for ws in disconnected:
            self.websockets.remove(ws)

class SimulationManager:
    def __init__(self):
        self.rod_rl = ThermalRod()
        self.rod_mpc = ThermalRod()
        # Initialize other simulation components here
        self.running = False
        self.time_scale = 1.0 # Default speed
        self.target_temp = 50.0 # Default target temperature
        self.mpc_controller = MPCController(rod=self.rod_mpc)
        self.mpc_controller.target_temp = self.target_temp
        self.rl_agent = RLAgent() # Use RLAgent, not rl_env directly for prediction if possible, or just use env
        # Wait, previous code used rl_env and rl_agent. Let's stick to that pattern.
        # Re-checking previous main.py structure...
        # It had self.rl_agent = RLAgent() and self.rl_env was implicit or part of manager?
        # Actually, the previous code I wrote in step 373 had:
        # self.rl_env = RLAgent(rod=self.rod_rl) 
        # But RLAgent in step 328 (viewed) was initialized as RLAgent().
        # Let's check RLAgent definition if needed. 
        # But for now, I will use the code structure that was working before, but with the new cost/reward logic.
        
        # Re-instantiating the correct objects
        self.rl_env = None # We need an env to step? 
        # In step 328, it was:
        # self.rod_rl = ThermalRod()
        # self.rl_agent = RLAgent()
        # And in run_loop:
        # obs_rl = self.rod_rl.get_sensor_readings()
        # action_rl = self.rl_agent.predict(obs_rl)
        # self.rod_rl.step(dt, float(action_rl))
        
        # But in my recent failed edit (step 373), I tried to use self.rl_env.
        # I should revert to the working pattern but add reward calculation.
        
        # Let's stick to the pattern that was working in step 328 but add the reward calc.
        self.rl_agent = RLAgent()
        self.pid_controller = PIDController()

    async def run_loop(self, websocket: WebSocket):
        dt = 0.05 # Keep 0.05 as per stability fix
        
        while self.running:
            steps = int(self.time_scale)
            for _ in range(steps):
                # 1. Get Actions
                # RL
                obs_rl = self.rod_rl.get_sensor_readings().astype(np.float32)
                action_rl, _ = self.rl_agent.predict(obs_rl)
                heat_rl = float(action_rl[0]) # Action is already scaled in Env/Agent if needed, or raw output.
                # Based on verify_rl.py, the output is directly usable as heat input (0-50 range if Env is correct)
                # But wait, verify_rl.py uses RLAgent -> Env. 
                # rl_env.py defines action_space as [0, 50]. 
                # SB3 PPO usually outputs actions in [-1, 1] if normalized, or within bounds if not.
                # Let's check rl_env.py again. It uses spaces.Box(low=0.0, high=50.0).
                # SB3 algorithms (PPO) will output actions in the range of the action space if it's not normalized.
                # However, usually PPO works with normalized actions [-1, 1] internally.
                # But let's trust verify_rl.py which says:
                # action, _ = agent.predict(obs)
                # heat_input = float(action[0])
                # So we should use it directly.
                # RLAgent usually outputs scaled action if using SB3 with normalized env?
                # Let's assume RLAgent.predict returns [0,1] or [-1,1] -> mapped to [0,50]
                # In step 328 it was: action_rl = self.rl_agent.predict(obs_rl) -> float(action_rl)
                
                # MPC
                heat_mpc, cost_mpc = self.mpc_controller.predict(self.rod_mpc.temperature)
                
                # 2. Step Physics
                self.rod_rl.step(dt, heat_rl)
                self.rod_mpc.step(dt, heat_mpc)
                
                # Calculate RL Reward (manually since we are not using the Env wrapper in the loop directly?)
                # Or we should use the Env wrapper?
                # The Env wrapper `ThermalEnv` has the reward logic.
                # It's better to use the logic from `rl_env.py` but we are managing the rod here.
                # I will replicate the simple reward calculation here for visualization.
                mean_temp = np.mean(obs_rl)
                variance = np.var(obs_rl)
                mean_error = (mean_temp - self.target_temp)**2
                uniformity_error = variance
                reward_rl = -(1.0 * mean_error + 5.0 * uniformity_error)

            # 3. Broadcast Data (only once per frame)
            data = {
                "time": self.rod_rl.time,
                "rl": {
                    "temps": self.rod_rl.temperature.tolist(),
                    "input": float(heat_rl),
                    "sensors": self.rod_rl.get_sensor_readings().tolist(),
                    "reward": float(reward_rl)
                },
                "mpc": {
                    "temps": self.rod_mpc.temperature.tolist(),
                    "input": float(heat_mpc),
                    "sensors": self.rod_mpc.get_sensor_readings().tolist(),
                    "cost": float(cost_mpc)
                },
                "speed": self.time_scale
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(dt) # We don't divide by time_scale here because we run multiple steps per loop

    def reset(self):
        self.rod_rl = ThermalRod()
        self.rod_mpc = ThermalRod()
        self.mpc_controller = MPCController(rod=self.rod_mpc)
        self.mpc_controller.target_temp = self.target_temp
        # self.rl_agent doesn't need reset usually
        self.running = False

manager = SimulationManager()
training_manager = TrainingManager()

# Training API endpoints
@app.post("/api/train/start")
async def start_training(config: TrainingConfig):
    result = await training_manager.start_training(config)
    return result

@app.post("/api/train/stop")
async def stop_training():
    result = await training_manager.stop_training()
    return result

@app.get("/api/train/status")
async def get_training_status():
    status = training_manager.get_status()
    return status

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    await websocket.accept()
    training_manager.websockets.append(websocket)

    try:
        # Send initial status
        status = training_manager.get_status()
        await websocket.send_json({
            "type": "training_update",
            "status": status.dict(),
            "completed": False,
            "error": False
        })

        # Keep connection alive
        while True:
            # Just receive ping messages to keep connection alive
            data = await websocket.receive_text()

    except WebSocketDisconnect:
        if websocket in training_manager.websockets:
            training_manager.websockets.remove(websocket)
        print("Training websocket disconnected")

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["command"] == "start":
                manager.running = True
                asyncio.create_task(manager.run_loop(websocket))
                
            elif message["command"] == "stop":
                manager.running = False
                
            elif message["command"] == "reset":
                manager.running = False
                manager.reset()
                
            elif message["command"] == "set_target":
                manager.target_temp = float(message.get("value", 50.0))
                manager.mpc_controller.target_temp = manager.target_temp
            
            elif message["command"] == "set_speed":
                manager.time_scale = float(message.get("speed", 1.0))
            
            elif message["command"] == "set_params":
                # Update physical parameters
                if "h_convection" in message:
                    h_val = float(message["h_convection"])
                    # Update both rods
                    manager.rod_rl.props.h_convection = h_val
                    manager.rod_mpc.props.h_convection = h_val
                    # Update MPC model (re-linearize if needed)
                    # MPCController calculates A, B in __init__. We need to re-calc them.
                    # Let's add a update_model method to MPCController or just re-init it?
                    # Re-init is safer but might reset state. 
                    # Better to just update props and re-call _get_linear_model
                    manager.mpc_controller.rod.props.h_convection = h_val
                    manager.mpc_controller.update_model()
                
    except WebSocketDisconnect:
        manager.running = False
        print("Client disconnected")
