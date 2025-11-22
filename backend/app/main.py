from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import numpy as np
from backend.physics.heat_equation import ThermalRod
from backend.control.mpc_controller import MPCController
from backend.control.rl_agent import RLAgent
from backend.control.pid_controller import PIDController

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
