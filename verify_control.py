import numpy as np
import matplotlib.pyplot as plt
from backend.physics.heat_equation import ThermalRod
from backend.control.mpc_controller import MPCController
from backend.control.pid_controller import PIDController

def verify_control():
    # Setup
    rod_mpc = ThermalRod()
    mpc = MPCController(rod_mpc, horizon=10)
    
    rod_pid = ThermalRod()
    pid = PIDController(target_temp=50.0)
    
    steps = 500
    dt = 0.1
    
    history_mpc = {'time': [], 'mean_temp': [], 'std_temp': [], 'input': []}
    history_pid = {'time': [], 'mean_temp': [], 'std_temp': [], 'input': []}
    
    print("Running simulation...")
    for i in range(steps):
        t = i * dt
        
        # MPC Step
        u_mpc = mpc.predict(rod_mpc.temperature)
        rod_mpc.step(dt, float(u_mpc))
        
        history_mpc['time'].append(t)
        history_mpc['mean_temp'].append(np.mean(rod_mpc.temperature))
        history_mpc['std_temp'].append(np.std(rod_mpc.temperature))
        history_mpc['input'].append(u_mpc)
        
        # PID Step
        u_pid = pid.predict(rod_pid.temperature)
        rod_pid.step(dt, float(u_pid))
        
        history_pid['time'].append(t)
        history_pid['mean_temp'].append(np.mean(rod_pid.temperature))
        history_pid['std_temp'].append(np.std(rod_pid.temperature))
        history_pid['input'].append(u_pid)
        
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Mean Temp
    axs[0].plot(history_mpc['time'], history_mpc['mean_temp'], label='MPC')
    axs[0].plot(history_pid['time'], history_pid['mean_temp'], label='PID')
    axs[0].axhline(y=50.0, color='r', linestyle='--', label='Target')
    axs[0].set_ylabel('Mean Temperature (C)')
    axs[0].legend()
    axs[0].set_title('Control Performance Comparison')
    
    # Std Dev (Uniformity)
    axs[1].plot(history_mpc['time'], history_mpc['std_temp'], label='MPC')
    axs[1].plot(history_pid['time'], history_pid['std_temp'], label='PID')
    axs[1].set_ylabel('Temp Std Dev (C)')
    axs[1].legend()
    
    # Input
    axs[2].plot(history_mpc['time'], history_mpc['input'], label='MPC')
    axs[2].plot(history_pid['time'], history_pid['input'], label='PID')
    axs[2].set_ylabel('Heat Input (W)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('control_verification.png')
    print("Verification plot saved to control_verification.png")

if __name__ == "__main__":
    verify_control()
