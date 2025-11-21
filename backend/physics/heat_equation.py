import numpy as np
from dataclasses import dataclass

@dataclass
class RodProperties:
    length: float = 0.2  # m
    diameter: float = 0.01  # m
    density: float = 8960  # kg/m^3 (Copper)
    specific_heat: float = 385  # J/(kg*K) (Copper)
    thermal_conductivity: float = 401  # W/(m*K) (Copper)
    h_convection: float = 15  # W/(m^2*K)
    ambient_temp: float = 25.0  # Celsius

class ThermalRod:
    def __init__(self, props: RodProperties = RodProperties(), n_nodes: int = 50):
        self.props = props
        self.n_nodes = n_nodes
        self.dx = props.length / (n_nodes - 1)
        
        # Geometry
        self.A_cross = np.pi * (props.diameter / 2)**2
        self.P_perimeter = np.pi * props.diameter
        
        # State
        self.temperature = np.ones(n_nodes) * props.ambient_temp
        self.time = 0.0
        
        # Sensor locations (indices)
        # 10 sensors equally spaced. 
        # If L=20cm, sensors at roughly 2, 4, ..., 20cm? Or 0, 2, ..., 18?
        # "Inside the rod... equally spaced". Let's place them at x_i = (i+1) * L / (N+1) or similar.
        # Let's assume they span the length.
        self.sensor_indices = np.linspace(0, n_nodes-1, 10, dtype=int)

    def step(self, dt: float, q_input: float):
        """
        Advance simulation by dt seconds.
        q_input: Heat input at x=0 in Watts.
        """
        T = self.temperature
        T_new = T.copy()
        
        alpha = self.props.thermal_conductivity / (self.props.density * self.props.specific_heat)
        beta = (self.props.h_convection * self.P_perimeter) / (self.props.density * self.props.specific_heat * self.A_cross)
        
        # Explicit FDM Stability check: dt <= dx^2 / (2*alpha)
        # For Cu: alpha ~ 1.1e-4. dx=0.2/49 ~ 0.004. dx^2 ~ 1.6e-5. 
        # Max dt ~ 1.6e-5 / 2.2e-4 ~ 0.07s. 
        # We might need implicit method if we want larger steps, but for real-time viz, small steps are fine.
        # Or we can sub-step.
        
        # Heat Equation: dT/dt = alpha * d2T/dx2 - beta * (T - T_inf)
        
        # Interior nodes (1 to N-2)
        d2T_dx2 = (T[2:] - 2*T[1:-1] + T[:-2]) / self.dx**2
        dT_dt = alpha * d2T_dx2 - beta * (T[1:-1] - self.props.ambient_temp)
        T_new[1:-1] = T[1:-1] + dT_dt * dt
        
        # Boundary Condition at x=0: Constant Heat Flux (Neumann)
        # -k * dT/dx = q_flux = q_input / A
        # Forward difference: (T[1] - T[0])/dx = -q_input / (k * A)
        # T[0] = T[1] + (q_input * dx) / (k * A)
        # However, this is for steady state or strictly enforcing gradient. 
        # For transient, we balance energy at the first node.
        # Node 0 volume = A * dx/2. 
        # m*Cp*dT/dt = Q_in + k*A*(T[1]-T[0])/dx - h*P*(dx/2)*(T[0]-T_inf)
        mass_0 = self.props.density * self.A_cross * (self.dx / 2)
        heat_cond_0 = self.props.thermal_conductivity * self.A_cross * (T[1] - T[0]) / self.dx
        heat_conv_0 = self.props.h_convection * (self.P_perimeter * self.dx / 2) * (T[0] - self.props.ambient_temp)
        
        dT_dt_0 = (q_input + heat_cond_0 - heat_conv_0) / (mass_0 * self.props.specific_heat)
        T_new[0] = T[0] + dT_dt_0 * dt
        
        # Boundary Condition at x=L: Adiabatic (Neumann dT/dx=0) or Convection?
        # Prompt says: "surface except tip... convection". 
        # Usually tip also has convection or is adiabatic. 
        # "tip을 제외한 표면은... 대류" implies tip might NOT have convection (Adiabatic) OR tip has different condition.
        # I'll assume Adiabatic for now as per plan.
        # Node N-1 volume = A * dx/2
        # m*Cp*dT/dt = -k*A*(T[N-1]-T[N-2])/dx - h*P*(dx/2)*(T[N-1]-T_inf)
        mass_N = self.props.density * self.A_cross * (self.dx / 2)
        heat_cond_N = -self.props.thermal_conductivity * self.A_cross * (T[-1] - T[-2]) / self.dx
        heat_conv_N = self.props.h_convection * (self.P_perimeter * self.dx / 2) * (T[-1] - self.props.ambient_temp)
        
        dT_dt_N = (heat_cond_N - heat_conv_N) / (mass_N * self.props.specific_heat)
        T_new[-1] = T[-1] + dT_dt_N * dt
        
        self.temperature = T_new
        self.time += dt

    def get_sensor_readings(self):
        return self.temperature[self.sensor_indices]
