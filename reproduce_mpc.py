import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from backend.physics.heat_equation import ThermalRod, RodProperties

class CustomMPC:
    def __init__(self, rod: ThermalRod, horizon: int = 10, dt: float = 0.05, 
                 weight_mean: float = 10.0, weight_uniformity: float = 1.0, weight_input: float = 0.01):
        self.rod = rod
        self.horizon = horizon
        self.dt = dt
        self.target_temp = 50.0
        self.weight_mean = weight_mean
        self.weight_uniformity = weight_uniformity
        self.weight_input = weight_input
        
        self.A, self.B, self.d = self._get_linear_model()
        
    def _get_linear_model(self):
        # Copy of the logic from MPCController
        n = self.rod.n_nodes
        alpha = self.rod.props.thermal_conductivity / (self.rod.props.density * self.rod.props.specific_heat)
        beta = (self.rod.props.h_convection * self.rod.P_perimeter) / (self.rod.props.density * self.rod.props.specific_heat * self.rod.A_cross)
        dx = self.rod.dx
        
        A_c = np.zeros((n, n))
        B_c = np.zeros((n, 1))
        d_c = np.zeros(n)
        
        coeff = alpha / dx**2
        
        for i in range(1, n-1):
            A_c[i, i-1] = coeff
            A_c[i, i] = -2*coeff - beta
            A_c[i, i+1] = coeff
            d_c[i] = beta * self.rod.props.ambient_temp
            
        mass_0 = self.rod.props.density * self.rod.A_cross * (dx / 2)
        denom = mass_0 * self.rod.props.specific_heat
        A_c[0, 0] = (-self.rod.props.thermal_conductivity * self.rod.A_cross / dx - self.rod.props.h_convection * self.rod.P_perimeter * dx / 2) / denom
        A_c[0, 1] = (self.rod.props.thermal_conductivity * self.rod.A_cross / dx) / denom
        B_c[0, 0] = 1.0 / denom
        d_c[0] = (self.rod.props.h_convection * self.rod.P_perimeter * dx / 2 * self.rod.props.ambient_temp) / denom
        
        mass_N = self.rod.props.density * self.rod.A_cross * (dx / 2)
        denom_N = mass_N * self.rod.props.specific_heat
        A_c[-1, -1] = (-self.rod.props.thermal_conductivity * self.rod.A_cross / dx - self.rod.props.h_convection * self.rod.P_perimeter * dx / 2) / denom_N
        A_c[-1, -2] = (self.rod.props.thermal_conductivity * self.rod.A_cross / dx) / denom_N
        d_c[-1] = (self.rod.props.h_convection * self.rod.P_perimeter * dx / 2 * self.rod.props.ambient_temp) / denom_N
        
        A = np.eye(n) + A_c * self.dt
        B = B_c * self.dt
        d = d_c * self.dt
        
        return A, B, d

    def predict(self, current_temp: np.ndarray):
        x = cp.Variable((self.rod.n_nodes, self.horizon + 1))
        u = cp.Variable((1, self.horizon))
        
        cost = 0
        constraints = []
        
        constraints.append(x[:, 0] == current_temp)
        
        for t in range(self.horizon):
            constraints.append(x[:, t+1] == self.A @ x[:, t] + self.B @ u[:, t] + self.d)
            
            # Cost 1: Mean Temp
            mean_temp = cp.sum(x[:, t+1]) / self.rod.n_nodes
            cost += cp.square(mean_temp - self.target_temp) * self.weight_mean
            
            # Cost 2: Uniformity (Sum of squares from target)
            # Note: Original code used sum_squares(x - target). 
            # This penalizes deviation from target for ALL nodes.
            # If we want PURE uniformity (variance), we should minimize sum_squares(x - mean_temp),
            # but that is convex? Yes, sum((x_i - x_bar)^2).
            # However, let's stick to the "Sum of squares from target" as "Uniformity+Accuracy" metric
            # or try to implement pure variance if possible.
            # Pure variance: x.T @ (I - 11^T/n) @ x. This is quadratic form.
            # Let's stick to the user's question: "squared effect like variance".
            # The current code does: sum((T_i - T_target)^2).
            cost += cp.sum_squares(x[:, t+1] - self.target_temp) * self.weight_uniformity
            
            cost += cp.square(u[:, t]) * self.weight_input
            
            constraints.append(u[:, t] >= 0)
            constraints.append(u[:, t] <= 50.0)
            
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True)
        except cp.SolverError:
            return 0.0
            
        if u.value is None:
            return 0.0
        return u.value[0, 0]

def run_sim():
    # High h to exaggerate the gradient
    props = RodProperties(h_convection=50.0) 
    
    # Case 1: Default Weights (Mean=10, Uniformity=1)
    rod1 = ThermalRod(props)
    mpc1 = CustomMPC(rod1, weight_mean=10.0, weight_uniformity=1.0)
    
    # Case 2: High Uniformity Weight (Mean=10, Uniformity=100)
    rod2 = ThermalRod(props)
    mpc2 = CustomMPC(rod2, weight_mean=10.0, weight_uniformity=100.0)
    
    steps = 400
    dt = 0.05
    
    hist1 = {'mean': [], 'std': [], 'input': []}
    hist2 = {'mean': [], 'std': [], 'input': []}
    
    print("Running simulations...")
    for i in range(steps):
        # Sim 1
        u1 = mpc1.predict(rod1.temperature)
        rod1.step(dt, float(u1))
        hist1['mean'].append(np.mean(rod1.temperature))
        hist1['std'].append(np.std(rod1.temperature))
        hist1['input'].append(u1)
        
        # Sim 2
        u2 = mpc2.predict(rod2.temperature)
        rod2.step(dt, float(u2))
        hist2['mean'].append(np.mean(rod2.temperature))
        hist2['std'].append(np.std(rod2.temperature))
        hist2['input'].append(u2)
        
    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    t = np.arange(steps) * dt
    
    axs[0].plot(t, hist1['mean'], label='Default (W_unif=1)')
    axs[0].plot(t, hist2['mean'], label='High Uniformity (W_unif=100)')
    axs[0].axhline(50, color='k', linestyle='--')
    axs[0].set_title('Mean Temperature')
    axs[0].legend()
    
    axs[1].plot(t, hist1['std'], label='Default (W_unif=1)')
    axs[1].plot(t, hist2['std'], label='High Uniformity (W_unif=100)')
    axs[1].set_title('Temperature Std Dev (Gradient)')
    axs[1].legend()
    
    axs[2].plot(t, hist1['input'], label='Default (W_unif=1)')
    axs[2].plot(t, hist2['input'], label='High Uniformity (W_unif=100)')
    axs[2].set_title('Input Power')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('mpc_comparison.png')
    print("Done.")

if __name__ == "__main__":
    run_sim()
