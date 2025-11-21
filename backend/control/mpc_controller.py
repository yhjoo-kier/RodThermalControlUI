import numpy as np
import cvxpy as cp
from backend.physics.heat_equation import ThermalRod, RodProperties

class MPCController:
    def __init__(self, rod: ThermalRod, horizon: int = 10, dt: float = 0.05):
        self.rod = rod
        self.horizon = horizon
        self.dt = dt
        self.target_temp = 50.0  # Target mean temperature
        
        # Linearize model: x_{k+1} = A x_k + B u_k + d
        # Since our model is linear (conduction + convection), we can extract A and B directly.
        # However, constructing A and B from the FDM matrix is cleaner.
        self.A, self.B, self.d = self._get_linear_model()
        
    def update_model(self):
        """Re-calculate system matrices based on current rod properties."""
        self.A, self.B, self.d = self._get_linear_model()
        
    def _get_linear_model(self):
        # Extract A, B matrices for x_{k+1} = A x_k + B u_k + d (d is affine term from T_inf)
        # We can simulate step responses or build matrix directly.
        # Building directly is better.
        
        n = self.rod.n_nodes
        alpha = self.rod.props.thermal_conductivity / (self.rod.props.density * self.rod.props.specific_heat)
        beta = (self.rod.props.h_convection * self.rod.P_perimeter) / (self.rod.props.density * self.rod.props.specific_heat * self.rod.A_cross)
        dx = self.rod.dx
        
        # Continuous time: dx/dt = A_c x + B_c u + d_c
        A_c = np.zeros((n, n))
        B_c = np.zeros((n, 1))
        d_c = np.zeros(n)
        
        coeff = alpha / dx**2
        
        # Interior nodes
        for i in range(1, n-1):
            A_c[i, i-1] = coeff
            A_c[i, i] = -2*coeff - beta
            A_c[i, i+1] = coeff
            d_c[i] = beta * self.rod.props.ambient_temp
            
        # Boundary 0: T[0] dynamics
        # dT/dt = (q_in + k*A*(T[1]-T[0])/dx - h*P*dx/2*(T[0]-T_inf)) / (m*Cp)
        mass_0 = self.rod.props.density * self.rod.A_cross * (dx / 2)
        denom = mass_0 * self.rod.props.specific_heat
        
        A_c[0, 0] = (-self.rod.props.thermal_conductivity * self.rod.A_cross / dx - self.rod.props.h_convection * self.rod.P_perimeter * dx / 2) / denom
        A_c[0, 1] = (self.rod.props.thermal_conductivity * self.rod.A_cross / dx) / denom
        B_c[0, 0] = 1.0 / denom
        d_c[0] = (self.rod.props.h_convection * self.rod.P_perimeter * dx / 2 * self.rod.props.ambient_temp) / denom
        
        # Boundary N-1: T[N-1] dynamics (Adiabatic tip)
        mass_N = self.rod.props.density * self.rod.A_cross * (dx / 2)
        denom_N = mass_N * self.rod.props.specific_heat
        
        A_c[-1, -1] = (-self.rod.props.thermal_conductivity * self.rod.A_cross / dx - self.rod.props.h_convection * self.rod.P_perimeter * dx / 2) / denom_N
        A_c[-1, -2] = (self.rod.props.thermal_conductivity * self.rod.A_cross / dx) / denom_N
        d_c[-1] = (self.rod.props.h_convection * self.rod.P_perimeter * dx / 2 * self.rod.props.ambient_temp) / denom_N
        
        # Discretize: x_{k+1} = (I + A_c * dt) x_k + (B_c * dt) u_k + (d_c * dt)
        # Euler forward
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
            # Dynamics
            constraints.append(x[:, t+1] == self.A @ x[:, t] + self.B @ u[:, t] + self.d)
            
            # Cost
            # 1. Maintain mean temp at target
            mean_temp = cp.sum(x[:, t+1]) / self.rod.n_nodes
            cost += cp.square(mean_temp - self.target_temp) * 10.0
            
            # 2. Minimize variance (uniformity) -> Minimize sum((T_i - T_mean)^2)
            # This is hard in CVXPY directly as quadratic over variables?
            # Instead, minimize sum((T_i - T_target)^2) which implies mean=target and variance=0
            cost += cp.sum_squares(x[:, t+1] - self.target_temp)
            
            # 3. Control effort regularization
            cost += cp.square(u[:, t]) * 0.01
            
            # Constraints
            constraints.append(u[:, t] >= 0)
            constraints.append(u[:, t] <= 50.0) # Max 50W
            
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True)
        except cp.SolverError:
            print("MPC Solver Error")
            return 0.0, 0.0

        
        if u.value is None:
            return 0.0, 0.0
            
        return u.value[0, 0], problem.value

