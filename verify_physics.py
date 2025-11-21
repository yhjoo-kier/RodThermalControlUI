import numpy as np
import matplotlib.pyplot as plt
from backend.physics.heat_equation import ThermalRod

def test_steady_state():
    rod = ThermalRod(n_nodes=50)
    dt = 0.01
    q_in = 5.0  # Watts
    
    # Run for enough time to reach steady state
    # Time constant tau ~ rho*Cp*V / (h*A_surf) ?
    # Or L^2/alpha ~ 0.04 / 1e-4 ~ 400s. 
    # Let's run for 2000s.
    for _ in range(200000):
        rod.step(dt, q_in)
        
    T = rod.temperature
    x = np.linspace(0, rod.props.length, rod.n_nodes)
    
    print(f"Final Tip Temp: {T[-1]:.2f} C")
    print(f"Final Base Temp: {T[0]:.2f} C")
    
    # Check energy balance
    # Q_in = Q_conv_total
    # Q_conv = integral(h * P * (T(x) - T_inf) dx)
    # Approximate with trapezoidal
    T_excess = T - rod.props.ambient_temp
    integral_T = np.trapz(T_excess, x)
    Q_out = rod.props.h_convection * rod.P_perimeter * integral_T
    
    print(f"Q_in: {q_in:.4f} W")
    print(f"Q_out (Convection): {Q_out:.4f} W")
    
    error = abs(q_in - Q_out) / q_in * 100
    print(f"Energy Balance Error: {error:.2f}%")
    
    if error < 5.0:
        print("PASS: Energy balance holds.")
    else:
        print("FAIL: Energy balance mismatch.")

if __name__ == "__main__":
    test_steady_state()
