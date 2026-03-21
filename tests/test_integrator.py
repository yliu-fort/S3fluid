import numpy as np
import pytest

from kivy_gpgpu_pde.integrator import RK4Integrator, RK3Integrator

def test_ut09_integrator_linear_decay():
    """
    UT-09: Integrator Switching - Linear Decay System
    Test that the integrator module correctly steps through a known analytical solution.
    We test the equation: dy/dt = -k * y, where y(0) = 1.0, k = 0.5
    Analytical solution: y(t) = exp(-k * t)
    """
    k = 0.5
    def rhs_func(y):
        return -k * y

    dt = 0.01
    num_steps = 100
    t_end = dt * num_steps

    # RK4
    rk4 = RK4Integrator(rhs_func, dt)
    y_rk4 = 1.0
    for _ in range(num_steps):
        y_rk4 = rk4.step(y_rk4)

    # RK3
    rk3 = RK3Integrator(rhs_func, dt)
    y_rk3 = 1.0
    for _ in range(num_steps):
        y_rk3 = rk3.step(y_rk3)

    # Exact
    y_exact = np.exp(-k * t_end)

    error_rk4 = np.abs(y_rk4 - y_exact)
    error_rk3 = np.abs(y_rk3 - y_exact)

    # RK4 should be highly accurate for linear decay
    assert error_rk4 < 1e-6, f"RK4 error {error_rk4} exceeds threshold"
    assert error_rk3 < 1e-6, f"RK3 error {error_rk3} exceeds threshold"
