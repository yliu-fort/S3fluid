import numpy as np

class TimeIntegrator:
    """
    Abstract base class for Time Integrators.
    Works by operating on NumPy coefficients since we are mocking the Time Integrator
    around the GPU pipeline, or we can use pure NumPy for verification.
    """
    def __init__(self, rhs_func, dt):
        self.rhs_func = rhs_func
        self.dt = dt

    def step(self, state):
        raise NotImplementedError

class RK3Integrator(TimeIntegrator):
    """
    Runge-Kutta 3rd Order (TVD).
    """
    def step(self, state):
        # Stage 1
        k1 = self.rhs_func(state)
        u1 = state + self.dt * k1

        # Stage 2
        k2 = self.rhs_func(u1)
        u2 = 0.75 * state + 0.25 * u1 + 0.25 * self.dt * k2

        # Stage 3
        k3 = self.rhs_func(u2)
        state_next = 1.0/3.0 * state + 2.0/3.0 * u2 + 2.0/3.0 * self.dt * k3

        return state_next

class RK4Integrator(TimeIntegrator):
    """
    Runge-Kutta 4th Order.
    """
    def step(self, state):
        k1 = self.rhs_func(state)
        k2 = self.rhs_func(state + 0.5 * self.dt * k1)
        k3 = self.rhs_func(state + 0.5 * self.dt * k2)
        k4 = self.rhs_func(state + self.dt * k3)

        state_next = state + (self.dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return state_next
