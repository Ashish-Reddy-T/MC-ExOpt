import numpy as np
from scipy.optimize import minimize_scalar

class ExecutionStrategy:
    def __init__(self, X, T, n_steps):
        self.X = X
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps

    def generate_schedule(self):
        raise NotImplementedError
    

class TWAP(ExecutionStrategy):
    def generate_schedule(self):
        return np.ones(self.n_steps) * (self.X / self.n_steps)

class VWAP(ExecutionStrategy):
    def __init__(self, X, T, n_steps, volume_profile=None):
        super().__init__(X, T, n_steps)

        if volume_profile is None:
            self.volume_profile = self._default_volume_profile()
        else:
            self.volume_profile = volume_profile / np.sum(volume_profile)
    
    def _default_volume_profile(self):
        x = np.linspace(0, 1, self.n_steps)
        profile = 1.5 - np.abs(2*x - 1)
        return profile / np.sum(profile)
    
    def generate_schedule(self):
        return self.volume_profile * self.X

class AlmgrenChriss(ExecutionStrategy):
    def __init__(self, X, T, n_steps, sigma, eta, gamma, lambda_risk=1e-6):
        super().__init__(self, X, T, n_steps)
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lambda_risk = lambda_risk