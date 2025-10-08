import numpy as np

class PriceSimulator:
    def __init__(self, S0, mu, sigma, eta, gamma, dt):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.dt = dt
    
    def simulate_path(self, T, seed=None):
        if seed is not None:
            pass