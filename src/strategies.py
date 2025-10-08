import numpy as np

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
        super().__init__(X, T, n_steps)
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lambda_risk = lambda_risk
    
    def generate_schedule(self):
        kappa = np.sqrt(self.lambda_risk * self.sigma**2 / self.eta)
        times = np.linspace(0, self.T, self.n_steps+1)

        if kappa * self.T < 1e-6:  # Linear trajectory as trajectory is small
            x_trajectory = self.X * (1 - times/self.T)
        else:                      # Hyperbolic trajectory
            x_trajectory = self.X * np.sinh(kappa * (self.T - times)) / np.sinh(kappa * self.T)

        schedule = -np.diff(x_trajectory)
        return schedule
    
    def compute_efficient_frontier(self, lambda_values):
        costs = []
        variances = []

        for lam in lambda_values:
            self.lambda_risk = lam
            schedule = self.generate_schedule()

            kappa = np.sqrt(lam * self.sigma**2 / self.eta)

            if kappa * self.T < 1e-6:
                E_cost = (0.5 * self.gamma + self.eta / self.T) * self.X ** 2
            else:
                E_cost = self.gamma * self.X**2 / 2
                E_cost += self.eta * self.X**2 * kappa / (2 * np.tanh(kappa * self.T / 2))
            
            if kappa * self.T < 1e-6:
                Var_cost = (self.sigma**2 * self.X**2 * self.T) / 3
            else:
                sinh_term = np.sinh(kappa * self.T)
                cosh_term = np.cosh(kappa * self.T)
                Var_cost = 0.5 * self.sigma**2 * self.X**2 / (kappa**2 * sinh_term**2)
                Var_cost *= (sinh_term * cosh_term - kappa * self.T)
            
            costs.append(E_cost)
            variances.append(Var_cost)
        
        return np.array(costs), np.array(variances)

if __name__ == "__main__":
    # Trade 100,000 shares over 1 trading day (6.5 hours)
    X = 100000
    T = 1/252  # 1 day in years
    n_steps = 390  # 1-minute intervals
    
    # Market parameters
    sigma = 0.15
    eta = 0.01
    gamma = 0.001
    
    print("=== TWAP ===")
    twap = TWAP(X, T, n_steps)
    twap_schedule = twap.generate_schedule()
    print(f"Total shares: {np.sum(twap_schedule):.0f}")
    print(f"Per-step: {twap_schedule[0]:.0f} shares")
    
    print("\n=== VWAP ===")
    vwap = VWAP(X, T, n_steps)
    vwap_schedule = vwap.generate_schedule()
    print(f"Total shares: {np.sum(vwap_schedule):.0f}")
    print(f"First 10 steps: {vwap_schedule[:10].round(0)}")
    
    print("\n=== Almgren-Chriss ===")
    ac = AlmgrenChriss(X, T, n_steps, sigma, eta, gamma, lambda_risk=1e-6)
    ac_schedule = ac.generate_schedule()
    print(f"Total shares: {np.sum(ac_schedule):.0f}")
    print(f"First 10 steps: {ac_schedule[:10].round(0)}")
    print(f"Last 10 steps: {ac_schedule[-10:].round(0)}")