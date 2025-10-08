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
            np.random.seed(seed)
        
        n_steps = int(T / self.dt)
        times = np.linspace(0, T, n_steps+1)

        dW = np.random.normal(0, np.sqrt(self.dt), n_steps)
        prices = np.zeros(n_steps+1)
        prices[0] = self.S0

        for i in range(n_steps):
            drift = self.mu * prices[i] * self.dt
            diffusion = self.sigma * prices[i] * dW[i]
            prices[i+1] = prices[i] + drift + diffusion
        
        return times, prices
    
    def execute_schedule(self, schedule, T, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        n_steps = len(schedule)
        times = np.linspace(0, T, n_steps+1)
        dt_schedule = T / n_steps

        dW = np.random.normal(0, np.sqrt(dt_schedule), n_steps)
        unaffected_prices = np.zeros(n_steps+1)
        unaffected_prices[0] = self.S0

        execution_prices = np.zeros(n_steps)
        cumulative_traded = 0
        total_cost = 0

        for i in range(n_steps):
            drift = self.mu * unaffected_prices[i] * dt_schedule
            diffusion = self.sigma * unaffected_prices[i] * dW[i]
            unaffected_prices[i+1] = unaffected_prices[i] + drift + diffusion
            
            # Trade rate: shares per unit time
            v_t = schedule[i] / dt_schedule     

            temporary_impact = self.eta * v_t
            permanent_impact = self.gamma * cumulative_traded

            execution_prices[i] = unaffected_prices[i] + temporary_impact + permanent_impact

            cumulative_traded += schedule[i]

            # Cost = (execution_price - arrival_price) * quantity
            total_cost += (execution_prices[i] - self.S0) * schedule[i]

        return {
            'total_cost': total_cost,
            'execution_prices': execution_prices,
            'unaffected_prices': unaffected_prices,
            'times': times
        }
    
    def monte_carlo_eval(self, schedule, T, n_sims=1000):
        costs = np.zeros(n_sims)

        for i in range(n_sims):
            result = self.execute_schedule(schedule, T, seed=i)
            costs[i] = result['total_cost']
        
        return {
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'var_cost': np.var(costs),
            'costs': costs
        }

if __name__ == "__main__":
    S0 = 450
    mu = 0.0
    sigma = 0.15
    eta = 0.01
    gamma = 0.001
    dt = 1/(252*390)

    sim = PriceSimulator(S0=S0, mu=mu, sigma=sigma, eta=eta, gamma=gamma, dt=dt)

    T = 1/252 
    times, prices = sim.simulate_path(T, seed=42)

    print(f"Simulated {len(prices)} prices!")
    print(f"Start: ${prices[0]:.2f}, End: ${prices[-1]:.2f}")
    print(f"Return: {(prices[-1]/prices[0] - 1)*100:.2f}%")