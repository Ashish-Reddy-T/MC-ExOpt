import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class MarketConfig:
    S0: float = 100.0       # Initial Price
    sigma: float = 0.01     # Daily Volatility
    dt: float = 1/390       # Time step (1 minute in a 6.5h trading day)
    T: float = 1.0          # Total time (1 day)
    eta: float = 0.01       # Temporary Impact Coefficient
    theta: float = 0.001    # Permanent Impact Coefficient
    risk_free_rate: float = 0.0

class MarketSimulator:
    def __init__(self, config: MarketConfig):
        self.config = config
        self.steps = int(config.T / config.dt)
        
    def simulate_price_path(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generates a Geometric Brownian Motion price path (unaffected price).
        """
        if seed is not None:
            np.random.seed(seed)
            
        S0 = self.config.S0
        dt = self.config.dt
        sigma = self.config.sigma
        mu = self.config.risk_free_rate
        
        # GBM Formula: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random shocks
        shocks = np.random.normal(0, 1, self.steps)
        
        # Calculate log returns
        log_returns = drift + diffusion * shocks
        
        # Calculate price path
        price_path = np.zeros(self.steps + 1)
        price_path[0] = S0
        price_path[1:] = S0 * np.exp(np.cumsum(log_returns))
        
        return price_path

    def calculate_execution_cost(self, 
                               trade_schedule: np.ndarray, 
                               price_path: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Calculates the implementation shortfall and average execution price 
        given a trade schedule and an unaffected price path.
        
        Args:
            trade_schedule: Array of shares sold at each step. Sum should equal total inventory.
            price_path: Array of unaffected prices (length = steps + 1).
            
        Returns:
            total_cost: Implementation Shortfall
            avg_price: VWAP of execution
            exec_prices: Array of execution prices at each step
        """
        if len(trade_schedule) != self.steps:
            # If schedule is shorter/longer, we might need to adjust logic, 
            # but for now assume 1 trade per step.
            # Note: price_path has steps+1 (t=0 to t=T). Trades happen from t=0 to t=T-1 or t=1 to t=T?
            # Usually we trade over the interval. Let's assume trade i happens at price i (or average of i and i+1).
            # For simplicity, let's say trade i happens at price_path[i] impacted.
            pass

        # Ensure trade schedule matches simulation steps
        n_trades = len(trade_schedule)
        # We use prices from index 0 to n_trades-1 for execution
        # (Assuming we trade at the start of each interval or throughout)
        
        S = price_path[:n_trades]
        
        # Permanent Impact: Price is depressed by total amount sold so far
        # S_t_perm = S_t - theta * sum(n_0...n_{t-1})
        # Actually, standard AC model often treats permanent impact as linear drift 
        # or impact on the *next* price.
        # Let's use the formula: S_t_impacted = S_t - theta * cumulative_sold
        
        cumulative_sold = np.cumsum(trade_schedule) - trade_schedule # Sold *before* this trade
        # Or should permanent impact affect the current trade? 
        # Usually permanent impact affects *subsequent* trades. 
        
        # Temporary Impact: eta * (rate of trading)
        # rate = trade_amount / dt
        # cost = eta * rate
        # execution_price = S_t_impacted - temporary_impact
        
        # Let's apply impacts
        perm_impact = self.config.theta * cumulative_sold
        temp_impact = self.config.eta * (trade_schedule / self.config.dt)
        
        # Executed Price = Unaffected Price - Permanent Impact - Temporary Impact
        # (Minus because we are selling. If buying, it would be +)
        exec_prices = S - perm_impact - temp_impact
        
        # Cash received
        cash_received = np.sum(trade_schedule * exec_prices)
        
        # Initial Value (Arrival Price)
        total_shares = np.sum(trade_schedule)
        initial_value = total_shares * price_path[0]
        
        # Implementation Shortfall
        is_cost = initial_value - cash_received
        
        avg_exec_price = cash_received / total_shares if total_shares > 0 else 0.0
        
        return is_cost, avg_exec_price, exec_prices

if __name__ == "__main__":
    # Test the simulator
    config = MarketConfig()
    sim = MarketSimulator(config)
    
    # Generate a price path
    prices = sim.simulate_price_path(seed=42)
    print(f"Price path start: {prices[0]:.2f}, end: {prices[-1]:.2f}")
    
    # Create a naive TWAP strategy (sell equal amount every step)
    total_shares = 1000
    twap_schedule = np.full(sim.steps, total_shares / sim.steps)
    
    cost, avg_price, exec_prices = sim.calculate_execution_cost(twap_schedule, prices)
    
    print(f"Total Shares: {total_shares}")
    print(f"Arrival Price: {prices[0]:.2f}")
    print(f"Average Exec Price: {avg_price:.2f}")
    print(f"Implementation Shortfall: {cost:.2f}")
