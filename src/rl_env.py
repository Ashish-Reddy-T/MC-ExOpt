import gymnasium as gym
import numpy as np
from gymnasium import spaces
from market_simulator import MarketSimulator, MarketConfig

class ExecutionEnv(gym.Env):
    """
    Gymnasium environment for optimal execution.
    """
    def __init__(self, simulator: MarketSimulator, total_shares: float = 1000.0):
        super(ExecutionEnv, self).__init__()
        self.simulator = simulator
        self.total_shares = total_shares
        
        # Action Space: Multiplier on TWAP rate.
        # Range: [-1, 1] mapped to e.g. [0, 2] or [0.5, 1.5]
        # Let's allow it to go from 0 (stop selling) to 2x TWAP (sell fast).
        # Action 0 -> 1x TWAP.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # State Space:
        # 0: Remaining Inventory (normalized by total_shares)
        # 1: Time Remaining (normalized by total_time)
        # 2: Current Price Deviation (log return from arrival price)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulator state
        # We generate a new price path for each episode
        # In a real training loop, we might want to sample from pre-generated scenarios
        self.price_path = self.simulator.simulate_price_path(seed=seed)
        self.current_step = 0
        self.inventory = self.total_shares
        self.arrival_price = self.price_path[0]
        self.cash = 0.0
        
        # Track trajectory for rendering/analysis
        self.trajectory = {
            "inventory": [self.inventory],
            "cash": [self.cash],
            "actions": [],
            "prices": [self.arrival_price]
        }
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize inputs
        norm_inventory = self.inventory / self.total_shares
        norm_time = (self.simulator.steps - self.current_step) / self.simulator.steps
        
        current_price = self.price_path[self.current_step]
        price_dev = np.log(current_price / self.arrival_price)
        
        return np.array([norm_inventory, norm_time, price_dev], dtype=np.float32)

    def step(self, action):
        # Calculate TWAP rate for remaining steps
        steps_remaining = self.simulator.steps - self.current_step
        if steps_remaining <= 0:
            twap_rate = self.inventory
        else:
            twap_rate = self.inventory / steps_remaining
            
        # Action is multiplier on TWAP rate
        # Map [-1, 1] to [0.1, 3.0] to FORCE trading
        multiplier = action[0] + 1.5 # Center around 1.5
        multiplier = np.clip(multiplier, 0.1, 3.0) # Force at least 10% of TWAP
        
        shares_to_sell = twap_rate * multiplier
        
        # Cap at remaining inventory
        shares_to_sell = min(shares_to_sell, self.inventory)
        
        # If this is the last step, we MUST sell everything
        if self.current_step >= self.simulator.steps - 1:
            shares_to_sell = self.inventory
            
        # Calculate execution price for this trade
        # We need to handle the simulator logic for a single step
        # Re-using logic from MarketSimulator.calculate_execution_cost but for single step
        
        current_price = self.price_path[self.current_step]
        
        # Permanent Impact (simplified: based on cumulative sold BEFORE this trade? 
        # Or we can track it. Let's assume permanent impact is already in the price path 
        # if we were simulating dynamically, but here price path is pre-generated (unaffected).
        # So we need to add impact.)
        
        cumulative_sold = self.total_shares - self.inventory
        perm_impact = self.simulator.config.theta * cumulative_sold
        
        # Temporary Impact
        # rate = shares_to_sell / dt
        temp_impact = self.simulator.config.eta * (shares_to_sell / self.simulator.config.dt)
        
        exec_price = current_price - perm_impact - temp_impact
        
        # Execute trade
        revenue = shares_to_sell * exec_price
        self.cash += revenue
        self.inventory -= shares_to_sell
        
        # Update step
        self.current_step += 1
        terminated = self.current_step >= self.simulator.steps
        truncated = False
        
        # Reward Calculation
        # 1. Implementation Shortfall Component: (Revenue - Cost at Arrival)
        #    This is equivalent to: shares_to_sell * (exec_price - arrival_price)
        is_reward = shares_to_sell * (exec_price - self.arrival_price)
        
        # 2. Risk Penalty Component (Almgren-Chriss style)
        #    Penalize holding inventory. 
        #    Penalty = -lambda * (inventory / total_shares)^2
        #    This encourages selling earlier to reduce variance.
        #    Use self.risk_aversion if set (by sweep), else default to 1.0
        risk_aversion = getattr(self, 'risk_aversion', 1.0)
        risk_penalty = -risk_aversion * (self.inventory / self.total_shares)**2
        
        step_reward = is_reward + risk_penalty
        
        # Optional: Penalty for volatility/risk?
        # step_reward -= lambda * variance?
        
        # Update trajectory
        self.trajectory["inventory"].append(self.inventory)
        self.trajectory["cash"].append(self.cash)
        self.trajectory["actions"].append(multiplier)
        if not terminated:
             self.trajectory["prices"].append(self.price_path[self.current_step])
        
        return self._get_obs(), step_reward, terminated, truncated, {}

    def render(self):
        # Print current status
        print(f"Step: {self.current_step}, Inventory: {self.inventory:.2f}, Cash: {self.cash:.2f}")
