import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from market_simulator import MarketSimulator, MarketConfig
from strategies import TWAPStrategy
from rl_env import ExecutionEnv
import pickle

class BacktestEngine:
    def __init__(self, model_path: str, scenarios_file: str):
        self.model = PPO.load(model_path)
        with open(scenarios_file, "rb") as f:
            self.scenarios = pickle.load(f)
        self.config = MarketConfig() # Use default config for now
        self.simulator = MarketSimulator(self.config)
        
    def run_backtest(self):
        results = []
        
        print(f"Backtesting on {len(self.scenarios)} scenarios...")
        
        for scenario in self.scenarios:
            price_path = scenario["price_path"]
            # Override simulator price path logic or just use the path directly
            # The simulator.calculate_execution_cost takes a price path
            
            # 1. Run TWAP
            twap = TWAPStrategy()
            twap_schedule = twap.get_trade_schedule(1000, self.simulator.steps)
            twap_cost, twap_avg_price, _ = self.simulator.calculate_execution_cost(twap_schedule, price_path)
            
            # 2. Run RL Agent
            # We need to step through the environment
            env = ExecutionEnv(self.simulator, total_shares=1000)
            # Manually set the price path for the env to match the scenario
            env.reset()
            env.price_path = price_path
            env.arrival_price = price_path[0]
            
            obs, _ = env.reset()
            # Force reset to use the specific price path again because env.reset() generates a new one
            env.price_path = price_path
            env.arrival_price = price_path[0]
            
            done = False
            agent_schedule = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                # We can recover the trade amount from the trajectory or just track it
                # But env.step updates inventory.
                # Let's look at the trajectory after done
            
            # Reconstruct schedule from trajectory
            inventory_hist = env.trajectory["inventory"]
            agent_schedule = -np.diff(inventory_hist)
            
            if scenario["id"] == 0:
                print("DEBUG: Scenario 0 Trajectory")
                print(f"Actions (First 10): {env.trajectory['actions'][:10]}")
                print(f"Inventory (First 10): {inventory_hist[:10]}")
                print(f"Inventory (Last 10): {inventory_hist[-10:]}")
                print(f"Agent Schedule (Last 10): {agent_schedule[-10:]}")
            
            # Calculate cost for Agent using the SAME simulator function for consistency
            agent_cost, agent_avg_price, _ = self.simulator.calculate_execution_cost(agent_schedule, price_path)
            
            results.append({
                "scenario_id": scenario["id"],
                "twap_cost": twap_cost,
                "agent_cost": agent_cost,
                "improvement": twap_cost - agent_cost, # Positive means Agent cost < TWAP cost
                "improvement_pct": (twap_cost - agent_cost) / twap_cost * 100 if twap_cost != 0 else 0
            })
            
        df = pd.DataFrame(results)
        avg_improvement = df["improvement_pct"].mean()
        print(f"Average Improvement over TWAP: {avg_improvement:.2f}%")
        
        return df

if __name__ == "__main__":
    engine = BacktestEngine("models/ppo_execution_agent", "data/scenarios.pkl")
    results = engine.run_backtest()
    results.to_csv("data/backtest_results.csv", index=False)
    print("Results saved to data/backtest_results.csv")
