import numpy as np
import pickle
import os
from market_simulator import MarketSimulator, MarketConfig

def generate_scenarios(n_scenarios: int = 1000, output_file: str = "data/scenarios.pkl"):
    """
    Generates N synthetic market scenarios and saves them to a file.
    """
    config = MarketConfig()
    sim = MarketSimulator(config)
    
    scenarios = []
    print(f"Generating {n_scenarios} scenarios...")
    
    for i in range(n_scenarios):
        # Randomize volatility slightly to create "varying volatility" conditions
        # Base sigma is 0.01, let's vary it between 0.005 and 0.02
        current_sigma = np.random.uniform(0.005, 0.02)
        sim.config.sigma = current_sigma
        
        price_path = sim.simulate_price_path(seed=i)
        scenarios.append({
            "id": i,
            "sigma": current_sigma,
            "price_path": price_path
        })
        
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "wb") as f:
        pickle.dump(scenarios, f)
        
    print(f"Saved {n_scenarios} scenarios to {output_file}")

if __name__ == "__main__":
    generate_scenarios()
