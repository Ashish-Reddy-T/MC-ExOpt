import gymnasium as gym
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from market_simulator import MarketSimulator, MarketConfig
from rl_env import ExecutionEnv
from backtest_engine import BacktestEngine
import os

# Create directories
os.makedirs("models/sweep", exist_ok=True)

def train_agent(config=None):
    with wandb.init(config=config, project="MC-ExOpt-Sweep", sync_tensorboard=True) as run:
        config = wandb.config
        
        # 1. Setup Environment with specific Risk Aversion
        market_config = MarketConfig() # Default market
        sim = MarketSimulator(market_config)
        env = ExecutionEnv(sim)
        
        # Manually set risk aversion (hacky but works for now, ideally pass to init)
        # We need to modify rl_env.py to accept this, or just monkey-patch it here
        # Let's monkey-patch for the sweep to avoid changing signature everywhere
        # Actually, let's just subclass or wrapper it? No, monkey patch is easier for script.
        # Wait, we can just set it on the env instance.
        # But env is re-created inside DummyVecEnv if we used that. 
        # SB3 wraps env. Let's assume single env for now or just set it.
        env.risk_aversion = config.risk_aversion
        
        # 2. Initialize Agent
        model = PPO("MlpPolicy", env, 
                    verbose=1, 
                    learning_rate=config.learning_rate,
                    ent_coef=config.ent_coef,
                    gamma=config.gamma,
                    tensorboard_log=f"runs/{run.id}")
        
        # 3. Train
        model.learn(total_timesteps=config.total_timesteps, 
                    callback=WandbCallback(
                        gradient_save_freq=100,
                        model_save_path=f"models/sweep/{run.id}",
                        verbose=2,
                    ))
        
        # 4. Evaluate (Quick Backtest)
        # Run on 50 scenarios to get a score
        print("Running validation backtest...")
        # We need to save the model first to load it into BacktestEngine, 
        # or we can just use the model object directly if we modify BacktestEngine.
        # Let's modify BacktestEngine to accept a model object.
        
        # For now, save and load.
        model_path = f"models/sweep/model_{run.id}"
        model.save(model_path)
        
        # Quick evaluation
        engine = BacktestEngine(model_path=model_path, scenarios_file="data/scenarios.pkl")
        # Hack: Limit scenarios to 50 for speed
        engine.scenarios = engine.scenarios[:50] 
        df = engine.run_backtest()
        
        avg_improvement = df["improvement_pct"].mean()
        wandb.log({"val_improvement_pct": avg_improvement})
        print(f"Validation Improvement: {avg_improvement:.2f}%")

def run_sweep():
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_improvement_pct',
            'goal': 'maximize'
        },
        'parameters': {
            'risk_aversion': {
                'values': [0.1, 0.5, 1.0, 2.0, 5.0]
            },
            'learning_rate': {
                'values': [1e-3, 3e-4, 1e-4]
            },
            'ent_coef': {
                'values': [0.0, 0.01, 0.05]
            },
            'gamma': {
                'values': [0.99, 0.999]
            },
            'total_timesteps': {
                'value': 200000
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="MC-ExOpt-Sweep")
    wandb.agent(sweep_id, train_agent, count=20)

if __name__ == "__main__":
    run_sweep()
