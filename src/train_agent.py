import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from market_simulator import MarketSimulator, MarketConfig
from rl_env import ExecutionEnv
import os

def train_agent():
    # Create log dir
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize Simulator and Env
    config = MarketConfig()
    sim = MarketSimulator(config)
    
    # We need a way to pass the simulator to the env when using make_vec_env
    # For simplicity, let's just use a lambda or wrapper, or single env for now.
    env = ExecutionEnv(sim)
    
    # Initialize PPO Agent
    # Initialize PPO Agent with tuned hyperparameters
    # learning_rate: Lower is often more stable
    # ent_coef: Higher entropy encourages exploration (prevents getting stuck in "sell nothing" or "sell all")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=3e-4,
                ent_coef=0.01,
                batch_size=64,
                n_steps=2048)
    
    # Train
    print("Starting training...")
    model.learn(total_timesteps=200000)
    print("Training complete.")
    
    # Save model
    model.save("models/ppo_execution_agent")
    print("Model saved to models/ppo_execution_agent")

if __name__ == "__main__":
    train_agent()
