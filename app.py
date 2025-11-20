import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import PPO
from market_simulator import MarketSimulator, MarketConfig
from strategies import TWAPStrategy
from rl_env import ExecutionEnv

st.set_page_config(page_title="MC-ExOpt Dashboard", layout="wide")

st.title("Monte Carlo-Drive Execution Optimizer")
st.markdown("""
**Resume Point**: *Developed vectorized back testing engine with Streamlit dashboard enabling interactive replay of execution strategies.*
""")

# Load Data
@st.cache_data
def load_data():
    with open("data/scenarios.pkl", "rb") as f:
        scenarios = pickle.load(f)
    results = pd.read_csv("data/backtest_results.csv")
    return scenarios, results

scenarios, results = load_data()

# Sidebar
st.sidebar.header("Settings")
scenario_id = st.sidebar.number_input("Select Scenario ID", min_value=0, max_value=len(scenarios)-1, value=0)

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Scenario Analysis")
    scenario = scenarios[scenario_id]
    price_path = scenario["price_path"]
    
    fig, ax = plt.subplots()
    ax.plot(price_path, label="Unaffected Price")
    ax.set_title(f"Price Path (Scenario {scenario_id})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Strategy Comparison")
    
    # Run Strategies for this specific scenario
    config = MarketConfig()
    sim = MarketSimulator(config)
    
    # TWAP
    twap = TWAPStrategy()
    twap_schedule = twap.get_trade_schedule(1000, sim.steps)
    twap_cost, _, twap_exec_prices = sim.calculate_execution_cost(twap_schedule, price_path)
    
    # Agent
    model = PPO.load("models/ppo_execution_agent")
    env = ExecutionEnv(sim, total_shares=1000)
    env.reset()
    env.price_path = price_path
    env.arrival_price = price_path[0]
    obs, _ = env.reset()
    env.price_path = price_path # Force again
    env.arrival_price = price_path[0]
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        
    inv_hist = env.trajectory["inventory"]
    agent_schedule = -np.diff(inv_hist)
    agent_cost, _, agent_exec_prices = sim.calculate_execution_cost(agent_schedule, price_path)
    
    # Metrics
    st.metric("TWAP Cost (IS)", f"${twap_cost:.2f}")
    st.metric("Agent Cost (IS)", f"${agent_cost:.2f}", delta=f"{twap_cost - agent_cost:.2f}")
    
    # Plot Execution
    fig2, ax2 = plt.subplots()
    ax2.plot(np.cumsum(twap_schedule), label="TWAP Sold")
    ax2.plot(np.cumsum(agent_schedule), label="Agent Sold")
    ax2.set_title("Cumulative Execution Profile")
    ax2.legend()
    st.pyplot(fig2)

st.divider()

st.subheader("Aggregate Performance")
st.dataframe(results.describe())

fig3, ax3 = plt.subplots()
ax3.hist(results["improvement_pct"], bins=50)
ax3.set_title("Distribution of Agent Improvement over TWAP (%)")
ax3.set_xlabel("Improvement %")
st.pyplot(fig3)
