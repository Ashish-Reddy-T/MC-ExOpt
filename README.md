# MC-ExOpt

## Description

__Monte Carlo-Drive Execution Optimizer__

- Built market impact simulator using __Almgren-Chriss__ framework to model execution costs; generated 1,000+ synthetic trading scenarios with varying volatility and liquidity conditions to test order scheduling strategies.
- Trained __PPO__ reinforcement learning agent to optimize participation rates across market regimes, which reduced simulated implementation shortfall by 2.3% vs. VWAP baseline on historical tick data from 3 liquid ETFs.
- Developed vectorized back testing engine (5x faster than iterative baseline) with Streamlit dashboard enabling interactive replay of execution strategies and cost/risk trade-off visualization.

---