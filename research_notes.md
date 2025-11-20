# Research Notes: Almgren-Chriss & RL Execution

## Almgren-Chriss Model (2000)

The Almgren-Chriss model provides a framework for optimal execution of portfolio transactions, balancing transaction costs (market impact) against volatility risk.

### Key Variables
- $X$: Total number of shares to sell (initial inventory).
- $T$: Total time horizon for liquidation.
- $N$: Number of trading intervals.
- $\tau = T/N$: Length of each interval.
- $\sigma$: Volatility of the asset (daily or per interval).
- $\eta$: Temporary market impact coefficient.
- $\theta$: Permanent market impact coefficient.

### Price Dynamics
The asset price $S_t$ evolves according to an arithmetic random walk with permanent impact:
```math
S_t = S_{t-1} + \sigma \sqrt{\tau} \xi_t - \theta \times n_t
```
where:
- $\xi_t \sim N(0, 1)$ is a standard normal random variable.
- $n_t$ is the number of shares sold in interval $t$.
- The permanent impact $\theta \times n_t$ depresses the price for all future times.

### Execution Price
The actual execution price $\tilde{S}_t$ includes the temporary impact:
$$ \tilde{S}_t = S_t - \eta \frac{n_t}{\tau} $$
where $\eta \frac{n_t}{\tau}$ is the temporary impact cost, proportional to the rate of trading.

### Implementation Shortfall (IS)
The total cost of execution is defined as the difference between the initial value of the portfolio and the final cash received.
$$ IS = X S_0 - \sum_{t=1}^N n_t \tilde{S}_t $$

## RL Environment Design

### State Space
The agent needs to know:
1.  **Remaining Inventory**: How much is left to sell.
2.  **Time Remaining**: How many steps are left.
3.  **Market Conditions**: Current price, volatility, maybe recent volume (if using volume profiles).

Proposed Observation Vector: `[inventory_left / X_0, time_left / T, (current_price - arrival_price) / arrival_price]`

### Action Space
The agent decides how much to sell in the current interval.
- **Action**: Fraction of *remaining* inventory to sell, or fraction of *initial* inventory.
- **Constraint**: Must sell all shares by time $T$.
- **Type**: Continuous `Box(0, 1)`.

### Reward Function
We want to minimize Implementation Shortfall.
- **Step Reward**: Change in portfolio value (Cash + Stock Value) relative to a benchmark (e.g., arrival price).
- Alternatively, a terminal reward equal to $-IS$.
- To encourage smooth trading, we might add a penalty for high variance or extreme actions.

## Synthetic Data Generation
We will use Geometric Brownian Motion (GBM) or Arithmetic Brownian Motion (ABM) for the "unaffected" price process.
- **GBM**: $S_t = S_{t-1} e^{(\mu - 0.5\sigma^2)\tau + \sigma\sqrt{\tau}\xi_t}$
- **ABM**: $S_t = S_{t-1} + \sigma\sqrt{\tau}\xi_t$ (Simpler, used in original AC paper).

We will start with ABM/GBM and add the impact components on top during the simulation.
