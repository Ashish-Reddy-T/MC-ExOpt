import numpy as np

class Strategy:
    def get_trade_schedule(self, total_shares: float, steps: int) -> np.ndarray:
        raise NotImplementedError

class TWAPStrategy(Strategy):
    def get_trade_schedule(self, total_shares: float, steps: int) -> np.ndarray:
        """
        Time-Weighted Average Price strategy: Sell equal amount every step.
        """
        # Simple equal split
        schedule = np.full(steps, total_shares / steps)
        return schedule