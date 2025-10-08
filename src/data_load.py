import yfinance as yf
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, ticker, period, interval):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        # print(yf.Ticker(self.ticker).history(period=self.period, interval=self.interval))

    def download(self, save_path):
        print(f"Downloading data to folder: \"{save_path}\"")
        ticker = yf.Ticker(self.ticker)
        self.data = ticker.history(period=self.period, interval=self.interval)

        if self.data.empty:
            raise ValueError(f"No data downloaded for ticker: {self.ticker}")

        filename = f"{save_path}{self.ticker}_{self.period}_{self.interval}.csv"
        self.data.to_csv(filename)
        print(f"Saved data as file: \"{filename}\"")
        return self.data

    def load(self, filepath):
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return self.data

    def get_trading_days(self):
        if self.data is None:
            raise ValueError("Data not detected. Check data is loaded proprly!")
        
        self.data['Date'] = self.data.index.date
        trading_days = {}
        
        for date, group in self.data.groupby('Date'):
            group = group.between_time('9:30', '16:00') # Just use regular market hours data
            if len(group) > 0:
                trading_days[date] = group[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"Found {len(trading_days)} trading days!")
        return trading_days

    def compute_statistics(self, data):
        if data is None:
            raise ValueError("Data not detected. Check data is loaded properly!")

        stats = {
            'mean_return': data['Close'].pct_change().dropna().mean(),
            'volatility': data['Close'].pct_change().dropna().std(),
            'avg_volume': data['Volume'].mean(),
            'avg_price': data['Close'].mean(),
            'num_observations': len(data)
        }

        if self.interval == '1m':
            periods_per_day = 390
        elif self.interval == '5m':
            periods_per_day = 390/5
        else:
            periods_per_day = 1
        
        stats['annual_volatility'] = stats['volatility'] * np.sqrt(252 * periods_per_day)
        return stats

if __name__ == "__main__":
    loader = DataLoader(ticker="SPY", period='1mo', interval="1d")
    #data = loader.download("data/raw/")
    data = loader.load("data/raw/SPY_7d_1m.csv")

    stats = loader.compute_statistics(data)
    print("\nData Statistics:")
    for key, val in stats.items():
        print(f"    {key}: {val:.6f}")
    
    days = loader.get_trading_days()
    print(f"\nFirst day sample:")
    first_day = list(days.values())[0]
    print(first_day.head())