import numpy as np
import pandas as pd

np.random.seed(42)  # makes data reproducible — same every run

class ScenarioBank:

    @staticmethod
    def _make_ohlcv(close_prices, volume_range=(100000, 500000)):
        n = len(close_prices)
        noise = np.random.uniform(0.002, 0.008, n)
        df = pd.DataFrame({
            'close': close_prices,
            'open':  close_prices * (1 - noise/2),
            'high':  close_prices * (1 + noise),
            'low':   close_prices * (1 - noise),
            'volume': np.random.randint(*volume_range, n).astype(float)
        })
        return df[['open','high','low','close','volume']]

    @staticmethod
    def generate_bull_trend(n=80):
        close = 100 + np.arange(n) * 0.8
        close += np.random.normal(0, 1.5, n)
        close = np.maximum(close, 1.0)
        return ScenarioBank._make_ohlcv(close)

    @staticmethod
    def generate_choppy_range(n=80):
        t = np.linspace(0, 4 * np.pi, n)
        close = 100 + 10 * np.sin(t)
        close += np.random.normal(0, 1.0, n)
        return ScenarioBank._make_ohlcv(close)

    @staticmethod
    def generate_flash_crash(n=80):
        close = np.ones(n) * 100.0
        for i in range(30, 34):
            close[i] = close[i-1] * 0.95   # drop 5% each step = -20% total
        for i in range(34, n):
            recovery = 1 + 0.003 * np.log1p(i - 34)
            close[i] = close[33] * recovery
        close += np.random.normal(0, 0.5, n)
        return ScenarioBank._make_ohlcv(close)


class IndicatorCalculator:

    @staticmethod
    def moving_average(series: pd.Series, period=20) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def rsi(series: pd.Series, period=14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series):
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, signal

    @staticmethod
    def bollinger_bands(series: pd.Series, period=20):
        ma = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std().fillna(0)
        return ma + 2*std, ma - 2*std   # upper, lower