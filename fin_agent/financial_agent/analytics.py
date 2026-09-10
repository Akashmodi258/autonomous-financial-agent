"""
Calculation engine — the "Calculation" half of the agent.
Pure pandas/numpy math on top of the fetched price history.
"""
import numpy as np
import pandas as pd


def daily_returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().dropna()


def volatility(df: pd.DataFrame, annualize: bool = True) -> float:
    """Standard deviation of daily returns, optionally annualized."""
    rets = daily_returns(df)
    vol = rets.std()
    if annualize:
        vol *= np.sqrt(252)
    return float(vol)


def moving_averages(df: pd.DataFrame) -> dict:
    close = df["Close"]
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=min(50, len(close))).mean().iloc[-1]
    return {"sma_20": float(sma_20), "sma_50": float(sma_50)}


def rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Relative Strength Index (Wilder's method)."""
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    value = rsi_series.iloc[-1]
    return float(value) if not np.isnan(value) else 50.0


def sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.06) -> float:
    """Annualized Sharpe ratio (risk-free default ~ Indian T-bill rate)."""
    rets = daily_returns(df)
    excess = rets - (risk_free_rate / 252)
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def run_all_metrics(df: pd.DataFrame) -> dict:
    ma = moving_averages(df)
    return {
        "current_price": float(df["Close"].iloc[-1]),
        "volatility": volatility(df),
        "sma_20": ma["sma_20"],
        "sma_50": ma["sma_50"],
        "rsi_14": rsi(df),
        "sharpe_ratio": sharpe_ratio(df),
    }
