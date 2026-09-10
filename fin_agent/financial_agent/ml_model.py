"""
scikit-learn layer — trains a lightweight regression on the fetched
history to project the next close price, and derives a simple
recommendation from it plus the RSI reading.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def predict_next_close(df: pd.DataFrame, lookback: int = 60) -> float:
    """
    Fits a linear regression on [day_index, SMA-5, SMA-10] -> next close,
    using the most recent `lookback` sessions, then projects one step
    ahead. Simple and explainable rather than state-of-the-art —
    the point is a clean, working ML step in the pipeline.
    """
    data = df.tail(lookback).copy().reset_index(drop=True)
    data["day_idx"] = np.arange(len(data))
    data["sma_5"] = data["Close"].rolling(5).mean()
    data["sma_10"] = data["Close"].rolling(10).mean()
    data = data.dropna().reset_index(drop=True)

    if len(data) < 10:
        # Not enough history for a meaningful fit — fall back to last close.
        return float(df["Close"].iloc[-1])

    X = data[["day_idx", "sma_5", "sma_10"]].values
    y = data["Close"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    next_row = np.array([[
        data["day_idx"].iloc[-1] + 1,
        data["Close"].tail(5).mean(),
        data["Close"].tail(10).mean(),
    ]])
    next_scaled = scaler.transform(next_row)
    prediction = model.predict(next_scaled)[0]
    return float(prediction)


def recommend(current_price: float, predicted_price: float, rsi_value: float) -> str:
    """Blend the ML trend signal with RSI overbought/oversold levels."""
    pct_move = (predicted_price - current_price) / current_price * 100

    if rsi_value >= 70:
        return "SELL"  # overbought, regardless of predicted drift
    if rsi_value <= 30:
        return "BUY"   # oversold
    if pct_move > 1:
        return "BUY"
    if pct_move < -1:
        return "SELL"
    return "HOLD"
