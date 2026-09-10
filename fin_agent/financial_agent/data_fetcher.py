"""
Market data access layer.

Note: fetching real financial data requires a data source library.
`yfinance` (free, no API key) is used here purely as a data pipe —
all research/calculation/prediction logic is your own code built on
Python, pandas-backed dataframes, scikit-learn, SQLAlchemy, and Flask.
"""
import yfinance as yf
import pandas as pd


def fetch_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Historical OHLCV data for `ticker`. Raises ValueError if empty/unknown ticker."""
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")
    data = data.reset_index()
    return data


def fetch_company_info(ticker: str) -> dict:
    """Basic fundamentals/company profile."""
    info = yf.Ticker(ticker).info or {}
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        "currency": info.get("currency", "N/A"),
    }
