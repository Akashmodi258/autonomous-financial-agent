"""
Autonomous Financial Research & Calculation Agent — Streamlit UI.

Deployable as-is on Streamlit Community Cloud: this file calls the
financial_agent package's functions directly (no separate server
needed for deployment). The Flask REST API in financial_agent/api.py
exposes the same logic over HTTP for local/standalone use.
"""
import pandas as pd
import streamlit as st

from financial_agent import data_fetcher, analytics, ml_model, database

st.set_page_config(page_title="Financial Research Agent",  layout="wide")

st.title(" Autonomous Financial Research & Calculation Agent")
st.caption(
    "Enter a ticker → the agent fetches market data, runs financial "
    "calculations, predicts the next close with a regression model, "
    "and logs the run to a SQL database."
)

with st.sidebar:
    st.header("Run Research")
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
    period = st.selectbox("History window", ["3mo", "6mo", "1y", "2y"], index=1)
    run = st.button("🔍 Run Agent", use_container_width=True)

if run and ticker:
    try:
        with st.spinner(f"Researching {ticker}..."):
            info = data_fetcher.fetch_company_info(ticker)
            df = data_fetcher.fetch_price_history(ticker, period=period)
            metrics = analytics.run_all_metrics(df)
            predicted = ml_model.predict_next_close(df)
            rec = ml_model.recommend(metrics["current_price"], predicted, metrics["rsi_14"])

            database.log_query({
                "ticker": ticker,
                "current_price": metrics["current_price"],
                "volatility": metrics["volatility"],
                "sma_20": metrics["sma_20"],
                "sma_50": metrics["sma_50"],
                "rsi_14": metrics["rsi_14"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "predicted_next_close": predicted,
                "recommendation": rec,
            })

        # --- Company snapshot ---
        st.subheader(f"{info['name']} ({ticker})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"{metrics['current_price']:.2f}")
        c2.metric("Predicted Next Close", f"{predicted:.2f}",
                   delta=f"{predicted - metrics['current_price']:.2f}")
        c3.metric("RSI (14)", f"{metrics['rsi_14']:.1f}")
        c4.metric("Recommendation", rec)

        st.write(
            f"**Sector:** {info['sector']}  |  **Industry:** {info['industry']}  |  "
            f"**P/E:** {info['pe_ratio']}  |  **Market Cap:** {info['market_cap']}"
        )

        # --- Price chart ---
        st.subheader("Price History")
        chart_df = df.set_index("Date")[["Close"]]
        st.line_chart(chart_df)

        # --- Metrics table ---
        st.subheader("Calculated Metrics")
        st.table(pd.DataFrame({
            "Metric": ["Volatility (annualized)", "SMA 20", "SMA 50", "RSI (14)", "Sharpe Ratio"],
            "Value": [
                f"{metrics['volatility']:.4f}",
                f"{metrics['sma_20']:.2f}",
                f"{metrics['sma_50']:.2f}",
                f"{metrics['rsi_14']:.2f}",
                f"{metrics['sharpe_ratio']:.2f}",
            ],
        }))

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.subheader(" Research History")
history_rows = database.get_history()
if history_rows:
    hist_df = pd.DataFrame([{
        "Ticker": r.ticker,
        "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
        "Price": round(r.current_price, 2),
        "Predicted Next": round(r.predicted_next_close, 2),
        "Recommendation": r.recommendation,
    } for r in history_rows])
    st.dataframe(hist_df, use_container_width=True)
else:
    st.info("No queries logged yet — run the agent from the sidebar.")
