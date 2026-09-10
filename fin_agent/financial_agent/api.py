"""
Flask RESTful API — same core logic exposed over HTTP.
Run standalone with:  python -m financial_agent.api
(Separate from the Streamlit app so Streamlit Cloud, which only
runs one process, doesn't need this — but it's here to demonstrate
the Flask / REST API skill as its own service.)
"""
from flask import Flask, jsonify

from financial_agent import data_fetcher, analytics, ml_model, database

app = Flask(__name__)


def _analyze(ticker: str) -> dict:
    df = data_fetcher.fetch_price_history(ticker)
    metrics = analytics.run_all_metrics(df)
    predicted = ml_model.predict_next_close(df)
    rec = ml_model.recommend(metrics["current_price"], predicted, metrics["rsi_14"])
    return {**metrics, "predicted_next_close": predicted, "recommendation": rec}


@app.get("/api/research/<ticker>")
def research(ticker):
    try:
        info = data_fetcher.fetch_company_info(ticker)
        return jsonify({"ticker": ticker.upper(), **info})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/calculate/<ticker>")
def calculate(ticker):
    try:
        df = data_fetcher.fetch_price_history(ticker)
        return jsonify({"ticker": ticker.upper(), **analytics.run_all_metrics(df)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/predict/<ticker>")
def predict(ticker):
    try:
        result = _analyze(ticker)
        database.log_query({"ticker": ticker.upper(), **{
            k: result[k] for k in
            ["current_price", "volatility", "sma_20", "sma_50",
             "rsi_14", "sharpe_ratio", "predicted_next_close", "recommendation"]
        }})
        return jsonify({"ticker": ticker.upper(), **result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/history")
def history():
    rows = database.get_history()
    return jsonify([{
        "ticker": r.ticker,
        "timestamp": r.timestamp.isoformat(),
        "current_price": r.current_price,
        "predicted_next_close": r.predicted_next_close,
        "recommendation": r.recommendation,
    } for r in rows])


if __name__ == "__main__":
    database.init_db()
    app.run(debug=True, port=5000)
