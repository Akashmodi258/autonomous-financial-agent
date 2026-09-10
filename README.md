# Autonomous Financial Research & Calculation Agent

A small autonomous agent that, given a stock ticker:
1. **Researches** — pulls live price history and company fundamentals.
2. **Calculates** — computes volatility, moving averages, RSI, Sharpe ratio.
3. **Predicts** — fits a scikit-learn regression to project the next close.
4. **Recommends** — turns the prediction + RSI into a BUY/HOLD/SELL call.
5. **Remembers** — logs every run to a SQL database via SQLAlchemy.

## Tech stack (mapped to skills)
| Skill | Where it's used |
|---|---|
| Python | Entire project |
| scikit-learn | `financial_agent/ml_model.py` — LinearRegression + StandardScaler |
| SQLAlchemy / SQL | `financial_agent/database.py` — ORM models, query history |
| Flask / RESTful API | `financial_agent/api.py` — `/api/research`, `/api/calculate`, `/api/predict`, `/api/history` |
| Streamlit | `streamlit_app.py` — deployable UI |

`yfinance` is used only as the data pipe to get real market prices — there's no way to get live financial data without a data-source library, but every calculation, model, API route, and DB layer is custom code.

## Project structure
```
fin_agent/
├── streamlit_app.py          # Deployable UI (entry point for Streamlit Cloud)
├── requirements.txt
└── financial_agent/
    ├── data_fetcher.py       # yfinance data access
    ├── analytics.py          # financial calculations
    ├── ml_model.py           # scikit-learn prediction + recommendation logic
    ├── database.py           # SQLAlchemy models + session
    └── api.py                # Flask REST API (standalone demo of the same logic)
```

## Run locally
```bash
pip install -r requirements.txt

# Streamlit UI (the main deliverable)
streamlit run streamlit_app.py

# Optional: run the Flask REST API separately, to demo that skill on its own
python -m financial_agent.api
# then: curl http://localhost:5000/api/predict/AAPL
```


    
