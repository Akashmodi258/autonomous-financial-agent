"""
SQLAlchemy layer — persists every research/calculation run so the
agent has a queryable history (SQL skill showcase).
"""
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = "sqlite:///research_agent.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ResearchQuery(Base):
    """One row per ticker analysed: snapshot of price, computed metrics,
    the ML prediction, and the resulting recommendation."""

    __tablename__ = "research_queries"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(16), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    current_price = Column(Float)
    volatility = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    rsi_14 = Column(Float)
    sharpe_ratio = Column(Float)

    predicted_next_close = Column(Float)
    recommendation = Column(String(16))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def log_query(record: dict) -> ResearchQuery:
    """Insert one research record and return the persisted row."""
    init_db()
    session = SessionLocal()
    try:
        row = ResearchQuery(**record)
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    finally:
        session.close()


def get_history(limit: int = 50):
    """Return the most recent `limit` research queries, newest first."""
    init_db()
    session = SessionLocal()
    try:
        return (
            session.query(ResearchQuery)
            .order_by(ResearchQuery.timestamp.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()
