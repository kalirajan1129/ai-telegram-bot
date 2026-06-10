"""
Database layer — SQLite for structured records, Parquet for OHLCV time-series.
No external databases required.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

try:
    from backend.core.config import settings
except ImportError:
    from config import settings

# ── SQLAlchemy Base ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ───────────────────────────────────────────────────────────────

class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    pair: Mapped[str] = mapped_column(sa.String(20), index=True)
    direction: Mapped[str] = mapped_column(sa.String(4))
    confidence: Mapped[float] = mapped_column(sa.Float)
    timeframe: Mapped[str] = mapped_column(sa.String(5))
    entry_price: Mapped[float] = mapped_column(sa.Float)
    target_price: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    reason: Mapped[str] = mapped_column(sa.Text, default="")
    features_json: Mapped[str] = mapped_column(sa.Text, default="{}")
    result: Mapped[Optional[str]] = mapped_column(sa.String(4), nullable=True)
    pnl_pct: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    closed_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    telegram_sent: Mapped[bool] = mapped_column(sa.Boolean, default=False)


class TradeResult(Base):
    __tablename__ = "trade_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(sa.ForeignKey("signals.id"))
    pair: Mapped[str] = mapped_column(sa.String(20), index=True)
    direction: Mapped[str] = mapped_column(sa.String(4))
    confidence: Mapped[float] = mapped_column(sa.Float)
    timeframe: Mapped[str] = mapped_column(sa.String(5))
    result: Mapped[str] = mapped_column(sa.String(4))
    pnl_pct: Mapped[float] = mapped_column(sa.Float)
    features_json: Mapped[str] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(sa.String(50), index=True)
    pair: Mapped[Optional[str]] = mapped_column(sa.String(20), nullable=True)
    accuracy: Mapped[float] = mapped_column(sa.Float)
    precision: Mapped[float] = mapped_column(sa.Float)
    recall: Mapped[float] = mapped_column(sa.Float)
    f1: Mapped[float] = mapped_column(sa.Float)
    auc: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    n_samples: Mapped[int] = mapped_column(sa.Integer)
    notes: Mapped[str] = mapped_column(sa.Text, default="")


class PairPerformance(Base):
    __tablename__ = "pair_performance"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(sa.String(20), index=True, unique=True)
    total_signals: Mapped[int] = mapped_column(sa.Integer, default=0)
    wins: Mapped[int] = mapped_column(sa.Integer, default=0)
    losses: Mapped[int] = mapped_column(sa.Integer, default=0)
    # FIX #7: Stored as 0–100 (percentage) for consistency with display layer
    win_rate: Mapped[float] = mapped_column(sa.Float, default=0.0)
    avg_confidence: Mapped[float] = mapped_column(sa.Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(sa.Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    model_name: Mapped[str] = mapped_column(sa.String(50))
    pair: Mapped[Optional[str]] = mapped_column(sa.String(20), nullable=True)
    status: Mapped[str] = mapped_column(sa.String(20))
    message: Mapped[str] = mapped_column(sa.Text, default="")
    metrics_json: Mapped[str] = mapped_column(sa.Text, default="{}")


# ── Engine & Session ──────────────────────────────────────────────────────────

_engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.DB_PATH}",
    echo=settings.DEBUG,
    pool_pre_ping=True,
)
AsyncSessionLocal = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Create all tables on first run."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialised at {}", settings.DB_PATH)


# ── Parquet helpers ───────────────────────────────────────────────────────────

def _parquet_path(pair: str, timeframe: str) -> Path:
    p = settings.PARQUET_DIR / pair.upper()
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{timeframe}.parquet"


def save_candles(pair: str, timeframe: str, df: pd.DataFrame) -> None:
    """Append/overwrite OHLCV candles to Parquet, deduplicating by timestamp."""
    path = _parquet_path(pair, timeframe)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.to_parquet(path, index=False, compression="snappy")


def load_candles(pair: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
    """Load the last `limit` candles from Parquet."""
    path = _parquet_path(pair, timeframe)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df.tail(limit).reset_index(drop=True)


# ── Repository helpers ────────────────────────────────────────────────────────

async def save_signal(signal_data: Dict[str, Any]) -> int:
    async with AsyncSessionLocal() as session:
        sig = Signal(**{
            "pair": signal_data["pair"],
            "direction": signal_data["direction"],
            "confidence": signal_data["confidence"],
            "timeframe": signal_data["timeframe"],
            "entry_price": signal_data["entry_price"],
            "target_price": signal_data.get("target_price"),
            "stop_loss": signal_data.get("stop_loss"),
            "reason": signal_data.get("reason", ""),
            "features_json": json.dumps(signal_data.get("features", {})),
            "result": "OPEN",
        })
        session.add(sig)
        await session.commit()
        await session.refresh(sig)
        return sig.id


async def close_signal(signal_id: int, result: str, pnl_pct: float) -> None:
    async with AsyncSessionLocal() as session:
        sig = await session.get(Signal, signal_id)
        if sig:
            sig.result = result
            sig.pnl_pct = pnl_pct
            sig.closed_at = datetime.now(timezone.utc)
            tr = TradeResult(
                signal_id=signal_id,
                pair=sig.pair,
                direction=sig.direction,
                confidence=sig.confidence,
                timeframe=sig.timeframe,
                result=result,
                pnl_pct=pnl_pct,
                features_json=sig.features_json,
            )
            session.add(tr)
            await session.commit()
        await _update_pair_performance(sig.pair, result, pnl_pct)


async def _update_pair_performance(pair: str, result: str, pnl_pct: float) -> None:
    async with AsyncSessionLocal() as session:
        stmt = sa.select(PairPerformance).where(PairPerformance.pair == pair)
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            row = PairPerformance(
                pair=pair, 
                total_signals=0, 
                wins=0, 
                losses=0, 
                win_rate=0.0, 
                avg_confidence=0.0, 
                profit_factor=0.0
            )
            session.add(row)
        
        # Ensure we don't have None values (fallback for existing dirty rows)
        row.total_signals = row.total_signals or 0
        row.wins = row.wins or 0
        row.losses = row.losses or 0
        
        row.total_signals += 1
        if result == "WIN":
            row.wins += 1
        else:
            row.losses += 1
        # FIX #7: Store win_rate as 0–100 percentage (was 0–1 ratio before)
        row.win_rate = round(row.wins / row.total_signals * 100, 1) if row.total_signals else 0.0
        row.updated_at = datetime.now(timezone.utc)
        await session.commit()


async def get_recent_trade_results(limit: int = 2000) -> List[Dict]:
    async with AsyncSessionLocal() as session:
        stmt = sa.select(TradeResult).order_by(TradeResult.created_at.desc()).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "pair": r.pair, "direction": r.direction, "confidence": r.confidence,
                "timeframe": r.timeframe, "result": r.result, "pnl_pct": r.pnl_pct,
                "features": json.loads(r.features_json),
            }
            for r in rows
        ]


async def get_open_signals() -> List[Signal]:
    async with AsyncSessionLocal() as session:
        stmt = sa.select(Signal).where(Signal.result == "OPEN")
        return list((await session.execute(stmt)).scalars().all())


async def get_signal_history(limit: int = 100) -> List[Dict]:
    async with AsyncSessionLocal() as session:
        stmt = sa.select(Signal).order_by(Signal.created_at.desc()).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "id": r.id, "pair": r.pair, "direction": r.direction,
                "confidence": r.confidence, "timeframe": r.timeframe,
                "entry_price": r.entry_price, "result": r.result,
                "pnl_pct": r.pnl_pct, "reason": r.reason,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


async def get_pair_stats() -> List[Dict]:
    async with AsyncSessionLocal() as session:
        stmt = sa.select(PairPerformance).order_by(PairPerformance.win_rate.desc())
        rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "pair": r.pair, "total": r.total_signals, "wins": r.wins,
                "losses": r.losses,
                # FIX #7: win_rate already stored as 0–100, no need to multiply again
                "win_rate": round(r.win_rate, 1),
                "profit_factor": r.profit_factor,
            }
            for r in rows
        ]


async def get_overall_stats() -> Dict:
    async with AsyncSessionLocal() as session:
        total = (await session.execute(sa.select(sa.func.count(Signal.id)))).scalar() or 0
        wins = (await session.execute(
            sa.select(sa.func.count(Signal.id)).where(Signal.result == "WIN")
        )).scalar() or 0
        losses = (await session.execute(
            sa.select(sa.func.count(Signal.id)).where(Signal.result == "LOSS")
        )).scalar() or 0
        avg_conf = (await session.execute(sa.select(sa.func.avg(Signal.confidence)))).scalar() or 0.0
        return {
            "total_signals": total, "wins": wins, "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "avg_confidence": round(avg_conf, 1),
        }

async def reset_stats() -> bool:
    """Clear all records from stats tables to reset calculations."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(sa.delete(Signal))
            await session.execute(sa.delete(TradeResult))
            await session.execute(sa.delete(PairPerformance))
            await session.commit()
        logger.info("All statistics have been reset.")
        return True
    except Exception as e:
        logger.error("Error resetting stats: {}", e)
        return False
