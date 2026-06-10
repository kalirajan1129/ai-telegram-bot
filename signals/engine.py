"""
Signal Engine — generates trading signals from ML + technical confluence.
Works for Crypto (Binance), Forex, and OTC pairs.
Includes multi-timeframe analysis with advanced indicators.
"""
from __future__ import annotations

import asyncio
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from database import load_candles, save_signal
from ml.trainer import get_trainer
from technical import FEATURE_COLUMNS, adx, atr, build_feature_vector, rsi
from signals.indicators import SignalEngine, TechnicalIndicators


_signal_counts: Dict[str, List[float]] = defaultdict(list)
_signal_engine = SignalEngine(
    buy_threshold=settings.SIGNAL_STRENGTH_BUY,
    sell_threshold=settings.SIGNAL_STRENGTH_SELL
)


def _rate_ok(pair: str) -> bool:
    now = datetime.now(timezone.utc).timestamp()
    _signal_counts[pair] = [t for t in _signal_counts[pair] if now - t < 3600]
    if len(_signal_counts[pair]) >= settings.MAX_SIGNALS_PER_HOUR:
        return False
    _signal_counts[pair].append(now)
    return True


async def _mtf_votes(pair: str) -> Tuple[int, int, float]:
    """Multi-timeframe ML vote. Returns (buy_votes, sell_votes, avg_confidence)."""
    trainer = get_trainer()
    buy_v, sell_v, confs = 0, 0, []
    for tf in ["5m", "15m", "30m", "1h", "4h"]:
        df = load_candles(pair, tf, limit=300)
        if len(df) < 50:
            continue
        try:
            df = build_feature_vector(df.copy())
            df.dropna(subset=FEATURE_COLUMNS, inplace=True)
            if df.empty:
                continue
            prob = trainer.predict_proba(df[FEATURE_COLUMNS].iloc[-1].values)
            confs.append(prob)
            if prob >= 0.5:
                buy_v += 1
            else:
                sell_v += 1
        except Exception as e:
            logger.debug("MTF error {} {}: {}", pair, tf, e)
    avg = float(np.mean(confs)) if confs else 0.5
    return buy_v, sell_v, avg


def _risk_ok(df: pd.DataFrame) -> bool:
    try:
        adx_v, _, _ = adx(df["high"], df["low"], df["close"])
        adx_series = adx_v.dropna()
        if adx_series.empty:
            return False
        if adx_series.iloc[-1] < settings.ADX_MIN_TREND:
            return False
        atr_series = atr(df["high"], df["low"], df["close"]).dropna()
        if atr_series.empty:
            return False
        atr_v = atr_series.iloc[-1]
        close = df["close"].iloc[-1]
        if close == 0:
            return False
        atr_pct = atr_v / close
        if not (settings.VOLATILITY_MIN <= atr_pct <= settings.VOLATILITY_MAX):
            return False
        return True
    except Exception as e:
        logger.debug("Risk check error: {}", e)
        return False


def _targets(df: pd.DataFrame, direction: str) -> Tuple[float, float, float]:
    c = df["close"].iloc[-1]
    a_series = atr(df["high"], df["low"], df["close"]).dropna()
    a = a_series.iloc[-1] if not a_series.empty else c * 0.001
    if direction == "BUY":
        return round(c, 6), round(c + 2 * a, 6), round(c - 1.5 * a, 6)
    return round(c, 6), round(c - 2 * a, 6), round(c + 1.5 * a, 6)


def _candlestick_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Quick candlestick pattern check for immediate signal confirmation.
    Returns 'BUY', 'SELL', or None.
    Uses last 3 candles.
    """
    if len(df) < 3:
        return None
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last["close"] - last["open"])
        total = last["high"] - last["low"]
        if total == 0:
            return None
        body_ratio = body / total

        # Hammer / Bullish pin bar
        lower_wick = min(last["open"], last["close"]) - last["low"]
        upper_wick = last["high"] - max(last["open"], last["close"])
        if lower_wick > 2 * body and upper_wick < body and body_ratio > 0.1:
            return "BUY"
        # Shooting star / Bearish pin bar
        if upper_wick > 2 * body and lower_wick < body and body_ratio > 0.1:
            return "SELL"
        # Bullish engulfing
        if (last["close"] > last["open"] and prev["close"] < prev["open"] and
                last["open"] < prev["close"] and last["close"] > prev["open"]):
            return "BUY"
        # Bearish engulfing
        if (last["close"] < last["open"] and prev["close"] > prev["open"] and
                last["open"] > prev["close"] and last["close"] < prev["open"]):
            return "SELL"
    except Exception:
        pass
    return None


async def generate_signal(pair: str) -> Optional[dict]:
    if not _rate_ok(pair):
        return None

    df = load_candles(pair, settings.PRIMARY_TIMEFRAME, limit=500)
    if len(df) < 50:
        logger.debug("Insufficient candles for {}: {}", pair, len(df))
        return None

    # For OTC pairs, skip strict risk checks and use technical signals directly
    try:
        # Generate technical signal
        tech_signal = _signal_engine.generate_signal(df, pair, settings.PRIMARY_TIMEFRAME)
        if not tech_signal:
            logger.debug("No technical signal for {}", pair)
            return None
        
        # Score is already 0-100 range from SignalEngine
        confidence = tech_signal.get("score", 0)
        if confidence < settings.MIN_CONFIDENCE:
            logger.debug("{} confidence too low: {:.1f}%", pair, confidence)
            return None
        
        direction = "BUY" if tech_signal["signal"] in ["BUY", "STRONG_BUY"] else "SELL" if tech_signal["signal"] in ["SELL", "STRONG_SELL"] else None
        if not direction:
            return None
            
        # Filter: Avoid sideways market using ADX (ADX < 25 indicates sideways)
        adx_line, _, _ = adx(df["high"], df["low"], df["close"])
        if adx_line.iloc[-1] < 25.0:
            logger.debug("Market is sideways for {} (ADX < 25)", pair)
            return None
            
        # Filter: Respect signal lock manager (so it doesn't fire BUY then SELL immediately)
        from signals.signal_lock import get_lock_manager
        from signals.advanced_analysis import AdvancedTechnicalAnalysis
        
        lock_mgr = get_lock_manager()
        if not lock_mgr.should_override_lock(pair, direction, confidence):
            logger.debug("Signal {} blocked by lock for {}", direction, pair)
            return None
        
        # Candlestick pattern confirmation bonus
        pattern_dir = _candlestick_signal(df)
        pattern_str = ""
        if pattern_dir == direction:
            confidence = min(confidence + 5.0, 99.9)
            pattern_str = " + candle pattern ✅"
        elif pattern_dir and pattern_dir != direction:
            confidence = max(confidence - 3.0, 0.0)
        
        entry, target, stop = _targets(df, direction)
        signal = {
            "pair": pair, "direction": direction,
            "confidence": round(confidence, 2),
            "timeframe": settings.PRIMARY_TIMEFRAME,
            "entry_price": entry, "target_price": target, "stop_loss": stop,
            "reason": f"Technical indicators ({len(tech_signal.get('indicators', []))} signals){pattern_str}",
            "features": {},
        }
        try:
            signal["id"] = await save_signal(signal)
            logger.info("Signal: {} {} @ {:.6f} ({:.1f}%)", pair, direction, entry, confidence)
            
            # Set lock to prevent immediate flips
            duration_info = AdvancedTechnicalAnalysis.calculate_signal_duration(pair, confidence, settings.PRIMARY_TIMEFRAME)
            lock_mgr.set_lock(pair, direction, confidence, duration_info["minutes"])
            
        except Exception as e:
            logger.error("Save signal failed: {}", e)
        return signal
    except Exception as e:
        logger.debug("Signal generation error for {}: {}", pair, e)
        return None


async def generate_mtf_signals(pair: str) -> List[dict]:
    """
    Generate multi-timeframe signals for 5m, 15m, 30m, 1h, 4h.
    Returns signals only when high-confidence BUY or SELL is detected.
    All timeframes support both BUY and SELL directions.
    """
    signals = []

    TIMEFRAME_CONFIG = {
        "5m":  {"min_score": 55, "label": "Short-term (5 min)"},
        "15m": {"min_score": 58, "label": "Short-term (15 min)"},
        "30m": {"min_score": 60, "label": "Medium-term (30 min)"},
        "1h":  {"min_score": 62, "label": "Mid-term (1 hour)"},
        "4h":  {"min_score": 65, "label": "Long-term (4 hour)"},
    }

    for timeframe, cfg in TIMEFRAME_CONFIG.items():
        try:
            df = load_candles(pair, timeframe, limit=300)
            if len(df) < 50:
                continue

            tech_sig = _signal_engine.generate_signal(df, pair, timeframe)
            if not tech_sig:
                continue

            score = tech_sig.get("score", 0)
            if score < cfg["min_score"]:
                continue

            raw_signal = tech_sig.get("signal", "NEUTRAL")
            if raw_signal in ("BUY", "STRONG_BUY"):
                action = "BUY"
            elif raw_signal in ("SELL", "STRONG_SELL"):
                action = "SELL"
            else:
                continue  # Skip NEUTRAL — no trade

            entry_price = df["close"].iloc[-1]
            entry, target, stop = _targets(df, action)

            signal = {
                "pair": pair,
                "timeframe": timeframe,
                "label": cfg["label"],
                "action": action,
                "entry_price": entry,
                "target_price": target,
                "stop_loss": stop,
                "confidence": round(score, 2),
                "signal_type": raw_signal,
                "indicators_used": len(tech_sig.get("indicators", [])),
                "timestamp": df.iloc[-1]["timestamp"],
            }
            signals.append(signal)
        except Exception as e:
            logger.debug("MTF signal error {} {}: {}", pair, timeframe, e)

    return signals


async def scan_pairs(pairs: List[str], on_signal: Optional[Callable] = None) -> List[dict]:
    results = []
    sem = asyncio.Semaphore(10)

    async def _one(pair):
        async with sem:
            try:
                sig = await generate_signal(pair)
                if sig:
                    results.append(sig)
                    if on_signal:
                        await on_signal(sig)
            except Exception as e:
                logger.debug("Scan error {}: {}", pair, e)

    await asyncio.gather(*[_one(p) for p in pairs])
    return results
