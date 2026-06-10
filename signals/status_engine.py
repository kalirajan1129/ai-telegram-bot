"""
Status Engine — generates per-pair status with confidence scores.
Shows BUY/SELL/WAIT for each pair every minute.

FIXES:
  1. Signal Lock: prevents BUY→SELL flip within validity window
  2. Multi-Timeframe Consensus: majority of TFs must agree for a signal
  3. Minimum confidence threshold raised to reduce noise
"""
from __future__ import annotations

import asyncio
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from database import load_candles
from technical import FEATURE_COLUMNS, adx, atr, build_feature_vector, rsi as calc_rsi
from ml.trainer import get_trainer
from signals.indicators import SignalEngine
from signals.advanced_analysis import AdvancedTechnicalAnalysis
from signals.signal_lock import get_lock_manager
from signals.consensus import get_consensus_engine


# Minimum timeframe agreement required to issue a signal
MIN_AGREEMENT_PCT = 50.0


class PairStatusEngine:
    """Generates per-pair trading status with confidence scores."""

    def __init__(self):
        self.signal_engine = SignalEngine(
            buy_threshold=settings.SIGNAL_STRENGTH_BUY,
            sell_threshold=settings.SIGNAL_STRENGTH_SELL
        )

    async def get_pair_status(self, pair: str) -> Dict:
        """
        Get trading status for a single pair.
        Returns: {pair, status: BUY/SELL/WAIT, confidence: 0-100, indicators: {...}}

        Changes from original:
          - Uses ConsensusEngine (multi-TF agreement required)
          - Uses SignalLockManager (no flip within validity window)
        """
        try:
            # Load recent candles for base calculations
            df = load_candles(pair, "5m", limit=500)
            if len(df) < 50:
                logger.debug("Insufficient candles for {}: {}", pair, len(df))
                return {
                    "pair": pair,
                    "status": "WAIT",
                    "confidence": 0.0,
                    "reason": f"Insufficient data ({len(df)} candles)",
                    "current_price": 0.0,
                    "indicators": {}
                }

            current_price = float(df["close"].iloc[-1])

            # Get indicator values
            adx_val = self._get_adx(df)
            rsi_val = self._get_rsi(df)
            atr_pct = self._get_atr_percent(df)

            # Advanced analysis
            adv = AdvancedTechnicalAnalysis()
            candlestick_patterns = adv.detect_candlestick_patterns(df)
            trend = adv.identify_trend(df)
            support_resist = adv.support_resistance(df)
            ma_cross = adv.moving_average_convergence(df)
            bb_position = adv.bollinger_band_position(df)

            fib_high = df["high"].tail(50).max()
            fib_low = df["low"].tail(50).min()
            fib_levels = adv.fibonacci_retracement(fib_high, fib_low)

            # Pattern boost
            pattern_boost = 0.0
            pattern_signal = None
            if candlestick_patterns:
                avg_strength = sum(p["strength"] for p in candlestick_patterns) / len(candlestick_patterns)
                pattern_boost = (avg_strength / 10) * 8
                pattern_signal = candlestick_patterns[0]["signal"]

            # ── MULTI-TIMEFRAME CONSENSUS ─────────────────────────────────────
            consensus = await get_consensus_engine().get_consensus(pair)
            consensus_dir = consensus["direction"]     # BUY / SELL / WAIT
            consensus_conf = consensus["confidence"]
            agreement_pct = consensus["agreement_pct"]

            # ── ML PREDICTION ─────────────────────────────────────────────────
            ml_prob = 0.5
            try:
                df_features = build_feature_vector(df.copy())
                df_features.dropna(subset=FEATURE_COLUMNS, inplace=True)
                if not df_features.empty and len(df_features) >= 50:
                    trainer = get_trainer()
                    ml_prob = trainer.predict_proba(df_features[FEATURE_COLUMNS].iloc[-1].values)
            except Exception as e:
                logger.debug("ML prediction error for {}: {}", pair, e)

            # ── FINAL CONFIDENCE ──────────────────────────────────────────────
            ml_score = ml_prob * 100
            # Get real technical score from SignalEngine (not reusing consensus)
            tech_sig = self.signal_engine.generate_signal(df, pair, settings.PRIMARY_TIMEFRAME)
            tech_score = tech_sig["score"] if tech_sig else 50.0
            # Blend: 40% consensus, 25% ML, 35% indicator score
            combined_confidence = (consensus_conf * 0.40 + ml_score * 0.25 + tech_score * 0.35)
            combined_confidence += pattern_boost
            combined_confidence = min(max(combined_confidence, 0), 100)

            # ── DIRECTION DECISION ────────────────────────────────────────────
            if consensus_dir == "WAIT" or agreement_pct < MIN_AGREEMENT_PCT:
                raw_status = "WAIT"
            elif combined_confidence >= settings.MIN_CONFIDENCE:
                if pattern_signal and pattern_signal != consensus_dir:
                    # Pattern conflicts with consensus — reduce confidence
                    combined_confidence = max(combined_confidence - 8, 0)
                    if combined_confidence < settings.MIN_CONFIDENCE:
                        raw_status = "WAIT"
                    else:
                        raw_status = consensus_dir
                else:
                    raw_status = consensus_dir
            else:
                raw_status = "WAIT"

            # ── SIGNAL LOCK: prevents flip within validity window ─────────────
            duration_info = AdvancedTechnicalAnalysis.calculate_signal_duration(pair, combined_confidence, settings.PRIMARY_TIMEFRAME)
            validity_min = duration_info["minutes"]

            lock_mgr = get_lock_manager()
            lock_result = lock_mgr.resolve_status(pair, raw_status, combined_confidence, validity_min)

            final_status = lock_result["status"]
            final_confidence = lock_result["confidence"]
            lock_remaining = lock_result.get("lock_remaining_min", 0)

            # Recalculate duration display based on final signal
            final_duration = AdvancedTechnicalAnalysis.calculate_signal_duration(pair, final_confidence, settings.PRIMARY_TIMEFRAME)

            logger.debug(
                "{}: raw={} final={} conf={:.1f}% agreement={:.0f}% lock_rem={:.0f}min",
                pair, raw_status, final_status, final_confidence, agreement_pct, lock_remaining
            )

            return {
                "pair": pair,
                "status": final_status,
                "confidence": round(final_confidence, 2),
                "current_price": round(current_price, 6),
                "duration": final_duration,
                "reason": f"MTF consensus ({agreement_pct:.0f}% agree)" if final_status != "WAIT" else "No consensus",
                "lock_remaining_min": lock_remaining,
                "consensus": {
                    "direction": consensus_dir,
                    "agreement_pct": agreement_pct,
                    "buy_score": consensus.get("buy_score", 0),
                    "sell_score": consensus.get("sell_score", 0),
                    "tf_details": consensus.get("details", [])
                },
                "indicators": {
                    "ADX": round(adx_val, 2),
                    "ATR%": round(atr_pct * 100, 4),
                    "RSI": round(rsi_val, 2),
                    "ML_Prob": round(ml_prob * 100, 2),
                    "Consensus_Score": round(consensus_conf, 2),
                    "Agreement%": round(agreement_pct, 1),
                    "Trend": trend["direction"],
                    "Trend_Strength": round(trend["strength"] * 100, 2),
                    "MA_Signal": ma_cross["signal"],
                    "BB_Position": round(bb_position["position"], 1),
                    "Patterns": len(candlestick_patterns)
                },
                "advanced": {
                    "patterns": candlestick_patterns,
                    "support_resistance": support_resist,
                    "fib_levels": {k: round(v, 6) for k, v in fib_levels.items()},
                    "trend": trend,
                    "bb_position": bb_position,
                    "ma_cross": ma_cross
                }
            }

        except Exception as e:
            logger.error("Status error for {}: {}", pair, str(e)[:100])
            return {
                "pair": pair,
                "status": "ERROR",
                "confidence": 0.0,
                "reason": str(e)[:50],
                "current_price": 0.0,
                "indicators": {},
                "advanced": {}
            }

    def _indicators_to_confidence(self, adx: float, rsi: float) -> float:
        rsi_conf = 100 - abs(rsi - 50) * 2
        adx_conf = min(adx * 3, 100)
        return (rsi_conf * 0.4 + adx_conf * 0.6)

    def _get_adx(self, df: pd.DataFrame) -> float:
        try:
            adx_val, _, _ = adx(df["high"], df["low"], df["close"])
            adx_series = adx_val.dropna()
            return float(adx_series.iloc[-1]) if not adx_series.empty else 20.0
        except Exception as e:
            logger.debug("ADX error: {}", e)
            return 20.0

    def _get_atr_percent(self, df: pd.DataFrame) -> float:
        try:
            atr_val = atr(df["high"], df["low"], df["close"]).dropna()
            if not atr_val.empty and df["close"].iloc[-1] > 0:
                return float(atr_val.iloc[-1] / df["close"].iloc[-1])
            return 0.005
        except Exception as e:
            logger.debug("ATR error: {}", e)
            return 0.005

    def _get_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < period:
                return 50.0
            rsi_series = calc_rsi(df["close"], period)
            rsi_val = rsi_series.iloc[-1]
            return float(rsi_val) if pd.notna(rsi_val) else 50.0
        except Exception as e:
            logger.debug("RSI error: {}", e)
            return 50.0

    async def get_all_pairs_status(self, pairs: List[str]) -> Dict[str, Dict]:
        """Get status for all pairs concurrently."""
        tasks = [self.get_pair_status(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        status_dict = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("Pair status error: {}", result)
            else:
                status_dict[result["pair"]] = result

        return status_dict


# Global instance
_status_engine: Optional[PairStatusEngine] = None

def get_status_engine() -> PairStatusEngine:
    global _status_engine
    if _status_engine is None:
        _status_engine = PairStatusEngine()
    return _status_engine
