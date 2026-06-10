"""
Multi-Timeframe Consensus Engine.
Aggregates signals across 1m, 5m, 15m, 1h timeframes.
A BUY/SELL signal is only valid when MAJORITY of timeframes agree.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from database import load_candles
from signals.indicators import SignalEngine, TechnicalIndicators


TIMEFRAME_WEIGHTS = {
    "1m":  0.05,   # Very low weight - too noisy
    "5m":  0.15,   # Short-term
    "15m": 0.20,   # Good balance
    "30m": 0.25,   # Medium-term confirmation
    "1h":  0.20,   # Strong trend confirmation
    "4h":  0.15,   # Long-term bias
}


class ConsensusEngine:
    """
    Aggregates signals across multiple timeframes.
    Returns a final consensus signal only if majority agree.
    """

    def __init__(self):
        self.engine = SignalEngine(buy_threshold=65.0, sell_threshold=65.0)
        self.ti = TechnicalIndicators()

    def get_timeframe_signal(self, df: pd.DataFrame, pair: str, tf: str) -> Optional[Dict]:
        """Get signal for a single timeframe."""
        if len(df) < 30:
            return None
        try:
            result = self.engine.generate_signal(df, pair, tf)
            if not result:
                return None
            return {
                "tf": tf,
                "signal": result["signal"],
                "score": result["score"],
                "weight": TIMEFRAME_WEIGHTS.get(tf, 0.25)
            }
        except Exception as e:
            logger.debug("Consensus TF error {} {}: {}", pair, tf, e)
            return None

    async def get_consensus(self, pair: str) -> Dict:
        """
        Load data for all timeframes and calculate weighted consensus.
        Returns: {direction, confidence, agreement_pct, timeframe_details}
        """
        tf_results = []

        for tf in ["1m", "5m", "15m", "30m", "1h", "4h"]:
            try:
                df = load_candles(pair, tf, limit=200)
                sig = self.get_timeframe_signal(df, pair, tf)
                if sig:
                    tf_results.append(sig)
            except Exception as e:
                logger.debug("Failed to load {} for {}: {}", tf, pair, e)

        if not tf_results:
            return {"direction": "WAIT", "confidence": 0.0, "agreement_pct": 0.0, "details": []}

        # Calculate weighted votes
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0

        for r in tf_results:
            w = r["weight"]
            sig = r["signal"]
            score = r["score"]
            if sig in ("BUY", "STRONG_BUY"):
                buy_weight += w * (score / 100)
                total_weight += w
            elif sig in ("SELL", "STRONG_SELL"):
                sell_weight += w * (score / 100)
                total_weight += w
            # NEUTRAL signals don't contribute to total_weight

        if total_weight == 0:
            return {"direction": "WAIT", "confidence": 0.0, "agreement_pct": 0.0, "details": tf_results}

        buy_score = (buy_weight / total_weight) * 100 if total_weight > 0 else 0
        sell_score = (sell_weight / total_weight) * 100 if total_weight > 0 else 0

        # Need clear majority (55% weighted threshold)
        if buy_score > sell_score and buy_score >= 55:
            direction = "BUY"
            confidence = buy_score
            # Agreement = what % of timeframes agree
            agreeing = sum(1 for r in tf_results if r["signal"] in ("BUY", "STRONG_BUY"))
        elif sell_score > buy_score and sell_score >= 55:
            direction = "SELL"
            confidence = sell_score
            agreeing = sum(1 for r in tf_results if r["signal"] in ("SELL", "STRONG_SELL"))
        else:
            direction = "WAIT"
            confidence = max(buy_score, sell_score)
            agreeing = 0

        agreement_pct = (agreeing / len(tf_results)) * 100 if tf_results else 0

        # Apply agreement multiplier — weak agreement = reduce confidence
        if direction != "WAIT" and agreement_pct < 50:
            confidence *= 0.75  # Penalize split timeframes
            if confidence < 55:
                direction = "WAIT"

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "agreement_pct": round(agreement_pct, 1),
            "buy_score": round(buy_score, 2),
            "sell_score": round(sell_score, 2),
            "details": tf_results
        }


_consensus_engine: Optional[ConsensusEngine] = None

def get_consensus_engine() -> ConsensusEngine:
    global _consensus_engine
    if _consensus_engine is None:
        _consensus_engine = ConsensusEngine()
    return _consensus_engine
