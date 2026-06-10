"""
Advanced Technical Analysis — Fibonacci, Support/Resistance, Trend Lines, Candlestick Patterns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class AdvancedTechnicalAnalysis:
    """Advanced technical analysis methods."""
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        Returns key levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
        """
        diff = high - low
        return {
            "0%": high,
            "23.6%": high - (diff * 0.236),
            "38.2%": high - (diff * 0.382),
            "50%": high - (diff * 0.5),
            "61.8%": high - (diff * 0.618),
            "78.6%": high - (diff * 0.786),
            "100%": low
        }
    
    @staticmethod
    def support_resistance(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """
        Find support and resistance levels using recent highs/lows.
        """
        if len(df) < lookback:
            return {"support": None, "resistance": None, "pivot": None}
        
        recent = df.tail(lookback)
        high = recent["high"].max()
        low = recent["low"].min()
        close = recent["close"].iloc[-1]
        
        # Pivot points (standard calculation)
        pivot = (high + low + close) / 3
        support_1 = (2 * pivot) - high
        resistance_1 = (2 * pivot) - low
        support_2 = pivot - (high - low)
        resistance_2 = pivot + (high - low)
        
        return {
            "support_2": support_2,
            "support_1": support_1,
            "pivot": pivot,
            "resistance_1": resistance_1,
            "resistance_2": resistance_2,
            "current": close,
            "distance_to_resistance": resistance_1 - close,
            "distance_to_support": close - support_1
        }
    
    @staticmethod
    def identify_trend(df: pd.DataFrame, period: int = 20) -> Dict[str, any]:
        """
        Identify trend direction and strength.
        Returns: direction (UP/DOWN), strength, consecutive candles
        """
        if len(df) < period:
            return {"direction": "NEUTRAL", "strength": 0.0, "up_candles": 0, "down_candles": 0}
        
        recent = df.tail(period)
        up_candles = sum(1 for i in range(len(recent)) if recent["close"].iloc[i] > recent["open"].iloc[i])
        down_candles = period - up_candles
        
        # Calculate trend strength
        returns = df["close"].pct_change().tail(period)
        up_avg = returns[returns > 0].mean()
        down_avg = abs(returns[returns < 0].mean())
        
        trend_strength = (up_avg - down_avg) / (up_avg + down_avg + 1e-10)
        
        if up_candles > down_candles * 1.5:
            direction = "UP"
        elif down_candles > up_candles * 1.5:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        return {
            "direction": direction,
            "strength": abs(trend_strength),
            "up_candles": up_candles,
            "down_candles": down_candles,
            "ratio": up_candles / down_candles if down_candles > 0 else up_candles
        }
    
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict]:
        """
        Detect common candlestick patterns in last few candles.
        Returns list of detected patterns with signal strength.
        """
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        body_last = abs(last["close"] - last["open"])
        body_prev = abs(prev["close"] - prev["open"])
        body_prev2 = abs(prev2["close"] - prev2["open"])
        
        total_last = last["high"] - last["low"]
        total_prev = prev["high"] - prev["low"]
        
        if total_last == 0 or total_prev == 0:
            return patterns
        
        body_ratio_last = body_last / total_last
        body_ratio_prev = body_prev / total_prev
        
        lower_wick = min(last["open"], last["close"]) - last["low"]
        upper_wick = last["high"] - max(last["open"], last["close"])
        
        # === BULLISH PATTERNS ===
        
        # Hammer: small body, long lower wick
        if lower_wick > body_last * 2 and upper_wick < body_last and body_ratio_last > 0.1:
            patterns.append({
                "name": "Hammer",
                "signal": "BUY",
                "strength": 7,
                "description": "Potential reversal from downtrend"
            })
        
        # Bullish Engulfing
        if (last["close"] > last["open"] and prev["close"] < prev["open"] and
                last["open"] < prev["close"] and last["close"] > prev["open"]):
            patterns.append({
                "name": "Bullish Engulfing",
                "signal": "BUY",
                "strength": 8,
                "description": "Strong reversal signal"
            })
        
        # Three White Soldiers
        if (len(df) >= 3 and 
            last["close"] > last["open"] and 
            prev["close"] > prev["open"] and 
            prev2["close"] > prev2["open"] and
            last["close"] > prev["close"] and 
            prev["close"] > prev2["close"]):
            patterns.append({
                "name": "Three White Soldiers",
                "signal": "BUY",
                "strength": 9,
                "description": "Strong bullish continuation"
            })
        
        # Piercing Line
        if (last["close"] > last["open"] and 
            prev["close"] < prev["open"] and
            last["open"] < prev["close"] and 
            last["close"] > prev["open"] * 0.5):
            patterns.append({
                "name": "Piercing Line",
                "signal": "BUY",
                "strength": 6,
                "description": "Bullish reversal pattern"
            })
        
        # === BEARISH PATTERNS ===
        
        # Shooting Star: small body, long upper wick
        if upper_wick > body_last * 2 and lower_wick < body_last and body_ratio_last > 0.1:
            patterns.append({
                "name": "Shooting Star",
                "signal": "SELL",
                "strength": 7,
                "description": "Potential reversal from uptrend"
            })
        
        # Bearish Engulfing
        if (last["close"] < last["open"] and prev["close"] > prev["open"] and
                last["open"] > prev["close"] and last["close"] < prev["open"]):
            patterns.append({
                "name": "Bearish Engulfing",
                "signal": "SELL",
                "strength": 8,
                "description": "Strong reversal signal"
            })
        
        # Three Black Crows
        if (len(df) >= 3 and 
            last["close"] < last["open"] and 
            prev["close"] < prev["open"] and 
            prev2["close"] < prev2["open"] and
            last["close"] < prev["close"] and 
            prev["close"] < prev2["close"]):
            patterns.append({
                "name": "Three Black Crows",
                "signal": "SELL",
                "strength": 9,
                "description": "Strong bearish continuation"
            })
        
        # Dark Cloud Cover
        if (last["close"] < last["open"] and 
            prev["close"] > prev["open"] and
            last["open"] > prev["close"] and 
            last["close"] < prev["open"] * 0.5):
            patterns.append({
                "name": "Dark Cloud Cover",
                "signal": "SELL",
                "strength": 6,
                "description": "Bearish reversal pattern"
            })
        
        return patterns
    
    @staticmethod
    def calculate_signal_duration(pair: str, current_confidence: float, timeframe: str = "5m") -> Dict[str, any]:
        """
        Calculate how long a signal should be valid based on the timeframe.
        """
        if timeframe.endswith("m"):
            minutes = int(timeframe[:-1])
            validity = f"{minutes} minutes"
        elif timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            minutes = hours * 60
            validity = f"{hours} hour{'s' if hours > 1 else ''}"
        elif timeframe.endswith("d"):
            days = int(timeframe[:-1])
            minutes = days * 1440
            validity = f"{days} day{'s' if days > 1 else ''}"
        else:
            # Fallback based on confidence
            if current_confidence >= 80:
                minutes = 240
                validity = "4 hours"
            elif current_confidence >= 70:
                minutes = 120
                validity = "2 hours"
            elif current_confidence >= 60:
                minutes = 60
                validity = "1 hour"
            else:
                minutes = 30
                validity = "30 minutes"
        
        return {
            "minutes": minutes,
            "validity": validity,
            "expires_at_minute": minutes // 5 if minutes > 0 else 0
        }
    
    @staticmethod
    def moving_average_convergence(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate if fast MA is above slow MA (bullish) or below (bearish)
        """
        if len(df) < 50:
            return {"signal": "NEUTRAL", "fast_ma": 0, "slow_ma": 0, "diff": 0}
        
        close = df["close"]
        fast_ma = close.ewm(span=12, adjust=False).mean()
        slow_ma = close.ewm(span=26, adjust=False).mean()
        
        fast_val = float(fast_ma.iloc[-1])
        slow_val = float(slow_ma.iloc[-1])
        diff = fast_val - slow_val
        
        if diff > 0:
            signal = "BULLISH"
        elif diff < 0:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "signal": signal,
            "fast_ma": fast_val,
            "slow_ma": slow_val,
            "diff": diff,
            "ma_crossover": "GOLDEN" if diff > 0 else "DEATH"
        }
    
    @staticmethod
    def bollinger_band_position(df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """
        Calculate where price is within Bollinger Bands.
        Returns position as % (0=lower band, 50=middle, 100=upper band)
        """
        if len(df) < period:
            return {"position": 50.0, "signal": "NEUTRAL", "squeeze": False}
        
        close = df["close"].tail(period)
        sma = close.mean()
        std = close.std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        current = close.iloc[-1]
        
        if upper == lower:
            position = 50.0
        else:
            position = ((current - lower) / (upper - lower)) * 100
            position = min(max(position, 0), 100)
        
        # Bollinger Squeeze (bands are very narrow)
        band_width = upper - lower
        avg_price = (upper + lower) / 2
        squeeze = (band_width / avg_price) < 0.01  # Less than 1% width
        
        if position > 80:
            signal = "OVERBOUGHT"
        elif position < 20:
            signal = "OVERSOLD"
        else:
            signal = "NEUTRAL"
        
        return {
            "position": round(position, 2),
            "signal": signal,
            "squeeze": squeeze,
            "upper_band": round(upper, 6),
            "lower_band": round(lower, 6),
            "middle_band": round(sma, 6)
        }
