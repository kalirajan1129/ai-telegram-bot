import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class IndicatorSignal:
    name: str
    score: float        # -100 (strong sell) to +100 (strong buy)
    signal: str         # "BUY", "SELL", "NEUTRAL"
    strength: str       # "STRONG", "MEDIUM", "WEAK"
    weight: float = 1.0 # indicator importance weight
    details: Dict = field(default_factory=dict)


class TechnicalIndicators:

    # ── RSI ───────────────────────────────────────────────────────────────────
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14) -> Optional[float]:
        if len(data) < period + 1:
            return None
        close = data["close"].values.astype(float)
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    @staticmethod
    def macd(data: pd.DataFrame, fast=12, slow=26, signal=9):
        if len(data) < slow + signal:
            return None, None, None
        close = pd.Series(data["close"].values.astype(float))
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_f - ema_s
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - sig_line
        return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float(hist.iloc[-1])

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period=20, std_dev=2):
        if len(data) < period:
            return None, None, None
        close = pd.Series(data["close"].values.astype(float))
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])

    # ── Stochastic Oscillator ─────────────────────────────────────────────────
    @staticmethod
    def stochastic(data: pd.DataFrame, k_period=14, d_period=3):
        if len(data) < k_period + d_period:
            return None, None
        high = data["high"].values.astype(float)
        low  = data["low"].values.astype(float)
        close= data["close"].values.astype(float)
        k_vals = []
        for i in range(k_period - 1, len(close)):
            h = high[i - k_period + 1: i + 1].max()
            l = low[i  - k_period + 1: i + 1].min()
            k_vals.append(100 * (close[i] - l) / (h - l) if h != l else 50)
        k = pd.Series(k_vals)
        d = k.rolling(d_period).mean()
        return float(k.iloc[-1]), float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else None

    # ── CCI ───────────────────────────────────────────────────────────────────
    @staticmethod
    def cci(data: pd.DataFrame, period=20):
        if len(data) < period:
            return None
        tp = (data["high"] + data["low"] + data["close"]) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci_val = (tp - sma) / (0.015 * mad)
        return float(cci_val.iloc[-1]) if not pd.isna(cci_val.iloc[-1]) else None

    # ── Williams %R ───────────────────────────────────────────────────────────
    @staticmethod
    def williams_r(data: pd.DataFrame, period=14):
        if len(data) < period:
            return None
        h = data["high"].tail(period).max()
        l = data["low"].tail(period).min()
        c = float(data["close"].iloc[-1])
        if h == l:
            return -50.0
        return -100 * (h - c) / (h - l)

    # ── ATR ───────────────────────────────────────────────────────────────────
    @staticmethod
    def atr(data: pd.DataFrame, period=14):
        if len(data) < period + 1:
            return None
        h = data["high"].values.astype(float)
        l = data["low"].values.astype(float)
        c = data["close"].values.astype(float)
        tr = np.maximum(h[1:] - l[1:],
             np.maximum(np.abs(h[1:] - c[:-1]),
                        np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))

    # ── ADX ───────────────────────────────────────────────────────────────────
    @staticmethod
    def adx(data: pd.DataFrame, period=14):
        if len(data) < period * 2:
            return None
        h = data["high"].values.astype(float)
        l = data["low"].values.astype(float)
        c = data["close"].values.astype(float)
        pdm = np.zeros(len(h))
        ndm = np.zeros(len(h))
        for i in range(1, len(h)):
            up = h[i] - h[i-1]
            dn = l[i-1] - l[i]
            pdm[i] = up if up > dn and up > 0 else 0
            ndm[i] = dn if dn > up and dn > 0 else 0
        tr_arr = np.zeros(len(h))
        tr_arr[0] = h[0] - l[0]
        for i in range(1, len(h)):
            tr_arr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr_v = np.mean(tr_arr[-period:])
        if atr_v == 0:
            return 0.0
        pdi = 100 * np.mean(pdm[-period:]) / atr_v
        ndi = 100 * np.mean(ndm[-period:]) / atr_v
        denom = pdi + ndi
        if denom == 0:
            return 0.0
        dx = 100 * abs(pdi - ndi) / denom
        return min(float(dx), 100.0)

    # ── EMA ───────────────────────────────────────────────────────────────────
    @staticmethod
    def ema(data: pd.DataFrame, period=20):
        if len(data) < period:
            return None
        return float(pd.Series(data["close"].values.astype(float)).ewm(span=period, adjust=False).mean().iloc[-1])

    # ── Support/Resistance (pivot) ────────────────────────────────────────────
    @staticmethod
    def support_resistance(data: pd.DataFrame, lookback=20):
        if len(data) < lookback:
            return {}
        rec = data.tail(lookback)
        h = float(rec["high"].max())
        l = float(rec["low"].min())
        c = float(rec["close"].iloc[-1])
        pivot = (h + l + c) / 3
        return {
            "support":      2 * pivot - h,
            "resistance":   2 * pivot - l,
            "pivot":        pivot,
            "support_2":    pivot - (h - l),
            "resistance_2": pivot + (h - l),
        }

    # ── Fibonacci Levels ──────────────────────────────────────────────────────
    @staticmethod
    def fibonacci_levels(data: pd.DataFrame, lookback=50):
        if len(data) < lookback:
            return {}
        rec  = data.tail(lookback)
        high = float(rec["high"].max())
        low  = float(rec["low"].min())
        diff = high - low
        return {
            "0%":    high,
            "23.6%": high - diff * 0.236,
            "38.2%": high - diff * 0.382,
            "50%":   high - diff * 0.500,
            "61.8%": high - diff * 0.618,
            "78.6%": high - diff * 0.786,
            "100%":  low,
        }

    # ── Fair Value Gap (FVG) ──────────────────────────────────────────────────
    @staticmethod
    def fair_value_gap(data: pd.DataFrame):
        """
        Bullish FVG: candle[i-2].high < candle[i].low  (gap up — price should revisit)
        Bearish FVG: candle[i-2].low  > candle[i].high (gap down — price should revisit)
        Returns most recent FVG if present within last 10 candles.
        """
        if len(data) < 3:
            return None
        results = []
        for i in range(len(data) - 1, max(len(data) - 11, 1), -1):
            c0 = data.iloc[i - 2]
            c2 = data.iloc[i]
            # Bullish FVG
            if c0["high"] < c2["low"]:
                gap_size = c2["low"] - c0["high"]
                results.append({
                    "type":     "BULLISH",
                    "gap_high": float(c2["low"]),
                    "gap_low":  float(c0["high"]),
                    "size":     float(gap_size),
                    "age":      len(data) - 1 - i,
                })
                break
            # Bearish FVG
            if c0["low"] > c2["high"]:
                gap_size = c0["low"] - c2["high"]
                results.append({
                    "type":     "BEARISH",
                    "gap_high": float(c0["low"]),
                    "gap_low":  float(c2["high"]),
                    "size":     float(gap_size),
                    "age":      len(data) - 1 - i,
                })
                break
        return results[0] if results else None

    # ── Order Block ───────────────────────────────────────────────────────────
    @staticmethod
    def order_block(data: pd.DataFrame, lookback=20):
        """
        Bullish OB: last significant bearish candle before a strong bullish move
        Bearish OB: last significant bullish candle before a strong bearish move
        """
        if len(data) < 5:
            return None
        check = data.tail(lookback).reset_index(drop=True)
        for i in range(len(check) - 2, 1, -1):
            c = check.iloc[i]
            nxt = check.iloc[i + 1]
            body = abs(c["close"] - c["open"])
            total = c["high"] - c["low"]
            if total == 0:
                continue
            # Bullish OB: bearish candle followed by strong bullish momentum
            if c["close"] < c["open"] and nxt["close"] > nxt["open"]:
                if (nxt["close"] - nxt["open"]) > body * 1.5:
                    return {
                        "type":  "BULLISH",
                        "high":  float(c["high"]),
                        "low":   float(c["low"]),
                        "mid":   float((c["high"] + c["low"]) / 2),
                        "index": i,
                    }
            # Bearish OB: bullish candle followed by strong bearish momentum
            if c["close"] > c["open"] and nxt["close"] < nxt["open"]:
                if (nxt["open"] - nxt["close"]) > body * 1.5:
                    return {
                        "type":  "BEARISH",
                        "high":  float(c["high"]),
                        "low":   float(c["low"]),
                        "mid":   float((c["high"] + c["low"]) / 2),
                        "index": i,
                    }
        return None

    # ── Trend Line (linear regression slope) ─────────────────────────────────
    @staticmethod
    def trend_line(data: pd.DataFrame, period=20):
        if len(data) < period:
            return {"slope": 0.0, "direction": "NEUTRAL", "angle": 0.0}
        close = data["close"].tail(period).values.astype(float)
        x = np.arange(len(close))
        slope, _ = np.polyfit(x, close, 1)
        # Normalise slope as % of price
        slope_pct = slope / close.mean() * 100 if close.mean() != 0 else 0
        direction = "UP" if slope_pct > 0.02 else "DOWN" if slope_pct < -0.02 else "FLAT"
        return {"slope": slope_pct, "direction": direction}

    # ── Candlestick Patterns ──────────────────────────────────────────────────
    @staticmethod
    def candlestick_patterns(data: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(data) < 4:
            return patterns
        c1 = data.iloc[-1]
        c2 = data.iloc[-2]
        c3 = data.iloc[-3]
        c4 = data.iloc[-4]

        body1  = abs(c1["close"] - c1["open"])
        body2  = abs(c2["close"] - c2["open"])
        body3  = abs(c3["close"] - c3["open"])
        total1 = c1["high"] - c1["low"] or 1e-10
        upper1 = c1["high"] - max(c1["open"], c1["close"])
        lower1 = min(c1["open"], c1["close"]) - c1["low"]

        def add(name, sig, strength, score):
            patterns.append({"name": name, "signal": sig, "strength": strength, "score": score})

        # ── Bullish ──
        if lower1 > body1 * 2 and upper1 < body1 * 0.5:
            add("Hammer", "BUY", "STRONG", 80)
        if lower1 > body1 * 2 and upper1 < body1 * 0.5 and c1["close"] < c2["low"]:
            add("Inverted Hammer (reversal)", "BUY", "STRONG", 82)
        if (c1["close"] > c1["open"] and c2["close"] < c2["open"] and
                c1["open"] < c2["close"] and c1["close"] > c2["open"]):
            add("Bullish Engulfing", "BUY", "STRONG", 85)
        if (c1["close"] > c1["open"] and c2["close"] < c2["open"] and
                c3["close"] < c3["open"] and c1["close"] > c3["close"] * 0.5):
            add("Morning Star", "BUY", "STRONG", 87)
        if (c1["close"] > c1["open"] and c2["close"] > c2["open"] and
                c3["close"] > c3["open"] and c1["close"] > c2["close"] > c3["close"]):
            add("Three White Soldiers", "BUY", "STRONG", 90)
        if (c1["close"] > c1["open"] and c2["close"] < c2["open"] and
                c1["open"] < c2["close"] and c1["close"] > (c2["open"] + c2["close"]) / 2):
            add("Piercing Line", "BUY", "MEDIUM", 72)
        # Doji at support
        if body1 / total1 < 0.1:
            add("Doji", "NEUTRAL", "WEAK", 50)

        # ── Bearish ──
        if upper1 > body1 * 2 and lower1 < body1 * 0.5:
            add("Shooting Star", "SELL", "STRONG", 80)
        if (c1["close"] < c1["open"] and c2["close"] > c2["open"] and
                c1["open"] > c2["close"] and c1["close"] < c2["open"]):
            add("Bearish Engulfing", "SELL", "STRONG", 85)
        if (c1["close"] < c1["open"] and c2["close"] > c2["open"] and
                c3["close"] > c3["open"] and c1["close"] < c3["close"] * 0.5):
            add("Evening Star", "SELL", "STRONG", 87)
        if (c1["close"] < c1["open"] and c2["close"] < c2["open"] and
                c3["close"] < c3["open"] and c1["close"] < c2["close"] < c3["close"]):
            add("Three Black Crows", "SELL", "STRONG", 90)
        if (c1["close"] < c1["open"] and c2["close"] > c2["open"] and
                c1["open"] > c2["close"] and c1["close"] < (c2["open"] + c2["close"]) / 2):
            add("Dark Cloud Cover", "SELL", "MEDIUM", 72)

        return patterns


# ─────────────────────────────────────────────────────────────────────────────
# SignalEngine — multi-indicator weighted confluence
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Generates high-accuracy signals using weighted multi-indicator confluence.
    A signal is only emitted when ≥3 strong indicators agree with confidence ≥70%.
    """

    # Indicator weights (higher = more important)
    WEIGHTS = {
        "RSI":             1.5,
        "MACD":            1.5,
        "Stochastic":      1.2,
        "CCI":             1.0,
        "Williams_R":      1.0,
        "Bollinger":       1.2,
        "EMA_Cross":       2.0,   # highest weight — trend confirmation
        "ADX_Trend":       1.8,
        "FVG":             1.6,
        "Order_Block":     1.6,
        "Trend_Line":      1.3,
        "Fibonacci":       1.2,
        "Support_Resist":  1.2,
        "Candlestick":     1.4,
        "Volume":          1.0,
    }

    def __init__(self, buy_threshold=70.0, sell_threshold=70.0):
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold
        self.ind = TechnicalIndicators()

    def generate_signal(self, data: pd.DataFrame, pair: str, timeframe: str) -> Optional[Dict]:
        if len(data) < 50:
            return None
        signals = self._collect_all(data)
        if not signals:
            return None

        buy_sigs  = [s for s in signals if s.signal == "BUY"]
        sell_sigs = [s for s in signals if s.signal == "SELL"]

        buy_score  = self._weighted_score(buy_sigs)
        sell_score = self._weighted_score(sell_sigs)

        # Require minimum 3 confirming indicators
        if len(buy_sigs) >= 3 and buy_score > sell_score * 1.2:
            final_signal = "STRONG_BUY" if buy_score >= 85 else "BUY"
            final_score  = buy_score
        elif len(sell_sigs) >= 3 and sell_score > buy_score * 1.2:
            final_signal = "STRONG_SELL" if sell_score >= 85 else "SELL"
            final_score  = sell_score
        else:
            return {
                "pair": pair, "timeframe": timeframe,
                "signal": "NEUTRAL", "score": 50.0,
                "indicators": signals, "timestamp": data.iloc[-1]["timestamp"]
            }

        # Trend filter: block counter-trend signals
        trend = self._trend_direction(data)
        if final_signal in ("BUY", "STRONG_BUY") and trend == "DOWN":
            # Reduce score by 15 for counter-trend
            final_score = max(final_score - 15, 0)
        elif final_signal in ("SELL", "STRONG_SELL") and trend == "UP":
            final_score = max(final_score - 15, 0)

        if final_score < self.buy_threshold:
            return {
                "pair": pair, "timeframe": timeframe,
                "signal": "NEUTRAL", "score": final_score,
                "indicators": signals, "timestamp": data.iloc[-1]["timestamp"]
            }

        return {
            "pair":       pair,
            "timeframe":  timeframe,
            "signal":     final_signal,
            "score":      round(final_score, 2),
            "indicators": signals,
            "timestamp":  data.iloc[-1]["timestamp"],
        }

    def _weighted_score(self, sigs: List[IndicatorSignal]) -> float:
        if not sigs:
            return 0.0
        total_w = sum(self.WEIGHTS.get(s.name, 1.0) for s in sigs)
        weighted = sum(abs(s.score) * self.WEIGHTS.get(s.name, 1.0) for s in sigs)
        return weighted / total_w if total_w > 0 else 0.0

    def _trend_direction(self, data: pd.DataFrame) -> str:
        ema20  = self.ind.ema(data, 20)
        ema50  = self.ind.ema(data, 50)
        if ema20 and ema50:
            return "UP" if ema20 > ema50 else "DOWN"
        return "NEUTRAL"

    def _collect_all(self, data: pd.DataFrame) -> List[IndicatorSignal]:
        sigs = []
        close = float(data["close"].iloc[-1])

        # ── RSI ──
        rsi_v = self.ind.rsi(data)
        if rsi_v is not None:
            if rsi_v < 25:
                sigs.append(IndicatorSignal("RSI", 85, "BUY", "STRONG", self.WEIGHTS["RSI"], {"value": rsi_v}))
            elif rsi_v < 35:
                sigs.append(IndicatorSignal("RSI", 72, "BUY", "MEDIUM", self.WEIGHTS["RSI"], {"value": rsi_v}))
            elif rsi_v > 75:
                sigs.append(IndicatorSignal("RSI", 85, "SELL", "STRONG", self.WEIGHTS["RSI"], {"value": rsi_v}))
            elif rsi_v > 65:
                sigs.append(IndicatorSignal("RSI", 72, "SELL", "MEDIUM", self.WEIGHTS["RSI"], {"value": rsi_v}))

        # ── MACD ──
        macd_l, sig_l, hist = self.ind.macd(data)
        if macd_l is not None:
            if macd_l > sig_l and hist > 0 and hist > abs(macd_l) * 0.1:
                sigs.append(IndicatorSignal("MACD", 78, "BUY", "MEDIUM", self.WEIGHTS["MACD"], {"hist": hist}))
            elif macd_l < sig_l and hist < 0 and abs(hist) > abs(macd_l) * 0.1:
                sigs.append(IndicatorSignal("MACD", 78, "SELL", "MEDIUM", self.WEIGHTS["MACD"], {"hist": hist}))

        # ── Stochastic ──
        k, d = self.ind.stochastic(data)
        if k is not None and d is not None:
            if k < 20 and d < 20 and k > d:
                sigs.append(IndicatorSignal("Stochastic", 80, "BUY", "STRONG", self.WEIGHTS["Stochastic"], {"k": k, "d": d}))
            elif k > 80 and d > 80 and k < d:
                sigs.append(IndicatorSignal("Stochastic", 80, "SELL", "STRONG", self.WEIGHTS["Stochastic"], {"k": k, "d": d}))

        # ── CCI ──
        cci_v = self.ind.cci(data)
        if cci_v is not None:
            if cci_v < -100:
                sigs.append(IndicatorSignal("CCI", 75, "BUY", "MEDIUM", self.WEIGHTS["CCI"], {"value": cci_v}))
            elif cci_v > 100:
                sigs.append(IndicatorSignal("CCI", 75, "SELL", "MEDIUM", self.WEIGHTS["CCI"], {"value": cci_v}))

        # ── Williams %R ──
        wr = self.ind.williams_r(data)
        if wr is not None:
            if wr < -80:
                sigs.append(IndicatorSignal("Williams_R", 74, "BUY", "MEDIUM", self.WEIGHTS["Williams_R"], {"value": wr}))
            elif wr > -20:
                sigs.append(IndicatorSignal("Williams_R", 74, "SELL", "MEDIUM", self.WEIGHTS["Williams_R"], {"value": wr}))

        # ── Bollinger Bands ──
        bb_u, bb_m, bb_l = self.ind.bollinger_bands(data)
        if bb_u is not None:
            if close < bb_l:
                sigs.append(IndicatorSignal("Bollinger", 78, "BUY", "STRONG", self.WEIGHTS["Bollinger"], {"position": "below_lower"}))
            elif close > bb_u:
                sigs.append(IndicatorSignal("Bollinger", 78, "SELL", "STRONG", self.WEIGHTS["Bollinger"], {"position": "above_upper"}))

        # ── EMA Cross (8/21/50) ──
        ema8  = self.ind.ema(data, 8)
        ema21 = self.ind.ema(data, 21)
        ema50 = self.ind.ema(data, 50)
        if ema8 and ema21 and ema50:
            if ema8 > ema21 > ema50:
                sigs.append(IndicatorSignal("EMA_Cross", 82, "BUY", "STRONG", self.WEIGHTS["EMA_Cross"], {"ema8": ema8, "ema21": ema21, "ema50": ema50}))
            elif ema8 < ema21 < ema50:
                sigs.append(IndicatorSignal("EMA_Cross", 82, "SELL", "STRONG", self.WEIGHTS["EMA_Cross"], {"ema8": ema8, "ema21": ema21, "ema50": ema50}))

        # ── ADX + Directional ──
        adx_v = self.ind.adx(data)
        if adx_v is not None and adx_v > 25:
            trend_dir = self._trend_direction(data)
            if trend_dir == "UP":
                sigs.append(IndicatorSignal("ADX_Trend", 76, "BUY", "STRONG", self.WEIGHTS["ADX_Trend"], {"adx": adx_v}))
            elif trend_dir == "DOWN":
                sigs.append(IndicatorSignal("ADX_Trend", 76, "SELL", "STRONG", self.WEIGHTS["ADX_Trend"], {"adx": adx_v}))

        # ── Fair Value Gap ──
        fvg = self.ind.fair_value_gap(data)
        if fvg:
            gap_mid = (fvg["gap_high"] + fvg["gap_low"]) / 2
            # Price inside or near FVG = high-probability reversal zone
            near_fvg = abs(close - gap_mid) / gap_mid < 0.005
            if fvg["type"] == "BULLISH" and (close <= fvg["gap_high"] or near_fvg):
                sigs.append(IndicatorSignal("FVG", 83, "BUY", "STRONG", self.WEIGHTS["FVG"], fvg))
            elif fvg["type"] == "BEARISH" and (close >= fvg["gap_low"] or near_fvg):
                sigs.append(IndicatorSignal("FVG", 83, "SELL", "STRONG", self.WEIGHTS["FVG"], fvg))

        # ── Order Block ──
        ob = self.ind.order_block(data)
        if ob:
            in_zone = ob["low"] <= close <= ob["high"]
            if ob["type"] == "BULLISH" and in_zone:
                sigs.append(IndicatorSignal("Order_Block", 85, "BUY", "STRONG", self.WEIGHTS["Order_Block"], ob))
            elif ob["type"] == "BEARISH" and in_zone:
                sigs.append(IndicatorSignal("Order_Block", 85, "SELL", "STRONG", self.WEIGHTS["Order_Block"], ob))

        # ── Trend Line ──
        tl = self.ind.trend_line(data)
        if tl["direction"] == "UP":
            sigs.append(IndicatorSignal("Trend_Line", 70, "BUY", "MEDIUM", self.WEIGHTS["Trend_Line"], tl))
        elif tl["direction"] == "DOWN":
            sigs.append(IndicatorSignal("Trend_Line", 70, "SELL", "MEDIUM", self.WEIGHTS["Trend_Line"], tl))

        # ── Fibonacci ──
        fib = self.ind.fibonacci_levels(data)
        if fib:
            for level_name, level_price in fib.items():
                if level_name in ("38.2%", "50%", "61.8%"):  # key levels
                    distance = abs(close - level_price) / level_price
                    if distance < 0.003:  # within 0.3% of fib level
                        # Price bouncing off key fib = reversal
                        trend_dir = self._trend_direction(data)
                        sig_dir = "BUY" if trend_dir == "UP" else "SELL"
                        sigs.append(IndicatorSignal("Fibonacci", 74, sig_dir, "MEDIUM",
                                                     self.WEIGHTS["Fibonacci"],
                                                     {"level": level_name, "price": level_price}))
                        break

        # ── Support/Resistance ──
        sr = self.ind.support_resistance(data)
        if sr:
            if close <= sr["support"] * 1.005:
                sigs.append(IndicatorSignal("Support_Resist", 72, "BUY", "MEDIUM", self.WEIGHTS["Support_Resist"], sr))
            elif close >= sr["resistance"] * 0.995:
                sigs.append(IndicatorSignal("Support_Resist", 72, "SELL", "MEDIUM", self.WEIGHTS["Support_Resist"], sr))

        # ── Candlestick Patterns ──
        patterns = self.ind.candlestick_patterns(data)
        for p in patterns:
            if p["signal"] != "NEUTRAL":
                sigs.append(IndicatorSignal("Candlestick", p["score"], p["signal"],
                                             p["strength"], self.WEIGHTS["Candlestick"],
                                             {"pattern": p["name"]}))

        # ── Volume (if available) ──
        if "volume" in data.columns:
            avg_vol = float(data["volume"].tail(20).mean())
            cur_vol = float(data["volume"].iloc[-1])
            if cur_vol > avg_vol * 1.5:
                # High volume confirms trend
                trend_dir = self._trend_direction(data)
                if trend_dir == "UP":
                    sigs.append(IndicatorSignal("Volume", 70, "BUY", "MEDIUM", self.WEIGHTS["Volume"], {"vol_ratio": cur_vol / avg_vol}))
                elif trend_dir == "DOWN":
                    sigs.append(IndicatorSignal("Volume", 70, "SELL", "MEDIUM", self.WEIGHTS["Volume"], {"vol_ratio": cur_vol / avg_vol}))

        return sigs