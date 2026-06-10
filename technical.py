"""
Manual implementation of all technical indicators.
No TA-lib dependency — pure NumPy/Pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ── Moving Averages ────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def dema(series: pd.Series, period: int) -> pd.Series:
    e = ema(series, period)
    return 2 * e - ema(e, period)


def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def hma(series: pd.Series, period: int) -> pd.Series:
    half = wma(series, period // 2)
    full = wma(series, period)
    raw = 2 * half - full
    return wma(raw, int(np.sqrt(period)))


def vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    return (close * volume).rolling(period).sum() / volume.rolling(period).sum()


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3
               ) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = sma(k, d_period)
    return k, d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest + 1e-10)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series - series.shift(period)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = sma(tp, period)
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    pos = raw_mf.where(tp > tp.shift(1), 0)
    neg = raw_mf.where(tp < tp.shift(1), 0)
    pmf = pos.rolling(period).sum()
    nmf = neg.rolling(period).sum()
    mfr = pmf / (nmf + 1e-10)
    return 100 - (100 / (1 + mfr))


# ── Volatility ────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0
                    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20, multiplier: float = 1.5
                     ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(close, period)
    a = atr(high, low, close, period)
    return mid + multiplier * a, mid, mid - multiplier * a


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20
                      ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    mid = (upper + lower) / 2
    return upper, mid, lower


# ── Trend ─────────────────────────────────────────────────────────────────────

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
        ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns ADX, +DI, -DI"""
    tr = atr(high, low, close, 1)  # raw TR
    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    smooth_tr = tr.ewm(com=period - 1, adjust=False).mean()
    smooth_pdm = pos_dm.ewm(com=period - 1, adjust=False).mean()
    smooth_ndm = neg_dm.ewm(com=period - 1, adjust=False).mean()
    pdi = 100 * smooth_pdm / (smooth_tr + 1e-10)
    ndi = 100 * smooth_ndm / (smooth_tr + 1e-10)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10)
    adx_line = dx.ewm(com=period - 1, adjust=False).mean()
    return adx_line, pdi, ndi


def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                  step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    length = len(close)
    sar = np.full(length, np.nan)
    bull = True
    af = step
    ep = low.iloc[0]
    hp = high.iloc[0]
    lp = low.iloc[0]
    sar[0] = lp

    for i in range(1, length):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (hp - prev_sar)
            sar[i] = min(sar[i], low.iloc[i - 1], low.iloc[max(i - 2, 0)])
            if low.iloc[i] < sar[i]:
                bull = False
                sar[i] = hp
                lp = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            sar[i] = prev_sar + af * (lp - prev_sar)
            sar[i] = max(sar[i], high.iloc[i - 1], high.iloc[max(i - 2, 0)])
            if high.iloc[i] > sar[i]:
                bull = True
                sar[i] = lp
                hp = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + step, max_step)
    return pd.Series(sar, index=close.index)


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    a = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * a
    lower_band = hl2 - multiplier * a
    supertrend_arr = np.full(len(close), np.nan)
    direction = np.full(len(close), 1)

    for i in range(1, len(close)):
        prev_upper = upper_band.iloc[i - 1]
        prev_lower = lower_band.iloc[i - 1]
        if close.iloc[i - 1] > prev_upper:
            lower_band.iloc[i] = max(lower_band.iloc[i], prev_lower)
        if close.iloc[i - 1] < prev_lower:
            upper_band.iloc[i] = min(upper_band.iloc[i], prev_upper)
        if close.iloc[i] <= lower_band.iloc[i]:
            direction[i] = -1
            supertrend_arr[i] = upper_band.iloc[i]
        else:
            direction[i] = 1
            supertrend_arr[i] = lower_band.iloc[i]

    return pd.Series(supertrend_arr, index=close.index), pd.Series(direction, index=close.index)


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, displacement: int = 26
             ) -> dict:
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(displacement)
    chikou_span = close.shift(-displacement)
    return {
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a": senkou_span_a,
        "senkou_b": senkou_span_b,
        "chikou": chikou_span,
    }


def pivot_points(high: float, low: float, close: float) -> dict:
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


# ── Volume ────────────────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 20) -> pd.Series:
    mf_mult = ((close - low) - (high - close)) / (high - low + 1e-10)
    mf_vol = mf_mult * volume
    return mf_vol.rolling(period).sum() / volume.rolling(period).sum()


# ── Trend Strength ────────────────────────────────────────────────────────────

def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series]:
    aroon_up = high.rolling(period + 1).apply(
        lambda x: (np.argmax(x) / period) * 100, raw=True
    )
    aroon_dn = low.rolling(period + 1).apply(
        lambda x: (np.argmin(x) / period) * 100, raw=True
    )
    return aroon_up, aroon_dn


def fractals(high: pd.Series, low: pd.Series, periods: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Williams fractals — fractal up/down signals."""
    n = len(high)
    up = pd.Series(np.nan, index=high.index)
    dn = pd.Series(np.nan, index=low.index)
    for i in range(periods, n - periods):
        h_window = high.iloc[i - periods: i + periods + 1]
        l_window = low.iloc[i - periods: i + periods + 1]
        if high.iloc[i] == h_window.max():
            up.iloc[i] = high.iloc[i]
        if low.iloc[i] == l_window.min():
            dn.iloc[i] = low.iloc[i]
    return up, dn


def zigzag(close: pd.Series, pct_change: float = 0.05) -> pd.Series:
    """ZigZag indicator — marks pivot points at given % deviation."""
    result = pd.Series(np.nan, index=close.index)
    last_pivot_idx = 0
    last_pivot_val = close.iloc[0]
    trend = 1  # 1 = up, -1 = down

    for i in range(1, len(close)):
        val = close.iloc[i]
        if trend == 1:
            if val > last_pivot_val:
                last_pivot_val = val
                last_pivot_idx = i
            elif (last_pivot_val - val) / last_pivot_val >= pct_change:
                result.iloc[last_pivot_idx] = last_pivot_val
                trend = -1
                last_pivot_val = val
                last_pivot_idx = i
        else:
            if val < last_pivot_val:
                last_pivot_val = val
                last_pivot_idx = i
            elif (val - last_pivot_val) / last_pivot_val >= pct_change:
                result.iloc[last_pivot_idx] = last_pivot_val
                trend = 1
                last_pivot_val = val
                last_pivot_idx = i
    result.iloc[last_pivot_idx] = last_pivot_val
    return result


# ── Candlestick Patterns ──────────────────────────────────────────────────────

def candlestick_patterns(open_: pd.Series, high: pd.Series,
                         low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Detect all candlestick patterns. Returns DataFrame with boolean columns."""
    body = (close - open_).abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    avg_body = body.rolling(10).mean()

    patterns: dict = {}

    # Bullish / Bearish Engulfing
    patterns["bullish_engulfing"] = (
        (close.shift(1) < open_.shift(1)) &
        (close > open_) &
        (open_ < close.shift(1)) &
        (close > open_.shift(1))
    )
    patterns["bearish_engulfing"] = (
        (close.shift(1) > open_.shift(1)) &
        (close < open_) &
        (open_ > close.shift(1)) &
        (close < open_.shift(1))
    )

    # Hammer / Inverted Hammer
    patterns["hammer"] = (
        (lower_wick >= 2 * body) &
        (upper_wick <= 0.3 * body) &
        (close > open_) &
        (body > 0.001 * close)
    )
    patterns["inverted_hammer"] = (
        (upper_wick >= 2 * body) &
        (lower_wick <= 0.3 * body) &
        (body > 0.001 * close)
    )

    # Doji variants
    doji_cond = body < 0.1 * avg_body
    patterns["doji"] = doji_cond
    patterns["dragonfly_doji"] = doji_cond & (lower_wick > 2 * upper_wick)
    patterns["gravestone_doji"] = doji_cond & (upper_wick > 2 * lower_wick)

    # Morning / Evening Star
    patterns["morning_star"] = (
        (close.shift(2) < open_.shift(2)) &
        (body.shift(1) < 0.3 * avg_body.shift(1)) &
        (close > open_) &
        (close > (open_.shift(2) + close.shift(2)) / 2)
    )
    patterns["evening_star"] = (
        (close.shift(2) > open_.shift(2)) &
        (body.shift(1) < 0.3 * avg_body.shift(1)) &
        (close < open_) &
        (close < (open_.shift(2) + close.shift(2)) / 2)
    )

    # Three White Soldiers / Three Black Crows
    patterns["three_white_soldiers"] = (
        (close > open_) & (close.shift(1) > open_.shift(1)) & (close.shift(2) > open_.shift(2)) &
        (open_ > open_.shift(1)) & (open_.shift(1) > open_.shift(2)) &
        (close > close.shift(1)) & (close.shift(1) > close.shift(2))
    )
    patterns["three_black_crows"] = (
        (close < open_) & (close.shift(1) < open_.shift(1)) & (close.shift(2) < open_.shift(2)) &
        (open_ < open_.shift(1)) & (open_.shift(1) < open_.shift(2)) &
        (close < close.shift(1)) & (close.shift(1) < close.shift(2))
    )

    # Harami
    patterns["bullish_harami"] = (
        (close.shift(1) < open_.shift(1)) &
        (close > open_) &
        (open_ > close.shift(1)) &
        (close < open_.shift(1))
    )
    patterns["bearish_harami"] = (
        (close.shift(1) > open_.shift(1)) &
        (close < open_) &
        (open_ < close.shift(1)) &
        (close > open_.shift(1))
    )

    # Piercing Line / Dark Cloud Cover
    patterns["piercing_line"] = (
        (close.shift(1) < open_.shift(1)) &
        (close > open_) &
        (open_ < low.shift(1)) &
        (close > (open_.shift(1) + close.shift(1)) / 2) &
        (close < open_.shift(1))
    )
    patterns["dark_cloud_cover"] = (
        (close.shift(1) > open_.shift(1)) &
        (close < open_) &
        (open_ > high.shift(1)) &
        (close < (open_.shift(1) + close.shift(1)) / 2) &
        (close > open_.shift(1))
    )

    # Pin Bars
    patterns["pin_bar_bullish"] = (lower_wick >= 2.5 * body) & (upper_wick < body)
    patterns["pin_bar_bearish"] = (upper_wick >= 2.5 * body) & (lower_wick < body)

    return pd.DataFrame(patterns).fillna(False).astype(bool)


# ── Price Action Analysis ─────────────────────────────────────────────────────

class PriceActionAnalyzer:
    """Detects support, resistance, BOS, CHoCH, supply/demand, order blocks, FVG."""

    @staticmethod
    def support_resistance(high: pd.Series, low: pd.Series, close: pd.Series,
                           lookback: int = 20, tolerance: float = 0.002
                           ) -> Tuple[list, list]:
        """Returns (supports, resistances) as price levels."""
        pivot_highs = []
        pivot_lows = []
        for i in range(lookback, len(close) - lookback):
            if high.iloc[i] == high.iloc[i - lookback:i + lookback + 1].max():
                pivot_highs.append(high.iloc[i])
            if low.iloc[i] == low.iloc[i - lookback:i + lookback + 1].min():
                pivot_lows.append(low.iloc[i])

        # Cluster nearby levels
        def cluster(levels):
            if not levels:
                return []
            levels = sorted(levels)
            clustered = [levels[0]]
            for lvl in levels[1:]:
                if abs(lvl - clustered[-1]) / clustered[-1] > tolerance:
                    clustered.append(lvl)
                else:
                    clustered[-1] = (clustered[-1] + lvl) / 2
            return clustered

        return cluster(pivot_lows), cluster(pivot_highs)

    @staticmethod
    def fair_value_gaps(open_: pd.Series, high: pd.Series, low: pd.Series,
                        close: pd.Series) -> pd.DataFrame:
        """Identifies Fair Value Gaps (imbalances) in price."""
        gaps = []
        for i in range(2, len(close)):
            # Bullish FVG: low[i] > high[i-2]
            if low.iloc[i] > high.iloc[i - 2]:
                gaps.append({
                    "idx": i, "type": "bullish",
                    "top": low.iloc[i], "bottom": high.iloc[i - 2],
                    "price": (low.iloc[i] + high.iloc[i - 2]) / 2,
                })
            # Bearish FVG: high[i] < low[i-2]
            if high.iloc[i] < low.iloc[i - 2]:
                gaps.append({
                    "idx": i, "type": "bearish",
                    "top": low.iloc[i - 2], "bottom": high.iloc[i],
                    "price": (low.iloc[i - 2] + high.iloc[i]) / 2,
                })
        return pd.DataFrame(gaps)

    @staticmethod
    def order_blocks(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, lookforward: int = 5) -> pd.DataFrame:
        """Identifies order blocks — last opposing candle before a strong move."""
        blocks = []
        for i in range(1, len(close) - lookforward):
            # Bullish order block: bearish candle before bullish expansion
            if close.iloc[i] < open_.iloc[i]:  # bearish candle
                future_close = close.iloc[i + 1:i + lookforward + 1]
                if future_close.max() > high.iloc[i] * 1.002:
                    blocks.append({
                        "idx": i, "type": "bullish_ob",
                        "top": open_.iloc[i], "bottom": close.iloc[i],
                    })
            # Bearish order block: bullish candle before bearish expansion
            if close.iloc[i] > open_.iloc[i]:  # bullish candle
                future_close = close.iloc[i + 1:i + lookforward + 1]
                if future_close.min() < low.iloc[i] * 0.998:
                    blocks.append({
                        "idx": i, "type": "bearish_ob",
                        "top": close.iloc[i], "bottom": open_.iloc[i],
                    })
        return pd.DataFrame(blocks)

    @staticmethod
    def market_structure(close: pd.Series, swing_size: int = 5) -> pd.Series:
        """Detect BOS (Break of Structure) and CHoCH (Change of Character)."""
        labels = pd.Series("neutral", index=close.index)
        highs = []
        lows = []
        for i in range(swing_size, len(close) - swing_size):
            window = close.iloc[i - swing_size:i + swing_size + 1]
            if close.iloc[i] == window.max():
                highs.append((i, close.iloc[i]))
            if close.iloc[i] == window.min():
                lows.append((i, close.iloc[i]))

        # BOS: price breaks previous structure high/low
        for j in range(1, len(highs)):
            if highs[j][1] > highs[j - 1][1]:
                labels.iloc[highs[j][0]] = "BOS_UP"
            elif highs[j][1] < highs[j - 1][1]:
                labels.iloc[highs[j][0]] = "CHoCH_DOWN"
        for j in range(1, len(lows)):
            if lows[j][1] < lows[j - 1][1]:
                labels.iloc[lows[j][0]] = "BOS_DOWN"
            elif lows[j][1] > lows[j - 1][1]:
                labels.iloc[lows[j][0]] = "CHoCH_UP"
        return labels

    @staticmethod
    def liquidity_sweep(high: pd.Series, low: pd.Series, close: pd.Series,
                        lookback: int = 20) -> pd.Series:
        """Detects liquidity grabs / stop hunts above/below recent highs/lows."""
        signals = pd.Series("none", index=close.index)
        for i in range(lookback, len(close)):
            prev_high = high.iloc[i - lookback:i].max()
            prev_low = low.iloc[i - lookback:i].min()
            # Wick above prior high then close below — bearish sweep
            if high.iloc[i] > prev_high and close.iloc[i] < prev_high:
                signals.iloc[i] = "bearish_sweep"
            # Wick below prior low then close above — bullish sweep
            if low.iloc[i] < prev_low and close.iloc[i] > prev_low:
                signals.iloc[i] = "bullish_sweep"
        return signals


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_feature_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from OHLCV data.
    Input df must have: open, high, low, close, volume columns.
    Returns df with all indicator columns appended.
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # MAs
    df["ema_8"] = ema(c, 8)
    df["ema_13"] = ema(c, 13)
    df["ema_21"] = ema(c, 21)
    df["ema_50"] = ema(c, 50)
    df["ema_200"] = ema(c, 200)
    df["sma_20"] = sma(c, 20)
    df["sma_50"] = sma(c, 50)
    df["wma_14"] = wma(c, 14)
    df["hma_14"] = hma(c, 14)
    df["dema_14"] = dema(c, 14)
    df["tema_14"] = tema(c, 14)
    df["vwma_20"] = vwma(c, v, 20)

    # Momentum
    df["rsi_14"] = rsi(c, 14)
    df["rsi_7"] = rsi(c, 7)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["stoch_k"], df["stoch_d"] = stochastic(h, l, c)
    df["cci_20"] = cci(h, l, c, 20)
    df["mfi_14"] = mfi(h, l, c, v, 14)
    df["williams_r"] = williams_r(h, l, c)
    df["roc_12"] = roc(c, 12)
    df["momentum_10"] = momentum(c, 10)

    # Volatility
    df["atr_14"] = atr(h, l, c, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(c)
    df["kc_upper"], df["kc_mid"], df["kc_lower"] = keltner_channels(h, l, c)
    df["dc_upper"], df["dc_mid"], df["dc_lower"] = donchian_channels(h, l)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-10)
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # Trend
    df["adx"], df["pdi"], df["ndi"] = adx(h, l, c)
    df["sar"] = parabolic_sar(h, l, c)
    df["supertrend"], df["supertrend_dir"] = supertrend(h, l, c)
    ich = ichimoku(h, l, c)
    df["ich_tenkan"] = ich["tenkan"]
    df["ich_kijun"] = ich["kijun"]
    df["ich_span_a"] = ich["senkou_a"]
    df["ich_span_b"] = ich["senkou_b"]

    # Volume
    df["obv"] = obv(c, v)
    df["cmf"] = cmf(h, l, c, v)
    df["vwap"] = vwap(h, l, c, v)
    df["aroon_up"], df["aroon_dn"] = aroon(h, l)

    # Derived / normalised features
    df["close_vs_ema21"] = (c - df["ema_21"]) / (df["ema_21"] + 1e-10)
    df["close_vs_ema50"] = (c - df["ema_50"]) / (df["ema_50"] + 1e-10)
    df["ema_trend"] = (df["ema_21"] - df["ema_50"]) / (df["ema_50"] + 1e-10)
    df["price_vs_vwap"] = (c - df["vwap"]) / (df["vwap"] + 1e-10)
    df["atr_pct"] = df["atr_14"] / (c + 1e-10)
    df["body_pct"] = (c - o).abs() / (c + 1e-10)
    df["upper_wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (c + 1e-10)
    df["lower_wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (c + 1e-10)
    df["volume_ma_ratio"] = v / (v.rolling(20).mean() + 1e-10)
    df["candle_direction"] = np.where(c >= o, 1, -1)
    df["ema_alignment"] = np.sign(
        (df["ema_8"] - df["ema_13"]) *
        (df["ema_13"] - df["ema_21"]) *
        (df["ema_21"] - df["ema_50"])
    )

    # Candlestick patterns
    cp = candlestick_patterns(o, h, l, c)
    for col in cp.columns:
        df[f"pat_{col}"] = cp[col].astype(int)

    return df


FEATURE_COLUMNS = [
    "rsi_14","rsi_7","macd","macd_signal","macd_hist","stoch_k","stoch_d",
    "cci_20","mfi_14","williams_r","roc_12","momentum_10",
    "adx","pdi","ndi","atr_14","bb_width","bb_pct",
    "close_vs_ema21","close_vs_ema50","ema_trend","price_vs_vwap",
    "atr_pct","body_pct","upper_wick_pct","lower_wick_pct","volume_ma_ratio",
    "candle_direction","ema_alignment","obv","cmf","aroon_up","aroon_dn",
    "supertrend_dir",
    "pat_bullish_engulfing","pat_bearish_engulfing","pat_hammer",
    "pat_inverted_hammer","pat_morning_star","pat_evening_star",
    "pat_doji","pat_dragonfly_doji","pat_gravestone_doji",
    "pat_three_white_soldiers","pat_three_black_crows",
    "pat_bullish_harami","pat_bearish_harami","pat_piercing_line",
    "pat_dark_cloud_cover","pat_pin_bar_bullish","pat_pin_bar_bearish",
]
