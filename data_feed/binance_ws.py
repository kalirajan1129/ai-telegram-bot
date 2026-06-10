"""
Binance WebSocket Data Feed — streams real-time OHLCV candles.
"""
from __future__ import annotations

import asyncio
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable, Dict, List, Optional
import pandas as pd
import websockets
from loguru import logger
from config import settings
from database import save_candles


TF_MAP: Dict[str, str] = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "4h": "4h",
}


def _parse_kline(msg: dict) -> Optional[dict]:
    try:
        k = msg["k"]
        if not k["x"]:  # only process closed candles
            return None
        return {
            "timestamp": pd.Timestamp(k["t"], unit="ms", tz="UTC"),
            "open": float(k["o"]), "high": float(k["h"]),
            "low": float(k["l"]), "close": float(k["c"]),
            "volume": float(k["v"]),
        }
    except (KeyError, ValueError):
        return None


class BinanceFeed:
    def __init__(self, pairs: List[str], timeframes: Optional[List[str]] = None, on_candle: Optional[Callable] = None):
        # FIX #14: Filter out OTC and FOREX pairs — Binance only handles crypto spot pairs.
        #          Original code only filtered OTC but passed all pairs; now explicitly
        #          keep only pairs that look like Binance crypto symbols (end with USDT/BTC/ETH/BNB).
        self.pairs = [
            p.upper() for p in pairs
            if "OTC" not in p and not _is_forex(p)
        ]
        self.timeframes = [tf for tf in (timeframes or ["5m", "15m", "1h"]) if tf in TF_MAP]
        self.on_candle = on_candle
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        self._running = True
        streams = [
            f"{pair.lower()}@kline_{TF_MAP[tf]}"
            for pair in self.pairs for tf in self.timeframes
        ]
        if not streams:
            logger.warning("BinanceFeed: no valid streams to subscribe to.")
            return
        batch_size = 200
        for i in range(0, len(streams), batch_size):
            task = asyncio.create_task(self._stream_batch(streams[i: i + batch_size]))
            self._tasks.append(task)
        logger.info("Binance feed started: {} pairs × {} timeframes", len(self.pairs), len(self.timeframes))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _stream_batch(self, streams: List[str]) -> None:
        path = "/".join(streams)
        url = f"wss://stream.binance.com:9443/stream?streams={path}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle(raw)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("WS error: {}. Reconnecting in 5s...", e)
                await asyncio.sleep(5)

    async def _handle(self, raw: str) -> None:
        try:
            outer = json.loads(raw)
            msg = outer.get("data", outer)
            if msg.get("e") != "kline":
                return
            candle = _parse_kline(msg)
            if not candle:
                return
            stream = outer.get("stream", "")
            parts = stream.split("@")
            if len(parts) < 2:
                return
            pair = parts[0].upper()
            tf_raw = parts[1].replace("kline_", "")
            tf = next((k for k, v in TF_MAP.items() if v == tf_raw), tf_raw)
            save_candles(pair, tf, pd.DataFrame([candle]))
            if self.on_candle:
                await self.on_candle(pair, tf, candle)
        except Exception as e:
            logger.debug("WS parse error: {}", e)


def _is_forex(pair: str) -> bool:
    """Return True if this looks like a Forex pair (no crypto quote currency)."""
    crypto_quotes = ("USDT", "BTC", "ETH", "BNB", "BUSD", "USDC", "TUSD")
    p = pair.upper()
    return not any(p.endswith(q) for q in crypto_quotes)


async def fetch_historical_candles(pair: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    import httpx
    # FIX #15: Skip forex/OTC pairs silently — Binance REST doesn't have them
    if _is_forex(pair) or "OTC" in pair.upper():
        logger.debug("Skipping non-Binance pair: {}", pair)
        return pd.DataFrame()

    interval = TF_MAP.get(timeframe, "5m")
    url = f"{settings.BINANCE_REST_URL}/api/v3/klines"
    params = {"symbol": pair.upper(), "interval": interval, "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()
    except Exception as e:
        logger.error("REST fetch failed for {} {}: {}", pair, timeframe, e)
        return pd.DataFrame()

    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "close_time", "qv", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    save_candles(pair, timeframe, df)
    logger.info("Fetched {} candles: {} {}", len(df), pair, timeframe)
    return df
