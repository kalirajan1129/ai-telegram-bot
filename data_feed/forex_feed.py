"""
Forex & OTC Data Feed — polls free REST APIs for Forex rates
and mirrors OTC pairs from their base Forex data.

Sources used (all free, no API key needed for basic use):
  - Twelve Data  (https://api.twelvedata.com) — 8 req/min free tier
  - Fallback: exchangerate.host — simple latest rates

OTC pairs (like EURUSD_OTC) simply mirror the real EURUSD data
because Pocket Option / Quotex OTC candles track the real pair
closely during market hours and are synthetic off-hours.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from config import settings
    from database import save_candles, load_candles
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import settings
    from database import save_candles, load_candles


# ── Twelve Data REST (free tier: 8 req/min, 800/day) ─────────────────────────
TWELVE_DATA_URL = "https://api.twelvedata.com"
TWELVE_DATA_MIN_DELAY = 7.5  # seconds between requests to respect rate limit (8 req/min)

# These are the Forex pairs we can fetch from Twelve Data (no key needed for basic)
FOREX_SYMBOL_MAP: Dict[str, str] = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD", "EURGBP": "EUR/GBP", "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY", "AUDJPY": "AUD/JPY", "AUDNZD": "AUD/NZD",
    "GBPAUD": "GBP/AUD", "GBPCAD": "GBP/CAD", "EURNZD": "EUR/NZD",
    "EURAUD": "EUR/AUD", "CADJPY": "CAD/JPY", "CHFJPY": "CHF/JPY",
    "GBPNZD": "GBP/NZD", "GBPCHF": "GBP/CHF", "CADCHF": "CAD/CHF",
    "NZDCHF": "NZD/CHF", "NZDCAD": "NZD/CAD", "NZDJPY": "NZD/JPY",
    "USDTRY": "USD/TRY", "USDZAR": "USD/ZAR", "USDMXN": "USD/MXN",
    "USDSGD": "USD/SGD", "USDNOK": "USD/NOK", "USDSEK": "USD/SEK",
}

# OTC pairs → maps to the base real pair
OTC_BASE_MAP: Dict[str, str] = {
    "EURUSD_OTC": "EURUSD",
    "GBPUSD_OTC": "GBPUSD",
    "USDJPY_OTC": "USDJPY",
    "AUDUSD_OTC": "AUDUSD",
    "USDCAD_OTC": "USDCAD",
    "EURGBP_OTC": "EURGBP",
    "EURJPY_OTC": "EURJPY",
    "GBPJPY_OTC": "GBPJPY",
    "AUDNZD_OTC": "AUDNZD",
    "NZDUSD_OTC": "NZDUSD",
    "BTCUSDT_OTC": "BTCUSDT",
    "ETHUSDT_OTC": "ETHUSDT",
    "BNBUSDT_OTC": "BNBUSDT",
    "XRPUSDT_OTC": "XRPUSDT",
}

# Timeframe map for Twelve Data
TF_MAP_12: Dict[str, str] = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "4h": "4h",
}

# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter with Multiple API Key Support
class RateLimiter:
    """Rate limiter with API key rotation for multiple keys."""
    def __init__(self, api_keys: list = None, min_interval: float = TWELVE_DATA_MIN_DELAY):
        self.api_keys = api_keys or []
        self.min_interval = min_interval
        self.last_request_time = 0.0
        self.last_key_index = 0
        self.lock = asyncio.Lock()

    async def wait(self) -> str:
        """Wait if necessary to maintain rate limit, return next API key."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()
            
            # Rotate API key
            if self.api_keys:
                self.last_key_index = (self.last_key_index + 1) % len(self.api_keys)
                return self.api_keys[self.last_key_index]
            return ""

# Global rate limiter instance - will be initialized with API keys from config
_rate_limiter = RateLimiter()

# ─────────────────────────────────────────────────────────────────────────────

async def fetch_forex_candles_twelvedata(
    pair: str, timeframe: str = "5m", limit: int = 500, api_key: str = ""
) -> pd.DataFrame:
    """Fetch OHLCV candles for a Forex pair via Twelve Data REST API with rate limiting and retries."""
    symbol = FOREX_SYMBOL_MAP.get(pair)
    if not symbol:
        return pd.DataFrame()
    interval = TF_MAP_12.get(timeframe, "5min")
    params: Dict = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": min(limit, 5000),
        "format": "JSON",
        "order": "ASC",
    }
    if api_key:
        params["apikey"] = api_key
    url = f"{TWELVE_DATA_URL}/time_series"
    
    # Implement retry with exponential backoff for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Get next API key (or use provided one)
            api_key_to_use = await _rate_limiter.wait()
            if not api_key_to_use and api_key:
                api_key_to_use = api_key
            
            params_copy = params.copy()
            if api_key_to_use:
                params_copy["apikey"] = api_key_to_use
            
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params_copy)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Too Many Requests
                wait_time = (2 ** attempt) * 5  # exponential backoff: 5s, 10s, 20s
                logger.warning("Rate limited on {} {}. Retrying in {}s...", pair, timeframe, wait_time)
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error("Twelve Data fetch error {} {}: {}", pair, timeframe, e)
                return pd.DataFrame()
        except Exception as e:
            logger.error("Twelve Data fetch error {} {}: {}", pair, timeframe, e)
            return pd.DataFrame()

        if "values" not in data:
            # Free tier sometimes returns status/message
            msg = data.get("message", data.get("status", "unknown error"))
            logger.warning("Twelve Data {} {}: {}", pair, timeframe, msg)
            return pd.DataFrame()

        rows = data["values"]
        df = pd.DataFrame(rows)
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Twelve Data doesn't always return volume for Forex — fill with 0
        if "volume" not in df.columns:
            df["volume"] = 0.0
        else:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
        df.sort_values("timestamp", inplace=True)
        save_candles(pair, timeframe, df)
        logger.info("Twelve Data: fetched {} candles for {} {}", len(df), pair, timeframe)
        return df
    
    logger.error("Twelve Data fetch failed for {} {} after {} retries", pair, timeframe, max_retries)
    return pd.DataFrame()


async def fetch_forex_latest_price(pair: str, api_key: str = "") -> Optional[float]:
    """Fetch the latest tick price for a Forex pair with rate limiting."""
    symbol = FOREX_SYMBOL_MAP.get(pair)
    if not symbol:
        return None
    params: Dict = {"symbol": symbol, "format": "JSON"}
    
    # Get next API key from rate limiter
    api_key_to_use = await _rate_limiter.wait()
    if not api_key_to_use and api_key:
        api_key_to_use = api_key
    
    if api_key_to_use:
        params["apikey"] = api_key_to_use
    
    url = f"{TWELVE_DATA_URL}/price"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            price = float(data.get("price", 0))
            return price if price > 0 else None
    except Exception as e:
        logger.debug("Twelve Data price {} error: {}", pair, e)
        return None


def mirror_otc_candles(otc_pair: str) -> None:
    """
    Copy candles from base pair to OTC pair storage.
    Called after base pair data is updated.
    """
    base = OTC_BASE_MAP.get(otc_pair)
    if not base:
        return
    for tf in ["1m", "5m", "15m", "1h"]:
        df = load_candles(base, tf, limit=2000)
        if not df.empty:
            save_candles(otc_pair, tf, df)


class ForexOTCFeed:
    """
    Polls Twelve Data for Forex pairs every `poll_interval` seconds,
    then mirrors data to OTC pairs.

    Since Forex REST APIs are rate-limited, we batch pairs and stagger fetches.
    For near-real-time 1-second candle building we aggregate ticks into
    synthetic 1m candles using the latest price endpoint.
    """

    def __init__(
        self,
        forex_pairs: List[str],
        otc_pairs: List[str],
        on_candle: Optional[Callable] = None,
        poll_interval: int = 10,          # seconds between price polls
        candle_refresh_interval: int = 60, # seconds between full OHLCV refresh
        api_keys: List[str] = None,
        api_key: str = "",
    ):
        # Filter pairs based on enabled list (if specified in config)
        try:
            enabled_forex = settings.get_enabled_forex_pairs()
            if enabled_forex:
                self.forex_pairs = [p for p in forex_pairs if p in FOREX_SYMBOL_MAP and p in enabled_forex]
            else:
                self.forex_pairs = [p for p in forex_pairs if p in FOREX_SYMBOL_MAP]
        except:
            self.forex_pairs = [p for p in forex_pairs if p in FOREX_SYMBOL_MAP]
        
        self.otc_pairs = otc_pairs
        self.on_candle = on_candle
        self.poll_interval = poll_interval
        self.candle_refresh_interval = candle_refresh_interval
        self.api_key = api_key
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Initialize rate limiter with API keys
        global _rate_limiter
        if not api_keys:
            try:
                api_keys = settings.get_api_keys_list()
            except:
                pass
        
        if api_keys:
            _rate_limiter = RateLimiter(api_keys=api_keys)
        else:
            _rate_limiter = RateLimiter()

        # In-memory tick accumulator for building real-time 1m candles
        # { pair: { "open": float, "high": float, "low": float, "last": float,
        #           "volume": float, "ts_open": datetime } }
        self._tick_candles: Dict[str, dict] = {}

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._price_poll_loop()),
            asyncio.create_task(self._candle_refresh_loop()),
            asyncio.create_task(self._otc_mirror_loop()),
        ]
        logger.info(
            "ForexOTCFeed started: {} forex + {} OTC pairs",
            len(self.forex_pairs), len(self.otc_pairs)
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # ── Tick price poll (every poll_interval seconds) ─────────────────────────

    async def _price_poll_loop(self) -> None:
        """Poll latest prices and accumulate into real-time 1m candles."""
        while self._running:
            for pair in self.forex_pairs:
                if not self._running:
                    break
                try:
                    price = await fetch_forex_latest_price(pair, self.api_key)
                    if price:
                        self._accumulate_tick(pair, price)
                        if self.on_candle:
                            candle = self._current_candle_dict(pair, price)
                            if candle:
                                await self.on_candle(pair, "1m", candle)
                except Exception as e:
                    logger.debug("Price poll {} error: {}", pair, e)
            await asyncio.sleep(self.poll_interval)

    def _accumulate_tick(self, pair: str, price: float) -> None:
        now = datetime.now(timezone.utc)
        minute_ts = now.replace(second=0, microsecond=0)

        if pair not in self._tick_candles or self._tick_candles[pair]["ts_open"] != minute_ts:
            # New candle
            prev = self._tick_candles.get(pair)
            if prev:
                # Flush completed candle to parquet
                finished = pd.DataFrame([{
                    "timestamp": prev["ts_open"],
                    "open": prev["open"], "high": prev["high"],
                    "low": prev["low"], "close": prev["last"],
                    "volume": prev["volume"],
                }])
                save_candles(pair, "1m", finished)
            self._tick_candles[pair] = {
                "ts_open": minute_ts, "open": price, "high": price,
                "low": price, "last": price, "volume": 0.0,
            }
        else:
            tc = self._tick_candles[pair]
            tc["high"] = max(tc["high"], price)
            tc["low"] = min(tc["low"], price)
            tc["last"] = price

    def _current_candle_dict(self, pair: str, price: float) -> Optional[dict]:
        tc = self._tick_candles.get(pair)
        if not tc:
            return None
        return {
            "timestamp": tc["ts_open"],
            "open": tc["open"], "high": tc["high"],
            "low": tc["low"], "close": price, "volume": tc["volume"],
        }

    # ── Full OHLCV candle refresh (every candle_refresh_interval seconds) ─────

    async def _candle_refresh_loop(self) -> None:
        """Refresh full OHLCV history for all Forex pairs periodically."""
        await asyncio.sleep(5)  # small delay on startup
        while self._running:
            for pair in self.forex_pairs:
                if not self._running:
                    break
                try:
                    # Rotate API key for each pair to distribute load
                    api_key = settings.get_next_api_key() or self.api_key
                    for tf in settings.FOREX_TIMEFRAMES:
                        await fetch_forex_candles_twelvedata(pair, tf, limit=300, api_key=api_key)
                except Exception as e:
                    logger.debug("Candle refresh {} error: {}", pair, e)
            logger.debug("ForexOTCFeed: candle refresh cycle complete")
            await asyncio.sleep(self.candle_refresh_interval)

    # ── OTC mirror loop ───────────────────────────────────────────────────────

    async def _otc_mirror_loop(self) -> None:
        """Every 30s mirror base pair candles → OTC pair storage."""
        while self._running:
            for otc in self.otc_pairs:
                try:
                    mirror_otc_candles(otc)
                    # Trigger signal scan for OTC pair via on_candle
                    base = OTC_BASE_MAP.get(otc)
                    if base and self.on_candle:
                        df = load_candles(base, "5m", limit=5)
                        if not df.empty:
                            last = df.iloc[-1]
                            candle_dict = {
                                "timestamp": last["timestamp"],
                                "open": last["open"], "high": last["high"],
                                "low": last["low"], "close": last["close"],
                                "volume": last["volume"],
                            }
                            await self.on_candle(otc, "5m", candle_dict)
                except Exception as e:
                    logger.debug("OTC mirror {} error: {}", otc, e)
            await asyncio.sleep(30)


async def bootstrap_forex_data(
    forex_pairs: List[str],
    timeframes: List[str] = None,
    api_key: str = "",
) -> None:
    """Bootstrap historical OHLCV data for all Forex pairs."""
    if timeframes is None:
        try:
            timeframes = settings.get_forex_timeframes()
        except:
            timeframes = ["5m", "15m", "1h"]
    
    # Filter pairs if enabled list is specified
    try:
        enabled_pairs = settings.get_enabled_forex_pairs()
        if enabled_pairs:
            forex_pairs = [p for p in forex_pairs if p in enabled_pairs]
    except:
        pass
    
    logger.info("Bootstrapping Forex data for {} pairs with timeframes: {}", len(forex_pairs), ",".join(timeframes))
    # Rate limiter handles delays automatically
    for pair in forex_pairs:
        for tf in timeframes:
            await fetch_forex_candles_twelvedata(pair, tf, limit=300, api_key=api_key)
    logger.info("Forex bootstrap complete.")
