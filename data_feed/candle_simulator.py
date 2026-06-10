"""
Candle Simulator — Generates realistic synthetic price movements for live trading simulation.
This simulates real market conditions by creating slightly different candles each minute,
making predictions dynamic and realistic.
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
import pandas as pd
from loguru import logger

try:
    from database import save_candles, load_candles
    from config import settings
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from database import save_candles, load_candles
    from config import settings


class CandleSimulator:
    """
    Generates realistic synthetic candles for live trading simulation.
    - Reads the last real candle
    - Simulates price movement with realistic volatility
    - Creates new candles every minute to simulate live updates
    - Maintains technical indicator characteristics (trend, support, resistance)
    """
    
    def __init__(self, pairs: List[str], timeframe: str = "5m"):
        self.pairs = pairs
        self.timeframe = timeframe
        self.last_prices: Dict[str, float] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Market simulation parameters
        self.volatility_factor = 0.0003  # 0.03% per movement
        self.trend_persistence = 0.6    # 60% chance trend continues
        self.mean_reversion = 0.4       # 40% chance revert to mean
        self.trends: Dict[str, float] = {}  # Current trend per pair (-1 to +1)
    
    async def start(self) -> None:
        """Start generating synthetic candles."""
        self._running = True
        
        # Initialize last prices from current candle data
        for pair in self.pairs:
            try:
                df = load_candles(pair, self.timeframe, limit=1)
                if not df.empty:
                    self.last_prices[pair] = float(df.iloc[-1]["close"])
                    self.trends[pair] = random.uniform(-0.3, 0.3)  # Random initial trend
                else:
                    logger.warning("No candle data found for {} {}", pair, self.timeframe)
            except Exception as e:
                logger.warning("Failed to load initial data for {}: {}", pair, e)
        
        logger.info("CandleSimulator started for {} pairs ({})", len(self.pairs), self.timeframe)
        
        # Start generation loop
        self._tasks = [
            asyncio.create_task(self._generate_loop())
        ]
    
    async def stop(self) -> None:
        """Stop generating synthetic candles."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def _generate_loop(self) -> None:
        """Generate new candles every minute for rapid signal updates."""
        # Wait for next minute boundary
        now = datetime.now(timezone.utc)
        seconds_until_next = 60 - now.second
        if seconds_until_next > 0:
            await asyncio.sleep(seconds_until_next)
        
        while self._running:
            try:
                for pair in self.pairs:
                    try:
                        await self._generate_candle(pair)
                    except Exception as e:
                        logger.debug("Candle generation error for {}: {}", pair, e)
                
                # Wait for next minute
                await asyncio.sleep(60)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Generate loop error: {}", e)
                await asyncio.sleep(5)
    
    async def _generate_candle(self, pair: str) -> None:
        """Generate a new synthetic 5-minute candle for a pair."""
        if pair not in self.last_prices:
            return
        
        try:
            # Load the last candle
            df = load_candles(pair, self.timeframe, limit=1)
            if df.empty:
                logger.warning("No data to simulate for {}", pair)
                return
            
            last_candle = df.iloc[-1]
            last_close = float(last_candle["close"])
            
            # Update trend with momentum
            if random.random() < self.trend_persistence:
                # Continue trend (inertia)
                self.trends[pair] *= 0.95  # Slight decay
                self.trends[pair] += random.uniform(-0.1, 0.1)
            else:
                # Revert or change trend
                self.trends[pair] = random.uniform(-0.3, 0.3)
            
            # Clamp trend
            self.trends[pair] = max(-1.0, min(1.0, self.trends[pair]))
            
            # Generate price movements with larger volatility for significant changes
            volatility = self.volatility_factor * (1.0 + abs(self.trends[pair]))
            
            # Generate OHLC with trend influence - increase range for visible changes
            open_price = last_close * (1 + random.uniform(-volatility * 0.5, volatility * 0.5))
            
            direction = 1 if self.trends[pair] > 0 else -1
            close_price = open_price * (1 + direction * random.uniform(volatility * 0.2, volatility * 1.5))
            
            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility * 0.8))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility * 0.8))
            
            # Ensure low < high
            if low_price > high_price:
                low_price, high_price = high_price, low_price
            
            # Volume simulation (slightly random)
            volume = float(last_candle.get("volume", 1000)) * random.uniform(0.8, 1.2)
            
            # Generate timestamp for NEXT 5-minute candle
            now = datetime.now(timezone.utc)
            # Calculate the timestamp for the current 5-minute candle boundary
            last_ts = pd.Timestamp(last_candle["timestamp"])
            # Next 5-minute candle is 5 minutes after the last candle
            new_ts = last_ts + timedelta(minutes=5)
            
            # Create and save new candle
            new_candle = pd.DataFrame([{
                "timestamp": new_ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }])
            
            save_candles(pair, self.timeframe, new_candle)
            self.last_prices[pair] = close_price
            
            logger.info(
                "📊 New candle {} {}: O:{:.6f} H:{:.6f} L:{:.6f} C:{:.6f} (Trend: {:.2f})",
                pair, new_ts, open_price, high_price, low_price, close_price, self.trends[pair]
            )
        
        except Exception as e:
            logger.error("Candle generation failed for {}: {}", pair, e)


async def start_candle_simulator(pairs: List[str]) -> CandleSimulator:
    """Initialize and start the candle simulator."""
    simulator = CandleSimulator(pairs=pairs, timeframe="5m")
    await simulator.start()
    return simulator
