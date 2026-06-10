"""
main.py — Entry point for ICT Signals Bot.
Supports Crypto (Binance WS), Forex (Twelve Data REST), and OTC (mirrored) pairs.

Run:  python main.py
"""
from __future__ import annotations

import asyncio
import os
import signal as sys_signal
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from loguru import logger
from config import settings, CRYPTO_PAIRS, FOREX_PAIRS, OTC_PAIRS
from database import init_db
from data_feed.binance_ws import BinanceFeed, fetch_historical_candles
from data_feed.forex_feed import ForexOTCFeed, bootstrap_forex_data
from data_feed.candle_simulator import start_candle_simulator
from signals.engine import scan_pairs
from signals.status_engine import get_status_engine
from signals.status_display import format_all_pairs_status, format_detailed_status_with_advanced, log_status_summary, log_detailed_status
from bot.handlers import build_application
from bot.notifier import send_alert, send_signal


def setup_logging() -> None:
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(settings.LOGS_DIR / "bot_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        compression="gz",
    )


async def bootstrap_crypto_data(pairs: list, timeframes: list) -> None:
    logger.info("Bootstrapping crypto data for {} pairs...", len(pairs))
    tasks = [
        fetch_historical_candles(pair, tf, limit=500)
        for pair in pairs
        for tf in timeframes
    ]
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        results = await asyncio.gather(*tasks[i: i + batch_size], return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Crypto bootstrap failed: {}", r)
        await asyncio.sleep(1)
    logger.info("Crypto bootstrap complete.")


async def signal_scan_loop(all_pairs: list, app, interval_seconds: int = 300) -> None:
    """
    Periodic full scan across ALL pairs every 5 minutes (matches primary TF).
    Only fires on_signal when a NEW signal is emitted (not every scan).
    """
    logger.info("Signal scanner started (every {}s) for {} pairs.", interval_seconds, len(all_pairs))
    while True:
        try:
            paused = app.bot_data.get("paused", False)
            if not paused:
                logger.debug("Scanning {} pairs...", len(all_pairs))
                signals = await scan_pairs(all_pairs, on_signal=send_signal)
                if signals:
                    logger.info("Found {} new signal(s) this cycle.", len(signals))
            else:
                logger.debug("Scanner paused.")
        except Exception as e:
            logger.error("Signal scan error: {}", e)
        await asyncio.sleep(interval_seconds)


async def status_display_loop(all_pairs: list, app, interval_seconds: int = 300) -> None:
    """
    Checks pair status every 5 minutes.
    ONLY sends Telegram message when a pair gets a FRESH BUY or SELL signal
    (direction was WAIT before, now BUY/SELL — prevents every-minute spam).
    """
    logger.info("Status display started (every {}s) for {} pairs.", interval_seconds, len(all_pairs))
    status_engine = get_status_engine()
    # Track last known direction per pair to detect NEW signals only
    last_status: dict = {pair: "WAIT" for pair in all_pairs}

    while True:
        try:
            paused = app.bot_data.get("paused", False)
            if not paused and all_pairs:
                statuses = await status_engine.get_all_pairs_status(all_pairs)

                if statuses:
                    log_status_summary(statuses)
                    new_signals = []

                    for pair, status in statuses.items():
                        st = status.get("status", "WAIT")
                        conf = status.get("confidence", 0)
                        prev = last_status.get(pair, "WAIT")

                        # Only alert on FRESH signals (prev was WAIT, now is BUY/SELL)
                        # OR direction flipped (BUY → SELL)
                        if st in ("BUY", "SELL") and st != prev and conf >= settings.MIN_CONFIDENCE:
                            new_signals.append((pair, status))
                            last_status[pair] = st
                        elif st == "WAIT":
                            last_status[pair] = "WAIT"

                    # Send one grouped message for all new signals
                    if new_signals:
                        from datetime import datetime, timezone, timedelta
                        now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
                        tf = settings.PRIMARY_TIMEFRAME
                        dur = int(tf[:-1]) if tf.endswith("m") else int(tf[:-1]) * 60
                        expiry_str = (now_ist + timedelta(minutes=dur)).strftime("%I:%M %p")

                        msg_lines = [
                            f"📊 *Market Signal Alert — {now_ist.strftime('%I:%M %p')} IST*\n",
                            f"🟢 BUY: {sum(1 for _, s in new_signals if s['status']=='BUY')}  "
                            f"🔴 SELL: {sum(1 for _, s in new_signals if s['status']=='SELL')}\n",
                        ]
                        for pair, status in new_signals:
                            st = status.get("status")
                            conf = status.get("confidence", 0)
                            emoji = "🟢" if st == "BUY" else "🔴"
                            arrow = "↑" if st == "BUY" else "↓"
                            price = status.get("current_price", 0)
                            msg_lines.append(
                                f"{emoji} *{pair}* {arrow} `{conf:.1f}%` — `{price:.5f}`\n"
                                f"   ⏳ {tf} trade · expires `{expiry_str} IST`"
                            )
                        msg_lines.append("\n⚠️ _Not financial advice._")
                        try:
                            await send_alert("\n".join(msg_lines), level="INFO")
                        except Exception as e:
                            logger.warning("Alert send failed: {}", e)
            else:
                logger.debug("Status display paused.")
        except Exception as e:
            logger.error("Status display error: {}", e)

        await asyncio.sleep(interval_seconds)


async def ml_retrain_loop(interval_hours: int = 6) -> None:
    """
    Periodically retrains the ML models in a background thread (non-blocking).
    """
    logger.info("ML retraining loop started (every {} hours).", interval_hours)
    from ml.trainer import retrain_all
    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)
            logger.info("Starting scheduled ML retraining...")
            # Run in thread pool so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, retrain_all)
            logger.info("ML retraining complete.")
        except Exception as e:
            logger.error("ML retrain error: {}", e)


async def resolve_open_signals_loop(interval_seconds: int = 60) -> None:
    """
    Periodically checks OPEN signals in the database.
    If the signal duration has passed, checks the current price to record WIN/LOSS.
    """
    from database import get_open_signals, close_signal, load_candles
    from datetime import datetime, timezone, timedelta
    
    logger.info("Trade resolver loop started (every {}s).", interval_seconds)
    while True:
        try:
            open_signals = await get_open_signals()
            now = datetime.now(timezone.utc)
            
            for sig in open_signals:
                # Calculate expiry time
                tf = sig.timeframe
                dur = int(tf[:-1]) if tf.endswith("m") else int(tf[:-1]) * 60
                
                # If sig.created_at is naive, make it UTC
                created = sig.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                    
                expiry = created + timedelta(minutes=dur)
                
                # If trade has expired, resolve it
                if now >= expiry:
                    # Load latest candle to get the closing price
                    df = load_candles(sig.pair, "5m", limit=1)
                    if not df.empty:
                        current_price = float(df["close"].iloc[-1])
                        entry = sig.entry_price
                        pnl = 0.0
                        
                        if sig.direction == "BUY":
                            is_win = current_price > entry
                            pnl = (current_price - entry) / entry * 100
                        else:  # SELL
                            is_win = current_price < entry
                            pnl = (entry - current_price) / entry * 100
                            
                        result = "WIN" if is_win else "LOSS"
                        await close_signal(sig.id, result, pnl)
                        logger.info("Resolved trade #{} for {}: {} -> {} (Entry: {}, Exit: {})", 
                                    sig.id, sig.pair, sig.direction, result, entry, current_price)
                        
        except Exception as e:
            logger.error("Trade resolver error: {}", e)
            
        await asyncio.sleep(interval_seconds)

async def main() -> None:
    setup_logging()
    logger.info("Starting {} v{}", settings.APP_NAME, settings.VERSION)

    # 1. Database
    await init_db()

    # 2. Telegram app
    tg_app = build_application()
    await tg_app.initialize()
    await tg_app.start()

    # 3. Define pairs to monitor
    # Only use OTC pairs for signal generation (optimized for API usage)
    enabled_otc = settings.get_enabled_otc_pairs()
    if enabled_otc:
        live_otc = enabled_otc
    else:
        live_otc = []
    
    live_crypto = []  # Disabled - use OTC only
    live_forex = []   # Disabled - use OTC only
    all_live = live_otc

    logger.info(
        "Monitoring: {} OTC pairs for signal generation",
        len(live_otc)
    )

    # 4. Bootstrap disabled - using OTC pairs only
    # Crypto and Forex bootstrapping skipped

    # 5. Binance WebSocket feed disabled - using OTC only
    # Crypto real-time feed skipped
    async def on_candle(pair, tf, candle):
        # Trigger scan on any of the signal timeframes (not just primary)
        if tf in ("5m", "15m", "30m", "1h", "4h"):
            paused = tg_app.bot_data.get("paused", False)
            if not paused:
                await scan_pairs([pair], on_signal=send_signal)

    # 7. Binance WebSocket feed disabled - using OTC only
    # crypto_feed = BinanceFeed(
    #     pairs=live_crypto,
    #     timeframes=["1m", "5m", "15m", "1h"],
    #     on_candle=on_candle,
    # )
    # await crypto_feed.start()

    # 8. OTC Feed - needed for continuous candle updates for signal generation
    forex_feed = ForexOTCFeed(
        forex_pairs=[],  # Empty - no forex pairs
        otc_pairs=live_otc,  # Only OTC pairs
        on_candle=on_candle,
        poll_interval=settings.FOREX_POLL_INTERVAL,
        candle_refresh_interval=settings.FOREX_CANDLE_REFRESH,
        api_keys=settings.get_api_keys_list(),  # Pass all API keys for rotation
    )
    await forex_feed.start()

    # 8b. Candle Simulator - generates realistic synthetic candles for live trading
    # This makes predictions dynamic by simulating price movements each 5-minute candle
    simulator = await start_candle_simulator(live_otc)

    # 9. Periodic full scan (every 60s for all pairs)
    scanner_task = asyncio.create_task(
        signal_scan_loop(all_live, tg_app, interval_seconds=60)
    )
    
    # 9b. Status display loop (every 60s for all pairs)
    status_task = asyncio.create_task(
        status_display_loop(all_live, tg_app, interval_seconds=60)
    )

    # 9c. ML Retraining loop (every X hours)
    ml_task = asyncio.create_task(
        ml_retrain_loop(interval_hours=settings.RETRAIN_INTERVAL_HOURS)
    )
    
    # 9d. Resolver loop
    resolver_task = asyncio.create_task(
        resolve_open_signals_loop(interval_seconds=60)
    )

    # 10. Notify admins
    await send_alert(
        f"🚀 {settings.APP_NAME} v{settings.VERSION} started!\n"
        f"📊 Monitoring {len(live_otc)} OTC pairs for signals\n"
        f"⚡ Advanced multi-indicator signal generation active.",
        level="SUCCESS",
    )

    logger.info("Bot running. Press Ctrl+C to stop.")

    try:
        await tg_app.updater.start_polling(drop_pending_updates=True)
        stop_event = asyncio.Event()

        def _handle_signal(*_):
            stop_event.set()

        loop = asyncio.get_event_loop()
        for sig in (sys_signal.SIGINT, sys_signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except NotImplementedError:
                pass

        await stop_event.wait()

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logger.info("Shutting down...")
        scanner_task.cancel()
        status_task.cancel()
        ml_task.cancel()
        try:
            await simulator.stop()
        except:
            pass
        try:
            await forex_feed.stop()
        except:
            pass
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()
        logger.info("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
