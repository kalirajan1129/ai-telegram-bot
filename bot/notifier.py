"""
Telegram Notifier — sends formatted trading signals and alerts.
"""
from __future__ import annotations

import asyncio
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
import httpx
from loguru import logger
from config import settings


async def send_message(text: str, chat_id: Optional[str] = None, parse_mode: str = "Markdown") -> bool:
    chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    if not settings.TELEGRAM_BOT_TOKEN or not chat_id:
        logger.warning("Telegram not configured — skipping send.")
        return False
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return True
            logger.error("Telegram API error {}: {}", resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        logger.error("Telegram send failed: {}", e)
        return False


async def send_signal(signal: dict) -> bool:
    from datetime import datetime, timezone, timedelta
    d = signal.get("direction", "?")
    pair = signal.get("pair", "?")
    tf = signal.get("timeframe", "5m")
    confidence = signal.get("confidence", 0)
    entry = signal.get("entry_price", 0)
    target = signal.get("target_price")
    stop = signal.get("stop_loss")
    reason = signal.get("reason", "")
    indicators_used = signal.get("indicators_used", 0)

    # Calculate trade timing
    now = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    now_ist = now + ist_offset

    # Parse timeframe duration in minutes
    if tf.endswith("m"):
        duration_min = int(tf[:-1])
    elif tf.endswith("h"):
        duration_min = int(tf[:-1]) * 60
    else:
        duration_min = 5

    expiry_ist = now_ist + timedelta(minutes=duration_min)
    trade_time_str = now_ist.strftime("%I:%M %p")
    expiry_time_str = expiry_ist.strftime("%I:%M %p")

    emoji = "🟢" if d == "BUY" else "🔴"
    action_word = "BUY ↑" if d == "BUY" else "SELL ↓"

    # Confidence bar
    filled = int(confidence / 10)
    bar = "█" * filled + "░" * (10 - filled)

    # Collect indicator signals from reason string
    indicator_lines = ""
    if reason:
        indicator_lines = f"\n📊 Basis: `{reason[:200]}`"

    lines = [
        f"{emoji} *{pair}* — *{action_word}*",
        f"",
        f"⏰ *Trade Now:* `{trade_time_str} IST`",
        f"⏳ *Duration:* `{tf}` ({duration_min} min)",
        f"🕒 *Expires At:* `{expiry_time_str} IST`",
        f"",
        f"🎯 *Confidence:* `{confidence:.1f}%`",
        f"`[{bar}]`",
        f"💰 *Entry:* `{entry:.5f}`",
    ]
    if target:
        lines.append(f"✅ *Target:* `{target:.5f}`")
    if stop:
        lines.append(f"🛑 *Stop:* `{stop:.5f}`")
    if indicator_lines:
        lines.append(indicator_lines)
    lines.append(f"")
    lines.append(f"⚠️ _Not financial advice. Binary trading is high risk._")

    return await send_message("\n".join(lines))


async def send_alert(message: str, level: str = "INFO") -> bool:
    icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🚨", "SUCCESS": "✅"}
    icon = icons.get(level.upper(), "ℹ️")
    text = f"{icon} *System*\n\n{message}"
    tasks = [send_message(text, chat_id=str(aid)) for aid in settings.TELEGRAM_ADMIN_IDS]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        return True
    return await send_message(text)
