"""
Enhanced status notifier — sends Telegram updates with rich formatting.
"""
from __future__ import annotations

import asyncio
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional
import httpx
from loguru import logger
from config import settings


async def send_status_update(statuses: Dict[str, Dict]) -> bool:
    """Send detailed status update to Telegram."""
    lines = []
    
    lines.append("📊 *TRADING STATUS UPDATE*\n")
    lines.append("━" * 50)
    
    buy_count = sum(1 for s in statuses.values() if s.get("status") == "BUY")
    sell_count = sum(1 for s in statuses.values() if s.get("status") == "SELL")
    wait_count = sum(1 for s in statuses.values() if s.get("status") == "WAIT")
    
    lines.append(f"🟢 BUY: {buy_count} | 🔴 SELL: {sell_count} | 🟡 WAIT: {wait_count}\n")
    
    for pair, status in sorted(statuses.items()):
        state = status.get("status", "?")
        conf = status.get("confidence", 0)
        price = status.get("current_price", 0)
        indicators = status.get("indicators", {})
        
        emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "WAIT": "🟡",
            "ERROR": "⚠️"
        }.get(state, "❓")
        
        lines.append(f"{emoji} *{pair}* — {state}")
        lines.append(f"   💰 {price:.6f}")
        lines.append(f"   📈 Confidence: {conf:.1f}%")
        lines.append(f"   📊 ADX: {indicators.get('ADX', 0):.0f} | RSI: {indicators.get('RSI', 0):.0f}")
        lines.append("")
    
    lines.append("━" * 50)
    lines.append("_⚠️ Not financial advice._")
    
    text = "\n".join(lines)
    return await send_message(text)


async def send_message(text: str, chat_id: Optional[str] = None, parse_mode: str = "Markdown") -> bool:
    chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    if not settings.TELEGRAM_BOT_TOKEN or not chat_id:
        logger.debug("Telegram not configured — skipping send.")
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
        logger.debug("Telegram send failed: {}", e)
        return False


async def send_signal(signal: dict) -> bool:
    d = signal.get("direction", "?")
    emoji = "🟢" if d == "BUY" else "🔴"
    lines = [
        f"{emoji} *{signal.get('pair','?')}* — {d}",
        f"⏱ Timeframe: `{signal.get('timeframe','?')}`",
        f"🎯 Confidence: `{signal.get('confidence',0):.1f}%`",
        f"💰 Entry: `{signal.get('entry_price',0):.5f}`",
    ]
    if signal.get("target_price"):
        lines.append(f"✅ Target: `{signal['target_price']:.5f}`")
    if signal.get("stop_loss"):
        lines.append(f"🛑 Stop: `{signal['stop_loss']:.5f}`")
    if signal.get("reason"):
        lines.append(f"📝 {signal['reason'][:300]}")
    lines.append("\n_⚠️ Not financial advice._")
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
