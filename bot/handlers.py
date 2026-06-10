"""
Telegram Bot Handlers — commands, callbacks, and message routing.
"""
from __future__ import annotations

import asyncio
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters,
)

from config import settings, ALL_PAIRS, FOREX_PAIRS, CRYPTO_PAIRS, OTC_PAIRS
from database import get_overall_stats, get_pair_stats, get_signal_history, load_candles, reset_stats


def _is_admin(user_id: int) -> bool:
    return user_id in settings.TELEGRAM_ADMIN_IDS


def _signal_emoji(direction: str) -> str:
    return "🟢" if direction == "BUY" else "🔴"


def _format_signal(sig: dict) -> str:
    emoji = _signal_emoji(sig["direction"])
    lines = [
        f"{emoji} *{sig['pair']}* — {sig['direction']}",
        f"⏱ Timeframe: `{sig.get('timeframe', '?')}`",
        f"🎯 Confidence: `{sig.get('confidence', 0):.1f}%`",
        f"💰 Entry: `{sig.get('entry_price', 0):.6f}`",
    ]
    if sig.get("target_price"):
        lines.append(f"✅ Target: `{sig['target_price']:.6f}`")
    if sig.get("stop_loss"):
        lines.append(f"🛑 Stop Loss: `{sig['stop_loss']:.6f}`")
    if sig.get("reason"):
        lines.append(f"📝 {sig['reason'][:200]}")
    return "\n".join(lines)


def _pair_type_emoji(pair: str) -> str:
    if "_OTC" in pair:
        return "🔵"
    if pair in CRYPTO_PAIRS:
        return "🟡"
    return "🌍"


async def _get_pairs_with_data() -> dict:
    """Returns dict of pair -> candle count for pairs that have data."""
    result = {"crypto": [], "forex": [], "otc": []}
    for pair in CRYPTO_PAIRS:
        df = load_candles(pair, "5m", limit=5)
        if not df.empty:
            result["crypto"].append(pair)
    for pair in FOREX_PAIRS:
        df = load_candles(pair, "5m", limit=5)
        if not df.empty:
            result["forex"].append(pair)
    for pair in OTC_PAIRS:
        df = load_candles(pair, "5m", limit=5)
        if not df.empty:
            result["otc"].append(pair)
    return result


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("📊 Stats", callback_data="stats"),
         InlineKeyboardButton("📈 Signals", callback_data="signals")],
        [InlineKeyboardButton("🪙 Crypto Pairs", callback_data="pairs_crypto"),
         InlineKeyboardButton("🌍 Forex Pairs", callback_data="pairs_forex")],
        [InlineKeyboardButton("🔵 OTC Pairs", callback_data="pairs_otc"),
         InlineKeyboardButton("✅ Live Data", callback_data="pairs_live")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")],
    ]
    await update.message.reply_text(
        f"🤖 *{settings.APP_NAME}* v{settings.VERSION}\n\n"
        "High-confidence ML trading signals for:\n"
        "🟡 Crypto (Binance real-time)\n"
        "🌍 Forex (Twelve Data)\n"
        "🔵 OTC (Pocket Option / Quotex)\n\n"
        "Use the buttons below or /help for commands.",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📋 *Commands*\n\n"
        "/start — Welcome menu\n"
        "/stats — Performance stats\n"
        "/signals — Last 10 signals\n"
        "/pairs — All pairs with data status\n"
        "/pairs\\_crypto — Crypto pairs\n"
        "/pairs\\_forex — Forex pairs\n"
        "/pairs\\_otc — OTC pairs\n"
        "/live — Pairs with live data\n"
        "/status — Bot status\n"
        "/help — This help\n\n"
        "_Admin:_ /retrain /pause /resume"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        s = await get_overall_stats()
        text = (
            "📊 *Overall Performance*\n\n"
            f"📨 Total: `{s['total_signals']}`\n"
            f"✅ Wins: `{s['wins']}`\n"
            f"❌ Losses: `{s['losses']}`\n"
            f"🏆 Win Rate: `{s['win_rate']}%`\n"
            f"🎯 Avg Confidence: `{s['avg_confidence']}%`\n\n"
        )
        pair_stats = await get_pair_stats()
        if pair_stats:
            text += "*Top Pairs:*\n"
            for ps in pair_stats[:5]:
                text += f"• `{ps['pair']}`: {ps['wins']}W / {ps['losses']}L ({ps['win_rate']}%)\n"
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.error("Stats error: {}", e)
        await update.message.reply_text("❌ Could not fetch stats.")


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        signals = await get_signal_history(limit=10)
        if not signals:
            await update.message.reply_text("📭 No signals yet.")
            return
        for sig in signals[:10]:
            await update.message.reply_text(_format_signal(sig), parse_mode="Markdown")
    except Exception as e:
        logger.error("Signals error: {}", e)
        await update.message.reply_text("❌ Could not fetch signals.")


async def cmd_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all pairs grouped by type with data status."""
    data = await _get_pairs_with_data()
    crypto_count = len(data["crypto"])
    forex_count  = len(data["forex"])
    otc_count    = len(data["otc"])
    total = crypto_count + forex_count + otc_count

    text = (
        f"📡 *Pairs with Live Data* ({total} active)\n\n"
        f"🟡 *Crypto* ({crypto_count}): " +
        (", ".join(f"`{p}`" for p in data["crypto"][:10]) or "fetching...") +
        ("\n  _(+{} more)_".format(crypto_count - 10) if crypto_count > 10 else "") +
        f"\n\n🌍 *Forex* ({forex_count}): " +
        (", ".join(f"`{p}`" for p in data["forex"][:10]) or "fetching...") +
        ("\n  _(+{} more)_".format(forex_count - 10) if forex_count > 10 else "") +
        f"\n\n🔵 *OTC* ({otc_count}): " +
        (", ".join(f"`{p.replace('_OTC','')}_OTC`" for p in data["otc"]) or "mirroring...") +
        "\n\n_Use /pairs\\_crypto, /pairs\\_forex, /pairs\\_otc for detail_"
    )
    keyboard = [[
        InlineKeyboardButton("🟡 Crypto", callback_data="pairs_crypto"),
        InlineKeyboardButton("🌍 Forex", callback_data="pairs_forex"),
        InlineKeyboardButton("🔵 OTC", callback_data="pairs_otc"),
    ]]
    await update.message.reply_text(
        text, parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def cmd_pairs_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["🟡 *Crypto Pairs (Binance Real-time)*\n"]
    for pair in CRYPTO_PAIRS:
        df = load_candles(pair, "5m", limit=2)
        status = "✅" if not df.empty else "⏳"
        candles = load_candles(pair, "5m", limit=500)
        n = len(candles)
        lines.append(f"{status} `{pair}` — {n} candles")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_pairs_forex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["🌍 *Forex Pairs (Twelve Data)*\n"]
    for pair in FOREX_PAIRS:
        df = load_candles(pair, "5m", limit=2)
        status = "✅" if not df.empty else "⏳"
        candles = load_candles(pair, "5m", limit=500)
        n = len(candles)
        lines.append(f"{status} `{pair}` — {n} candles")
    await update.message.reply_text("\n".join(lines[:30]), parse_mode="Markdown")


async def cmd_pairs_otc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["🔵 *OTC Pairs (Pocket Option / Quotex)*\n",
             "_OTC data mirrors real pair candles_\n"]
    for pair in OTC_PAIRS:
        base = pair.replace("_OTC", "")
        df = load_candles(pair, "5m", limit=2)
        status = "✅" if not df.empty else "⏳"
        candles = load_candles(pair, "5m", limit=500)
        n = len(candles)
        lines.append(f"{status} `{pair}` ← `{base}` — {n} candles")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show only pairs that currently have live data."""
    data = await _get_pairs_with_data()
    all_live = data["crypto"] + data["forex"] + data["otc"]
    if not all_live:
        await update.message.reply_text("⏳ Fetching data... try again in 1 minute.")
        return
    lines = [f"✅ *{len(all_live)} pairs with live data:*\n"]
    for p in all_live:
        lines.append(f"• `{p}`")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    data = await _get_pairs_with_data()
    total_live = len(data["crypto"]) + len(data["forex"]) + len(data["otc"])
    text = (
        f"✅ *Bot Status: Running*\n\n"
        f"🕐 Time: `{now}`\n"
        f"🤖 Version: `{settings.VERSION}`\n"
        f"📡 Primary TF: `{settings.PRIMARY_TIMEFRAME}`\n"
        f"🎯 Min Confidence: `{settings.MIN_CONFIDENCE}%`\n"
        f"📊 Live Pairs: `{total_live}`\n"
        f"  🟡 Crypto: `{len(data['crypto'])}`\n"
        f"  🌍 Forex: `{len(data['forex'])}`\n"
        f"  🔵 OTC: `{len(data['otc'])}`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Admin only.")
        return
    msg = await update.message.reply_text("🔄 Retraining ML models in background... This may take a few minutes.")
    
    async def run_retrain():
        from ml.trainer import retrain_all
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, retrain_all)
        if metrics:
            await msg.edit_text(f"✅ *Retraining Complete*\n\nRandom Forest: `{metrics.get('rf_acc', 0):.2f}`\nXGBoost: `{metrics.get('xgb_acc', 0):.2f}`\nLightGBM: `{metrics.get('lgb_acc', 0):.2f}`", parse_mode="Markdown")
        else:
            await msg.edit_text("❌ Retraining failed or no data available.")
            
    asyncio.create_task(run_retrain())


async def cmd_resetstats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Admin only.")
        return
        
    args = context.args
    if not args or args[0].lower() != "confirm":
        await update.message.reply_text(
            "⚠️ *WARNING*: This will permanently delete all signal history and win/loss records.\n\n"
            "To proceed, send: `/resetstats confirm`",
            parse_mode="Markdown"
        )
        return
        
    success = await reset_stats()
    if success:
        await update.message.reply_text("✅ All statistics and signal history have been completely reset.")
    else:
        await update.message.reply_text("❌ Error resetting statistics.")


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Admin only.")
        return
    context.bot_data["paused"] = True
    await update.message.reply_text("⏸ Signal generation paused.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Admin only.")
        return
    context.bot_data["paused"] = False
    await update.message.reply_text("▶️ Signal generation resumed.")


# ── Callback Query Handler ────────────────────────────────────────────────────

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "stats":
        s = await get_overall_stats()
        text = (
            "📊 *Performance*\n\n"
            f"📨 Total: `{s['total_signals']}`\n"
            f"✅ Wins: `{s['wins']}`\n"
            f"❌ Losses: `{s['losses']}`\n"
            f"🏆 Win Rate: `{s['win_rate']}%`"
        )
        await query.edit_message_text(text, parse_mode="Markdown")

    elif data == "signals":
        signals = await get_signal_history(limit=5)
        if not signals:
            await query.edit_message_text("📭 No signals yet.")
            return
        lines = ["📈 *Last 5 Signals*\n"]
        for sig in signals:
            e = _signal_emoji(sig["direction"])
            lines.append(
                f"{e} `{sig['pair']}` {sig['direction']} "
                f"({sig.get('confidence', 0):.0f}%) — {sig.get('result', 'OPEN')}"
            )
        await query.edit_message_text("\n".join(lines), parse_mode="Markdown")

    elif data in ("pairs_crypto", "pairs_forex", "pairs_otc", "pairs_live"):
        pair_data = await _get_pairs_with_data()
        if data == "pairs_crypto":
            pairs = CRYPTO_PAIRS
            title = "🟡 *Crypto Pairs*"
        elif data == "pairs_forex":
            pairs = FOREX_PAIRS
            title = "🌍 *Forex Pairs*"
        elif data == "pairs_otc":
            pairs = OTC_PAIRS
            title = "🔵 *OTC Pairs*"
        else:
            pairs = pair_data["crypto"] + pair_data["forex"] + pair_data["otc"]
            title = "✅ *Pairs with Live Data*"

        lines = [f"{title} ({len(pairs)})\n"]
        active = pair_data["crypto"] + pair_data["forex"] + pair_data["otc"]
        for p in pairs[:25]:
            status = "✅" if p in active else "⏳"
            lines.append(f"{status} `{p}`")
        if len(pairs) > 25:
            lines.append(f"_...and {len(pairs) - 25} more_")
        await query.edit_message_text("\n".join(lines), parse_mode="Markdown")

    elif data == "help":
        await query.edit_message_text(
            "📋 *Commands*\n\n"
            "/stats /signals /pairs\n"
            "/pairs\\_crypto /pairs\\_forex /pairs\\_otc\n"
            "/live /status /help",
            parse_mode="Markdown"
        )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("❓ Unknown command. Use /help.")


# ── Build Application ─────────────────────────────────────────────────────────

def build_application() -> Application:
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("pairs", cmd_pairs))
    app.add_handler(CommandHandler("pairs_crypto", cmd_pairs_crypto))
    app.add_handler(CommandHandler("pairs_forex", cmd_pairs_forex))
    app.add_handler(CommandHandler("pairs_otc", cmd_pairs_otc))
    app.add_handler(CommandHandler("live", cmd_live))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("retrain", cmd_retrain))
    app.add_handler(CommandHandler("resetstats", cmd_resetstats))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    return app
