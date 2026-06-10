"""
Status Display — formats and displays per-pair trading status with advanced analysis.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List
from loguru import logger


def format_pair_status(status: Dict) -> str:
    """Format a single pair's status for display."""
    pair = status.get("pair", "?")
    state = status.get("status", "ERROR")
    conf = status.get("confidence", 0)
    price = status.get("current_price", 0)
    duration = status.get("duration", {})
    
    # Emoji status
    emoji = {
        "BUY": "🟢",
        "SELL": "🔴",
        "WAIT": "🟡",
        "ERROR": "⚠️"
    }.get(state, "❓")
    
    validity = duration.get("validity", "")
    
    line = f"{emoji} {pair:8} | {state:4} | {conf:6.1f}% | {validity:10} | {price:.6f}"
    
    return line


def format_all_pairs_status(statuses: Dict[str, Dict]) -> str:
    """Format status for all pairs."""
    lines = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    lines.append(f"\n{'='*130}")
    lines.append(f"📊 TRADING STATUS — {timestamp}")
    lines.append(f"{'='*130}")
    lines.append(f"{'Pair':8} | {'State':4} | {'Confidence':10} | {'Duration':10} | {'Price':12} | Additional Indicators")
    lines.append(f"{'-'*130}")
    
    buy_count = 0
    sell_count = 0
    wait_count = 0
    
    for pair, status in sorted(statuses.items()):
        state = status.get("status", "ERROR")
        conf = status.get("confidence", 0)
        price = status.get("current_price", 0)
        duration = status.get("duration", {})
        indicators = status.get("indicators", {})
        advanced = status.get("advanced", {})
        
        emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "WAIT": "🟡",
            "ERROR": "⚠️"
        }.get(state, "❓")
        
        conf_bar = "▇" * int(conf // 10) + "▁" * (10 - int(conf // 10))
        validity = duration.get("validity", "N/A")
        
        # Additional info
        trend = indicators.get("Trend", "?")
        trend_str = indicators.get("Trend_Strength", 0)
        rsi = indicators.get("RSI", 0)
        adx = indicators.get("ADX", 0)
        patterns = indicators.get("Patterns", 0)
        
        add_info = f"ADX:{adx:.0f} RSI:{rsi:.0f} Trend:{trend} Patterns:{patterns}"
        
        line = f"{emoji} {pair:6} | {state:4} | {conf:6.1f}% | {validity:10} | {price:12.6f} | {add_info}"
        lines.append(line)
        
        if state == "BUY":
            buy_count += 1
        elif state == "SELL":
            sell_count += 1
        elif state == "WAIT":
            wait_count += 1
    
    lines.append(f"{'-'*130}")
    lines.append(f"Summary: 🟢 BUY={buy_count} | 🔴 SELL={sell_count} | 🟡 WAIT={wait_count}")
    lines.append(f"{'='*130}\n")
    
    return "\n".join(lines)


def format_detailed_status_with_advanced(statuses: Dict[str, Dict]) -> str:
    """Format detailed status with all advanced indicators."""
    lines = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    lines.append(f"\n╔{'═'*150}╗")
    lines.append(f"║  📊 DETAILED TRADING STATUS WITH ADVANCED ANALYSIS — {timestamp:<98}║")
    lines.append(f"╠{'═'*150}╣")
    
    for pair, status in sorted(statuses.items()):
        state = status.get("status", "ERROR")
        conf = status.get("confidence", 0)
        price = status.get("current_price", 0)
        duration = status.get("duration", {})
        indicators = status.get("indicators", {})
        advanced = status.get("advanced", {})
        
        emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "WAIT": "🟡",
            "ERROR": "⚠️"
        }.get(state, "❓")
        
        conf_bar = "▇" * int(conf // 10) + "▁" * (10 - int(conf // 10))
        
        lines.append(f"║")
        lines.append(f"║  {emoji} {pair:8} — {state:4} @ {price:.6f}")
        lines.append(f"║     Confidence: {conf:6.1f}% [{conf_bar}] | Duration: {duration.get('validity', 'N/A')}")
        lines.append(f"║     ")
        
        # Basic indicators
        lines.append(f"║     📈 Basic Indicators:")
        lines.append(f"║        • ADX: {indicators.get('ADX', 0):.2f} (Trend Strength)")
        lines.append(f"║        • RSI: {indicators.get('RSI', 0):.2f} (Momentum)")
        lines.append(f"║        • ATR%: {indicators.get('ATR%', 0):.4f} (Volatility)")
        lines.append(f"║        • MA Signal: {indicators.get('MA_Signal', '?')} (Golden/Death Cross)")
        lines.append(f"║        • BB Position: {indicators.get('BB_Position', 0):.1f}% (Bollinger Bands)")
        
        # Trend
        trend = advanced.get("trend", {})
        if trend:
            lines.append(f"║     ")
            lines.append(f"║     📊 Trend Analysis:")
            lines.append(f"║        • Direction: {trend.get('direction', '?')}")
            lines.append(f"║        • Strength: {trend.get('strength', 0) * 100:.1f}%")
            lines.append(f"║        • Up Candles: {trend.get('up_candles', 0)} | Down Candles: {trend.get('down_candles', 0)}")
        
        # Support/Resistance
        sr = advanced.get("support_resistance", {})
        if sr and sr.get("support_1"):
            lines.append(f"║     ")
            lines.append(f"║     🎯 Support & Resistance:")
            lines.append(f"║        • Resistance 1: {sr.get('resistance_1', 0):.6f}")
            lines.append(f"║        • Pivot: {sr.get('pivot', 0):.6f}")
            lines.append(f"║        • Support 1: {sr.get('support_1', 0):.6f}")
            dist_r = sr.get('distance_to_resistance', 0)
            dist_s = sr.get('distance_to_support', 0)
            lines.append(f"║        • Distance to Resistance: {dist_r:.6f} | Support: {dist_s:.6f}")
        
        # Candlestick Patterns
        patterns = advanced.get("patterns", [])
        if patterns:
            lines.append(f"║     ")
            lines.append(f"║     🕯️  Candlestick Patterns Detected:")
            for pattern in patterns:
                strength = pattern.get("strength", 0)
                strength_bar = "★" * strength
                lines.append(f"║        • {pattern.get('name', '?'):20} [{strength_bar:9}] Signal: {pattern.get('signal', '?')}")
                lines.append(f"║          └─ {pattern.get('description', '')}")
        
        # Fibonacci Retracement
        fib = advanced.get("fib_levels", {})
        if fib:
            lines.append(f"║     ")
            lines.append(f"║     📐 Fibonacci Retracement Levels:")
            for level, price_val in [("0%", fib.get("0%")), ("23.6%", fib.get("23.6%")), ("38.2%", fib.get("38.2%")), 
                                      ("50%", fib.get("50%")), ("61.8%", fib.get("61.8%")), ("78.6%", fib.get("78.6%")), 
                                      ("100%", fib.get("100%"))]:
                if price_val:
                    lines.append(f"║        • {level:6} → {price_val:.6f}")
        
        lines.append(f"║")
    
    lines.append(f"╚{'═'*150}╝\n")
    
    return "\n".join(lines)


def log_status_summary(statuses: Dict[str, Dict]) -> None:
    """Log a summary to stdout and logger."""
    output = format_all_pairs_status(statuses)
    logger.info(output)


def log_detailed_status(statuses: Dict[str, Dict]) -> None:
    """Log detailed status with all indicators."""
    output = format_detailed_status_with_advanced(statuses)
    logger.info(output)
