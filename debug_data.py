"""
Debug script — check what data is in the database.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from pathlib import Path
from config import settings
from database import load_candles
from technical import rsi, adx, atr

# Check what pairs/files exist
parquet_dir = settings.PARQUET_DIR
print(f"Parquet directory: {parquet_dir}\n")

if parquet_dir.exists():
    for pair_dir in sorted(parquet_dir.iterdir()):
        if pair_dir.is_dir():
            pair = pair_dir.name
            print(f"📁 {pair}:")
            for tf_file in sorted(pair_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(tf_file)
                    print(f"   {tf_file.stem}: {len(df)} candles")
                    if len(df) > 0:
                        print(f"      Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
                except Exception as e:
                    print(f"   {tf_file.stem}: ERROR - {e}")
            print()
else:
    print("❌ Parquet directory doesn't exist!")
    sys.exit(1)

print("\n" + "="*80)
print("DETAILED CANDLE CHECK FOR EURUSD 5m")
print("="*80 + "\n")

# Load EURUSD 5m data
df = load_candles("EURUSD", "5m", limit=500)
print(f"Loaded {len(df)} candles for EURUSD 5m\n")

if len(df) > 0:
    print("Sample data:")
    print(df.head())
    print("\n")
    print(df.tail())
    print("\n")
    
    # Check for NaN values
    print(f"NaN values: {df.isna().sum().sum()}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Try to calculate indicators
    print("Testing indicator calculations:")
    
    try:
        rsi_val = rsi(df["close"], period=14)
        print(f"✓ RSI: {rsi_val.iloc[-1]:.2f}")
    except Exception as e:
        print(f"✗ RSI error: {e}")
    
    try:
        adx_val, _, _ = adx(df["high"], df["low"], df["close"])
        print(f"✓ ADX: {adx_val.iloc[-1]:.2f}")
    except Exception as e:
        print(f"✗ ADX error: {e}")
    
    try:
        atr_val = atr(df["high"], df["low"], df["close"])
        print(f"✓ ATR: {atr_val.iloc[-1]:.6f}")
    except Exception as e:
        print(f"✗ ATR error: {e}")
else:
    print("❌ No candles found for EURUSD 5m!")
