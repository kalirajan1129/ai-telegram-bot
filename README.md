# 🤖 AI Trading Telegram Bot

Multi-timeframe ensemble ML trading signal bot for Forex, Crypto & OTC pairs.
Streams live Binance data, runs Random Forest + XGBoost + LightGBM + CatBoost predictions,
and sends high-confidence signals to your Telegram channel.

---

## 📁 Project Structure

```
Telegram bot/
├── main.py                 ← Entry point — run this
├── config.py               ← Root config (also at backend/core/config.py)
├── database.py             ← SQLite + Parquet storage layer
├── technical.py            ← All TA indicators (pure NumPy/Pandas)
├── requirements.txt        ← Python dependencies
├── .env.example            ← Copy to .env and fill values
│
├── backend/core/
│   └── config.py           ← Pydantic settings (loaded from .env)
│
├── bot/
│   ├── handlers.py         ← Telegram command & callback handlers
│   └── notifier.py         ← Signal formatter & Telegram sender
│
├── data_feed/
│   └── binance_ws.py       ← Binance WebSocket + REST candle feed
│
├── ml/
│   └── trainer.py          ← Ensemble model trainer & predictor
│
└── signals/
    └── engine.py           ← Signal generation & pair scanner
```

---

## ⚙️ Setup

### 1. Prerequisites

- **Python 3.11+**
- A **Telegram Bot Token** (from [@BotFather](https://t.me/BotFather))
- Your **Telegram Chat ID** (from [@userinfobot](https://t.me/userinfobot))

### 2. Create virtual environment

```bash
cd "Telegram bot"
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
python -m pip install --upgrade pip setuptools wheel
```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `catboost` and `vectorbt` can be slow to install.
> If you don't need them, comment them out in `requirements.txt`.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:
```
TELEGRAM_BOT_TOKEN=7123456789:AAFxxx...
TELEGRAM_CHAT_ID=-1001234567890
```

To get your **Chat ID**:
1. Add your bot to a channel/group
2. Send a message
3. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Copy the `chat.id` value

---

## 🚀 Running the Bot

```bash
python main.py
```

The bot will:
1. ✅ Initialise the SQLite database
2. 📥 Download historical candles from Binance
3. 🌐 Start live WebSocket streams
4. 🔍 Scan all pairs for signals
5. 📲 Send signals to your Telegram channel

---

## 📲 Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome menu with quick-access buttons |
| `/stats` | Overall win rate & performance |
| `/signals` | Last 10 generated signals |
| `/pairs` | Top pairs by win rate |
| `/status` | Bot uptime & config info |
| `/help` | Show all commands |

**Admin only:**
| Command | Description |
|---------|-------------|
| `/retrain` | Trigger ML model retrain |
| `/pause` | Pause signal generation |
| `/resume` | Resume signal generation |

To make yourself an admin, add your Telegram User ID to `.env`:
```
TELEGRAM_ADMIN_IDS=[YOUR_USER_ID]
```

---

## 🧠 How It Works

1. **Data Feed** — Binance WebSocket streams real-time OHLCV candles for all crypto pairs
2. **Feature Engineering** — 60+ technical indicators computed per candle (RSI, MACD, BB, ADX, etc.)
3. **ML Ensemble** — 4 models (RF, XGBoost, LightGBM, CatBoost) vote on BUY/SELL
4. **Multi-Timeframe** — Signal only fires when multiple timeframes agree (configurable)
5. **Risk Filters** — ADX trend filter + ATR volatility range check
6. **Signal Dispatch** — Formatted signal with entry, target & stop-loss sent to Telegram

---

## 🔧 Configuration

Key settings in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_CONFIDENCE` | `85.0` | Minimum ML confidence % to send a signal |
| `MIN_AGREEING_TIMEFRAMES` | `4` | Timeframes that must agree |
| `MAX_SIGNALS_PER_HOUR` | `10` | Anti-spam per pair |
| `ADX_MIN_TREND` | `25.0` | Minimum trend strength |
| `RETRAIN_INTERVAL_HOURS` | `6` | Auto-retrain frequency |

---

## ⚠️ Disclaimer

This bot is for **educational and research purposes only**.
Trading involves risk. This is **not financial advice**.
Always use proper risk management.

---

## 🛠 Troubleshooting

**Bot not sending messages?**
- Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
- Make sure the bot is an admin in your channel

**No signals generated?**
- Lower `MIN_CONFIDENCE` temporarily to test (e.g. `60.0`)
- Lower `MIN_AGREEING_TIMEFRAMES` to `2`
- Check that candles are being fetched: look at logs

**Import errors?**
- Make sure you're running from inside the `Telegram bot/` directory
- Run: `pip install -r requirements.txt` again

**CatBoost/LightGBM install fails?**
- Comment them out in `requirements.txt` — the bot works without them
