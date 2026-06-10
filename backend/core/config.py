"""
Central configuration for the AI Trading Platform.
All settings are loaded from environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "AI Trading Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Paths ────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    DB_PATH: Path = BASE_DIR / "data" / "trading.db"
    PARQUET_DIR: Path = BASE_DIR / "data" / "parquet"
    JSON_DIR: Path = BASE_DIR / "data" / "json"

    # ── Telegram ─────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TELEGRAM_ADMIN_IDS: list[int] = []

    # ── Signal Settings ──────────────────────────────────
    MIN_CONFIDENCE: float = 85.0
    SIGNAL_EXPIRY_SECONDS: int = 300
    MAX_SIGNALS_PER_HOUR: int = 10

    # ── Timeframes (seconds) ─────────────────────────────
    TIMEFRAMES: list[str] = ["5m", "15m", "30m", "1h", "4h"]
    PRIMARY_TIMEFRAME: str = "5m"
    MIN_AGREEING_TIMEFRAMES: int = 2

    # ── ML Settings ──────────────────────────────────────
    RETRAIN_INTERVAL_HOURS: int = 6
    MIN_SAMPLES_FOR_TRAINING: int = 500
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    ENSEMBLE_WEIGHTS: dict = {
        "random_forest": 0.25,
        "xgboost": 0.30,
        "lightgbm": 0.30,
        "catboost": 0.15,
    }

    # ── Binance WebSocket ────────────────────────────────
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/ws"
    BINANCE_REST_URL: str = "https://api.binance.com"

    # ── API Keys & External Services ─────────────────────
    TWELVE_DATA_API_KEYS: str = ""  # Comma-separated API keys for rotation
    
    # ── Pair Selection for Monitoring ────────────────────
    ENABLED_FOREX_PAIRS: str = ""  # Comma-separated. If empty, all forex pairs enabled
    ENABLED_CRYPTO_PAIRS: str = ""  # Comma-separated. If empty, all crypto pairs enabled
    ENABLED_OTC_PAIRS: str = ""    # Comma-separated OTC pairs for signal generation
    
    # ── Forex Settings ───────────────────────────────────
    FOREX_POLL_INTERVAL: int = 10
    FOREX_CANDLE_REFRESH: int = 300
    FOREX_TIMEFRAMES: str = "5m,15m,1h"

    # ── API ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000","http://localhost:5173"]

    # ── News Filter ──────────────────────────────────────
    NEWS_BUFFER_MINUTES: int = 30
    HIGH_IMPACT_EVENTS: list[str] = ["NFP","CPI","FOMC","GDP","Interest Rate","Employment"]

    # ── Risk Settings ────────────────────────────────────
    ADX_MIN_TREND: float = 25.0
    VOLATILITY_MIN: float = 0.0005
    VOLATILITY_MAX: float = 0.05
    
    # ── Signal Thresholds for Different Timeframes ───────
    SIGNAL_STRENGTH_BUY: float = 70.0   # Minimum score for BUY signal
    SIGNAL_STRENGTH_SELL: float = 65.0  # Minimum score for SELL signal
    SIGNAL_STRENGTH_STRONG: float = 85.0  # Strong signal threshold

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("TELEGRAM_ADMIN_IDS", mode="before")
    @classmethod
    def parse_admin_ids(cls, v):
        if isinstance(v, str):
            try:
                return [int(id.strip()) for id in v.split(",") if id.strip()]
            except ValueError:
                return []
        elif isinstance(v, int):
            return [v]
        elif isinstance(v, list):
            return v
        return []

    def get_api_keys_list(self) -> list[str]:
        """Get list of available API keys."""
        if self.TWELVE_DATA_API_KEYS:
            return [k.strip() for k in self.TWELVE_DATA_API_KEYS.split(",") if k.strip()]
        return []

    def get_enabled_forex_pairs(self) -> list[str]:
        """Get list of enabled forex pairs."""
        if self.ENABLED_FOREX_PAIRS:
            return [p.strip() for p in self.ENABLED_FOREX_PAIRS.split(",") if p.strip()]
        return []

    def get_enabled_crypto_pairs(self) -> list[str]:
        """Get list of enabled crypto pairs."""
        if self.ENABLED_CRYPTO_PAIRS:
            return [p.strip() for p in self.ENABLED_CRYPTO_PAIRS.split(",") if p.strip()]
        return []

    def get_enabled_otc_pairs(self) -> list[str]:
        """Get list of enabled OTC pairs."""
        if self.ENABLED_OTC_PAIRS:
            return [p.strip() for p in self.ENABLED_OTC_PAIRS.split(",") if p.strip()]
        return []

    def get_forex_timeframes(self) -> list[str]:
        """Get list of forex timeframes."""
        if hasattr(self, 'FOREX_TIMEFRAMES') and self.FOREX_TIMEFRAMES:
            return [t.strip() for t in self.FOREX_TIMEFRAMES.split(",") if t.strip()]
        return ["5m", "15m", "1h"]

    def create_dirs(self):
        for d in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR,
                  self.PARQUET_DIR, self.JSON_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_dirs()

# ── Supported Pairs ───────────────────────────────────────────────────────────
FOREX_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD",
    "EURGBP","EURJPY","GBPJPY","AUDJPY","AUDNZD","GBPAUD","GBPCAD",
    "EURNZD","EURAUD","CADJPY","CHFJPY","EURCZK","EURDKK","EURHUF",
    "EURPLN","EURSEK","EURMXN","USDRUB","USDTRY","USDZAR","USDSGD",
    "USDHKD","USDMXN","USDPLN","USDSEK","USDDKK","USDNOK","USDCZK",
    "GBPNZD","GBPCHF","CADCHF","NZDCHF","NZDCAD","NZDJPY",
]

CRYPTO_PAIRS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT","LINKUSDT","LTCUSDT",
    "UNIUSDT","ATOMUSDT","NEARUSDT","FTMUSDT","ALGOUSDT","VETUSDT",
    "MANAUSDT","SANDUSDT","AXSUSDT","TRXUSDT","XLMUSDT","ETCUSDT",
    "FILUSDT","THETAUSDT","EGLDUSDT","HBARUSDT","ICPUSDT","FLOWUSDT",
    "AAVEUSDT","MKRUSDT","COMPUSDT","SNXUSDT","CRVUSDT","SUSHIUSDT",
]

OTC_PAIRS = [
    "EURUSD_OTC","GBPUSD_OTC","USDJPY_OTC","AUDUSD_OTC","USDCAD_OTC",
    "EURGBP_OTC","EURJPY_OTC","GBPJPY_OTC","AUDNZD_OTC","NZDUSD_OTC",
    "BTCUSDT_OTC","ETHUSDT_OTC","BNBUSDT_OTC","XRPUSDT_OTC",
]

ALL_PAIRS = FOREX_PAIRS + CRYPTO_PAIRS + OTC_PAIRS
