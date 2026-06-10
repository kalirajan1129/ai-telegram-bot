"""
Central configuration for the AI Trading Platform.
Supports Crypto (Binance), Forex (Twelve Data), and OTC (mirrored) pairs.
"""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "ICT Signals"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Paths ────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "backend" / "data"
    MODELS_DIR: Path = BASE_DIR / "backend" / "models"
    LOGS_DIR: Path = BASE_DIR / "backend" / "logs"
    DB_PATH: Path = BASE_DIR / "backend" / "data" / "trading.db"
    PARQUET_DIR: Path = BASE_DIR / "backend" / "data" / "parquet"
    JSON_DIR: Path = BASE_DIR / "backend" / "data" / "json"

    # ── Telegram ─────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TELEGRAM_ADMIN_IDS: list[int] = []

    # ── Twelve Data (Multiple API Keys) ──────────────────
    TWELVE_DATA_API_KEYS: str = ""  # Comma-separated API keys
    TWELVE_DATA_API_KEY: str = ""  # Single key (backward compatibility)

    # ── Enabled Pairs (Selective Monitoring) ─────────────
    ENABLED_FOREX_PAIRS: str = ""   # Comma-separated. Empty = all forex pairs
    ENABLED_CRYPTO_PAIRS: str = ""  # Comma-separated. Empty = limited crypto pairs
    ENABLED_OTC_PAIRS: str = ""     # Comma-separated OTC pairs for signal generation

    # ── Signal Settings ──────────────────────────────────
    MIN_CONFIDENCE: float = 65.0
    SIGNAL_EXPIRY_SECONDS: int = 300
    MAX_SIGNALS_PER_HOUR: int = 10

    # ── Signal Thresholds for Indicators ─────────────────
    SIGNAL_STRENGTH_BUY: float = 70.0
    SIGNAL_STRENGTH_SELL: float = 65.0
    SIGNAL_STRENGTH_STRONG: float = 85.0

    # ── Timeframes ───────────────────────────────────────
    TIMEFRAMES: list[str] = [
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "4h",
    ]

    PRIMARY_TIMEFRAME: str = "5m"
    MIN_AGREEING_TIMEFRAMES: int = 2
    FOREX_TIMEFRAMES: str = "5m,15m,30m,1h,4h"  # Comma-separated forex timeframes

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

    # ── Binance ──────────────────────────────────────────
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/ws"
    BINANCE_REST_URL: str = "https://api.binance.com"

    # ── Forex Feed ───────────────────────────────────────
    FOREX_POLL_INTERVAL: int = 10
    FOREX_CANDLE_REFRESH: int = 300

    # ── API ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # ── News Filter ──────────────────────────────────────
    NEWS_BUFFER_MINUTES: int = 30

    HIGH_IMPACT_EVENTS: list[str] = [
        "NFP",
        "CPI",
        "FOMC",
        "GDP",
        "Interest Rate",
        "Employment",
    ]

    # ── Risk Settings ────────────────────────────────────
    ADX_MIN_TREND: float = 20.0
    VOLATILITY_MIN: float = 0.0003
    VOLATILITY_MAX: float = 0.08

    @field_validator("TELEGRAM_ADMIN_IDS", mode="before")
    @classmethod
    def parse_admin_ids(cls, v):
        if isinstance(v, int):
            return [v]
        elif isinstance(v, str) and v:
            try:
                return [int(id.strip()) for id in v.split(",") if id.strip()]
            except (ValueError, AttributeError):
                return []
        elif isinstance(v, list):
            return [int(x) for x in v]
        return []
    
    def create_dirs(self):
        for directory in [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.PARQUET_DIR,
            self.JSON_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_api_keys_list(self) -> list[str]:
        """Get list of API keys from TWELVE_DATA_API_KEYS or fallback to single key."""
        if self.TWELVE_DATA_API_KEYS:
            return [k.strip() for k in self.TWELVE_DATA_API_KEYS.split(",") if k.strip()]
        elif self.TWELVE_DATA_API_KEY:
            return [self.TWELVE_DATA_API_KEY]
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
        if self.FOREX_TIMEFRAMES:
            return [t.strip() for t in self.FOREX_TIMEFRAMES.split(",") if t.strip()]
        return ["5m", "15m", "1h"]


settings = Settings()
settings.create_dirs()

# ── Supported Pairs ────────────────────────────

FOREX_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "AUDNZD",
    "GBPAUD",
    "GBPCAD",
    "EURNZD",
    "EURAUD",
    "CADJPY",
    "CHFJPY",
    "GBPNZD",
    "GBPCHF",
    "CADCHF",
    "NZDCHF",
    "NZDCAD",
    "NZDJPY",
    "USDTRY",
    "USDZAR",
    "USDMXN",
    "USDSGD",
    "USDNOK",
    "USDSEK",
]

CRYPTO_PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "FILUSDT",
    "ETCUSDT",
    "AAVEUSDT",
    "MKRUSDT",
]

OTC_PAIRS = [
    "EURUSD_OTC",
    "GBPUSD_OTC",
    "USDJPY_OTC",
    "AUDUSD_OTC",
    "USDCAD_OTC",
    "EURGBP_OTC",
    "EURJPY_OTC",
    "GBPJPY_OTC",
    "AUDNZD_OTC",
    "NZDUSD_OTC",
    "BTCUSDT_OTC",
    "ETHUSDT_OTC",
    "BNBUSDT_OTC",
    "XRPUSDT_OTC",
]

ALL_PAIRS = FOREX_PAIRS + CRYPTO_PAIRS + OTC_PAIRS