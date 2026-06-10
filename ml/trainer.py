"""
ML Trainer — ensemble of RandomForest, XGBoost, LightGBM.
"""
from __future__ import annotations

import asyncio
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import settings
from technical import FEATURE_COLUMNS, build_feature_vector

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


MODELS_DIR = settings.MODELS_DIR


def _path(name: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"{name}.pkl"


def generate_labels(df: pd.DataFrame, lookahead: int = 3, threshold: float = 0.0015) -> pd.Series:
    future_max = df["close"].shift(-1).rolling(lookahead).max().shift(-(lookahead - 1))
    future_min = df["close"].shift(-1).rolling(lookahead).min().shift(-(lookahead - 1))
    up = (future_max - df["close"]) / df["close"] >= threshold
    dn = (df["close"] - future_min) / df["close"] >= threshold
    labels = pd.Series(np.nan, index=df.index)
    labels[up & ~dn] = 1
    labels[dn & ~up] = 0
    return labels


class EnsembleTrainer:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.scaler: Optional[StandardScaler] = None
        self._load()

    def _load(self):
        sp = _path("scaler")
        if sp.exists():
            try:
                self.scaler = joblib.load(sp)
            except Exception:
                pass
        for name in ["random_forest", "xgboost", "lightgbm"]:
            p = _path(name)
            if p.exists():
                try:
                    self.models[name] = joblib.load(p)
                    logger.info("Loaded: {}", name)
                except Exception:
                    pass

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        df = build_feature_vector(df.copy())
        df["label"] = generate_labels(df)
        df.dropna(subset=FEATURE_COLUMNS + ["label"], inplace=True)
        if len(df) < settings.MIN_SAMPLES_FOR_TRAINING:
            raise ValueError(f"Need {settings.MIN_SAMPLES_FOR_TRAINING} samples, got {len(df)}")
        X = df[FEATURE_COLUMNS].values
        y = df["label"].values.astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)
        joblib.dump(self.scaler, _path("scaler"))
        metrics = {}
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42, class_weight="balanced")
        rf.fit(X_tr_s, y_tr)
        self.models["random_forest"] = rf
        joblib.dump(rf, _path("random_forest"))
        metrics["rf_acc"] = round(accuracy_score(y_te, rf.predict(X_te_s)), 4)
        if HAS_XGB:
            m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric="logloss", random_state=42)
            m.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
            self.models["xgboost"] = m
            joblib.dump(m, _path("xgboost"))
            metrics["xgb_acc"] = round(accuracy_score(y_te, m.predict(X_te_s)), 4)
        if HAS_LGB:
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
            m.fit(X_tr_s, y_tr)
            self.models["lightgbm"] = m
            joblib.dump(m, _path("lightgbm"))
            metrics["lgb_acc"] = round(accuracy_score(y_te, m.predict(X_te_s)), 4)
        logger.info("Training done: {}", metrics)
        return metrics

    def predict_proba(self, X: np.ndarray) -> float:
        if self.scaler is None or not self.models:
            return 0.5
        X_s = self.scaler.transform(X.reshape(1, -1))
        weights = {"random_forest": 0.35, "xgboost": 0.40, "lightgbm": 0.25}
        total_w, weighted = 0.0, 0.0
        for name, model in self.models.items():
            w = weights.get(name, 0)
            try:
                prob = model.predict_proba(X_s)[0][1]
                weighted += w * prob
                total_w += w
            except Exception:
                pass
        return weighted / total_w if total_w > 0 else 0.5


_trainer: Optional[EnsembleTrainer] = None

def get_trainer() -> EnsembleTrainer:
    global _trainer
    if _trainer is None:
        _trainer = EnsembleTrainer()
    return _trainer

def retrain_all() -> dict:
    from database import load_candles
    from config import CRYPTO_PAIRS
    logger.info("Starting retrain...")
    trainer = get_trainer()
    frames = []
    for pair in CRYPTO_PAIRS[:10]:
        df = load_candles(pair, settings.PRIMARY_TIMEFRAME, limit=5000)
        if not df.empty:
            frames.append(df)
    if not frames:
        logger.warning("No data for retrain.")
        return {}
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    try:
        metrics = trainer.train(combined)
        logger.info("Retrain done: {}", metrics)
        return metrics
    except Exception as e:
        logger.error("Retrain failed: {}", e)
        return {}
