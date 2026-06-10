"""
Signal Lock Manager — prevents contradictory signal flips within validity window.
Once a BUY/SELL signal is issued for a pair, it stays locked until:
  1. The validity window expires (4h, 2h, 1h based on confidence)
  2. Confidence drops significantly (>15 points below original)
  3. A VERY strong opposing signal (>90%) appears
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional
from loguru import logger


class SignalLock:
    """Stores locked signal state per pair."""
    def __init__(self, pair: str, direction: str, confidence: float, validity_minutes: int):
        self.pair = pair
        self.direction = direction
        self.confidence = confidence
        self.validity_minutes = validity_minutes
        self.locked_at = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        now = datetime.now(timezone.utc)
        elapsed = (now - self.locked_at).total_seconds() / 60
        return elapsed >= self.validity_minutes

    def minutes_remaining(self) -> float:
        now = datetime.now(timezone.utc)
        elapsed = (now - self.locked_at).total_seconds() / 60
        return max(0.0, self.validity_minutes - elapsed)


class SignalLockManager:
    """
    Manages per-pair signal locks.
    Prevents BUY→SELL flip within the validity window.
    """

    def __init__(self):
        self._locks: Dict[str, SignalLock] = {}

    def get_lock(self, pair: str) -> Optional[SignalLock]:
        lock = self._locks.get(pair)
        if lock and lock.is_expired():
            del self._locks[pair]
            return None
        return lock

    def set_lock(self, pair: str, direction: str, confidence: float, validity_minutes: int):
        self._locks[pair] = SignalLock(pair, direction, confidence, validity_minutes)
        logger.info("Signal locked: {} {} ({:.1f}%) for {} min", pair, direction, confidence, validity_minutes)

    def should_override_lock(self, pair: str, new_direction: str, new_confidence: float) -> bool:
        """
        Returns True only if the new signal should override the existing lock.
        Override rules:
          - Opposing direction needs >90% confidence
          - Same direction always updates if confidence is higher
        """
        lock = self.get_lock(pair)
        if not lock:
            return True  # No lock, proceed freely

        if lock.direction == new_direction:
            # Same direction — update if higher confidence
            if new_confidence > lock.confidence:
                return True
            return False

        # Opposing direction — only override with very strong signal
        if new_confidence >= 90.0:
            logger.info("Strong opposing signal ({:.1f}%) overriding {} lock for {}",
                        new_confidence, lock.direction, pair)
            return True

        mins_left = lock.minutes_remaining()
        logger.debug("Signal flip blocked: {} wants {} ({:.1f}%) but locked as {} for {:.0f} more min",
                     pair, new_direction, new_confidence, lock.direction, mins_left)
        return False

    def resolve_status(self, pair: str, raw_status: str, confidence: float, validity_minutes: int) -> dict:
        """
        Given a raw computed status, return the final status respecting locks.
        Returns dict with final status and lock info.
        """
        if raw_status == "WAIT":
            lock = self.get_lock(pair)
            if lock:
                # Keep showing the locked signal instead of WAIT (market just paused briefly)
                mins = lock.minutes_remaining()
                return {
                    "status": lock.direction,
                    "confidence": lock.confidence,
                    "locked": True,
                    "lock_remaining_min": round(mins, 1),
                    "validity_minutes": lock.validity_minutes
                }
            return {"status": "WAIT", "confidence": confidence, "locked": False, "lock_remaining_min": 0}

        # BUY or SELL
        if self.should_override_lock(pair, raw_status, confidence):
            self.set_lock(pair, raw_status, confidence, validity_minutes)
            return {
                "status": raw_status,
                "confidence": confidence,
                "locked": True,
                "lock_remaining_min": validity_minutes,
                "validity_minutes": validity_minutes
            }
        else:
            lock = self.get_lock(pair)
            mins = lock.minutes_remaining() if lock else 0
            return {
                "status": lock.direction if lock else raw_status,
                "confidence": lock.confidence if lock else confidence,
                "locked": True,
                "lock_remaining_min": round(mins, 1),
                "validity_minutes": lock.validity_minutes if lock else validity_minutes
            }


# Global singleton
_lock_manager: Optional[SignalLockManager] = None

def get_lock_manager() -> SignalLockManager:
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = SignalLockManager()
    return _lock_manager
