"""Password, opaque-session-token, and sign-in throttling helpers."""

from __future__ import annotations

import bcrypt
import hashlib
import secrets
import threading
import time
from collections import OrderedDict, deque


SESSION_COOKIE_NAME = "rag_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
PASSWORD_BCRYPT_ROUNDS = 12
SIGNIN_LIMIT = 5
SIGNIN_WINDOW_SECONDS = 15 * 60

_DUMMY_PASSWORD_HASH: bytes | None = None
_DUMMY_HASH_LOCK = threading.Lock()


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=PASSWORD_BCRYPT_ROUNDS),
    ).decode("ascii")


def verify_password(password: str, encoded_hash: str | None) -> bool:
    global _DUMMY_PASSWORD_HASH
    try:
        password_bytes = password.encode("utf-8")
        if len(password_bytes) > 72:
            return False
        if encoded_hash is None:
            with _DUMMY_HASH_LOCK:
                if _DUMMY_PASSWORD_HASH is None:
                    _DUMMY_PASSWORD_HASH = bcrypt.hashpw(
                        b"not-a-real-user-password",
                        bcrypt.gensalt(rounds=PASSWORD_BCRYPT_ROUNDS),
                    )
            target_hash = _DUMMY_PASSWORD_HASH
        else:
            target_hash = encoded_hash.encode("ascii")
        return bcrypt.checkpw(password_bytes, target_hash)
    except (ValueError, TypeError, UnicodeEncodeError):
        return False


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SignInRateLimiter:
    """Small process-local fixed-window limiter keyed by (email, client IP)."""

    def __init__(
        self,
        *,
        limit: int = SIGNIN_LIMIT,
        window_seconds: int = SIGNIN_WINDOW_SECONDS,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._attempts: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _active_attempts(self, key: tuple[str, str], now: float) -> deque[float]:
        attempts = self._attempts.setdefault(key, deque())
        while attempts and now - attempts[0] >= self.window_seconds:
            attempts.popleft()
        if not attempts:
            self._attempts.pop(key, None)
            return deque()
        self._attempts.move_to_end(key)
        return attempts

    def is_limited(self, key: tuple[str, str], *, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            return len(self._active_attempts(key, timestamp)) >= self.limit

    def record_failure(self, key: tuple[str, str], *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            attempts = self._active_attempts(key, timestamp)
            attempts.append(timestamp)
            self._attempts[key] = attempts
            self._attempts.move_to_end(key)
            while len(self._attempts) > self.max_keys:
                self._attempts.popitem(last=False)

    def clear(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._attempts.pop(key, None)


signin_rate_limiter = SignInRateLimiter()