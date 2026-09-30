"""Username/password auth for the I-Observe console.

Browser sessions use an HttpOnly ``iobserve_session`` cookie; API clients may
send the same JWT as ``Authorization: Bearer <token>``. Passwords are hashed
with PBKDF2-HMAC-SHA256 (stdlib only) and users live in ``data/users.sqlite``,
separate from the event timeline so auth never touches detection data.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "users.sqlite"
SECRET_PATH = DATA_DIR / ".auth_secret"

COOKIE_NAME = "iobserve_session"
TOKEN_TTL_S = 12 * 3600
PBKDF2_ITERATIONS = 220_000

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def secret_key() -> str:
    """HS256 signing key: env first, else a generated file key (stable restarts)."""
    env = (os.getenv("AUTH_SECRET") or "").strip()
    if env:
        return env
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SECRET_PATH.exists():
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    SECRET_PATH.write_text(key, encoding="utf-8")
    return key


def _db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.execute(
        "CREATE TABLE IF NOT EXISTS users("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "username TEXT UNIQUE NOT NULL,"
        "password_hash TEXT NOT NULL,"
        "created_at REAL NOT NULL)"
    )
    return db


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, ref: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = ref.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def user_count() -> int:
    db = _db()
    try:
        return int(db.execute("SELECT COUNT(*) FROM users").fetchone()[0])
    finally:
        db.close()


def create_user(username: str, password: str) -> str:
    """Register a user. Returns the canonical username. Raises ValueError."""
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-32 chars: letters, digits, . _ -")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    db = _db()
    try:
        exists = db.execute("SELECT 1 FROM users WHERE lower(username)=lower(?)", (username,)).fetchone()
        if exists:
            raise ValueError("Username is already taken")
        db.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
            (username, hash_password(password), time.time()),
        )
        db.commit()
        return username
    finally:
        db.close()


def verify_user(username: str, password: str) -> str | None:
    """Return the canonical username when credentials match, else None."""
    username = (username or "").strip()
    db = _db()
    try:
        row = db.execute(
            "SELECT username, password_hash FROM users WHERE lower(username)=lower(?)", (username,)
        ).fetchone()
    finally:
        db.close()
    if not row or not verify_password(password or "", row[1]):
        return None
    return row[0]


def make_token(username: str) -> str:
    from jose import jwt

    now = int(time.time())
    return jwt.encode({"sub": username, "iat": now, "exp": now + TOKEN_TTL_S},
                      secret_key(), algorithm="HS256")


def read_token(token: str | None) -> str | None:
    """Return the username for a valid non-expired JWT, else None."""
    if not token:
        return None
    try:
        from jose import jwt

        payload = jwt.decode(token, secret_key(), algorithms=["HS256"])
        sub = payload.get("sub")
        return str(sub) if sub else None
    except Exception:
        return None
