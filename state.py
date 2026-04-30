"""SQLite-backed state: ATM IV history for IVR, and alert dedupe."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "leaps_state.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS iv_history(
        ticker TEXT, date TEXT, atm_iv REAL,
        PRIMARY KEY(ticker, date))""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts(
        ticker TEXT, setup_hash TEXT, ts TEXT,
        PRIMARY KEY(ticker, setup_hash))""")
    return c


def record_iv(ticker: str, atm_iv: float) -> None:
    if not atm_iv or atm_iv <= 0:
        return
    d = datetime.now().date().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO iv_history(ticker, date, atm_iv) VALUES(?,?,?)",
            (ticker, d, float(atm_iv)),
        )


def load_iv_history(ticker: str) -> pd.Series:
    with _conn() as c:
        rows = c.execute(
            "SELECT date, atm_iv FROM iv_history WHERE ticker=? ORDER BY date",
            (ticker,),
        ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name=ticker)


def was_recently_alerted(ticker: str, setup_hash: str, hours: int) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT ts FROM alerts WHERE ticker=? AND setup_hash=?",
            (ticker, setup_hash),
        ).fetchone()
    if not row:
        return False
    ts = datetime.fromisoformat(row[0])
    return datetime.now() - ts < timedelta(hours=hours)


def mark_alerted(ticker: str, setup_hash: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO alerts(ticker, setup_hash, ts) VALUES(?,?,?)",
            (ticker, setup_hash, datetime.now().isoformat()),
        )


def compute_setup_hash(tier: str, leaps_strike: Optional[float], leaps_expiry: Optional[str]) -> str:
    key = f"{tier}|{leaps_strike}|{leaps_expiry}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def compute_channel_hash(channel: str, *parts) -> str:
    """Channel-prefixed alert fingerprint.

    Example: same ticker can show up in both `weekly` and `market` channels
    with the same strike/expiry — without a channel prefix, posting to one
    would suppress the other. The channel string keeps them in separate
    dedup namespaces while sharing one alerts table.

    Args:
        channel: e.g. "weekly", "market", "meme"
        parts:   anything stable that defines "same setup" — None values OK
    """
    key = "|".join([channel] + [str(p) for p in parts])
    return hashlib.md5(key.encode()).hexdigest()[:12]
