"""Data layer: yfinance primary, Polygon free tier optional supplement.

All functions swallow/propagate errors predictably so the scanner can degrade
gracefully (missing fundamentals shouldn't kill a trend-only scan).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ----------------------------- Equity bars ----------------------------------

def fetch_bars(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Return OHLCV DataFrame indexed by date. Empty DataFrame if unavailable."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # Flatten possible multiindex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=str.title)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(f"[data] bars fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_bars_batch(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Batch-download daily bars for multiple tickers. Returns dict[ticker]=df."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as e:
        print(f"[data] batch bars fetch failed: {e}")
        return {t: fetch_bars(t, period) for t in tickers}

    out: Dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        t = tickers[0]
        if raw is None or raw.empty:
            return {t: pd.DataFrame()}
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=str.title)
        out[t] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return out

    for t in tickers:
        try:
            df = raw[t].copy() if t in raw.columns.levels[0] else pd.DataFrame()
            if df.empty:
                out[t] = pd.DataFrame()
                continue
            df = df.rename(columns=str.title)
            out[t] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            out[t] = pd.DataFrame()
    return out


# ----------------------------- Options chain --------------------------------

@dataclass
class OptionsChain:
    ticker: str
    expiry: str           # YYYY-MM-DD
    dte: int
    spot: float
    calls: pd.DataFrame   # yfinance calls DataFrame
    puts: pd.DataFrame
    atm_iv: float         # mid of ATM call+put IV


def fetch_leaps_chain(
    ticker: str,
    min_dte: int = 300,
    max_dte: int = 600,
) -> Optional[OptionsChain]:
    """Fetch the closest-to-1yr-out expiry chain within [min_dte, max_dte]."""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        today = datetime.now().date()
        best = None
        best_dte = None
        for exp in exps:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                # Prefer the expiry closest to ~365 DTE
                if best is None or abs(dte - 365) < abs(best_dte - 365):
                    best = exp
                    best_dte = dte
        if best is None:
            return None

        ch = t.option_chain(best)
        calls = ch.calls.copy()
        puts = ch.puts.copy()
        if calls.empty or puts.empty:
            return None

        # Spot price
        spot = None
        try:
            info = t.fast_info
            spot = float(info.get("last_price") or info.get("lastPrice"))
        except Exception:
            pass
        if not spot:
            bars = fetch_bars(ticker, period="5d")
            if bars.empty:
                return None
            spot = float(bars["Close"].iloc[-1])

        # ATM IV = mid of nearest-to-spot call + put IV
        try:
            call_atm = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
            put_atm = puts.iloc[(puts["strike"] - spot).abs().argsort().iloc[0]]
            call_iv = float(call_atm.get("impliedVolatility") or 0.0)
            put_iv = float(put_atm.get("impliedVolatility") or 0.0)
            atm_iv = (call_iv + put_iv) / 2.0 if (call_iv and put_iv) else (call_iv or put_iv)
        except Exception:
            atm_iv = 0.0

        return OptionsChain(
            ticker=ticker,
            expiry=best,
            dte=best_dte,
            spot=float(spot),
            calls=calls,
            puts=puts,
            atm_iv=float(atm_iv),
        )
    except Exception as e:
        print(f"[data] options chain fetch failed for {ticker}: {e}")
        return None


def fetch_specific_option_price(
    ticker: str,
    expiry: str,
    strike: float,
    option_type: str = "call",
) -> Optional[Dict]:
    """Return bid/ask/mid for a specific contract (your actual position).
    Finds the closest listed strike if an exact match doesn't exist."""
    try:
        t = yf.Ticker(ticker)
        ch = t.option_chain(expiry)
        df = ch.calls.copy() if option_type.lower() == "call" else ch.puts.copy()
        if df.empty:
            return None
        row = df.iloc[(df["strike"] - strike).abs().argsort().iloc[0]]
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "strike": float(row["strike"]),
            "iv": float(row.get("impliedVolatility") or 0),
            "oi": int(row.get("openInterest") or 0),
        }
    except Exception as e:
        label = f"{strike:.0f}{option_type[0].upper()} {expiry}"
        print(f"[data] option price fetch failed for {ticker} {label}: {e}")
        return None


def fetch_front_month_iv(ticker: str) -> Optional[float]:
    """ATM IV for the nearest expiry (>=7 DTE). Used for term-structure check."""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None
        today = datetime.now().date()
        chosen = None
        for exp in exps:
            try:
                dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            except ValueError:
                continue
            if dte >= 7:
                chosen = exp
                break
        if not chosen:
            return None
        ch = t.option_chain(chosen)
        calls, puts = ch.calls, ch.puts
        if calls.empty or puts.empty:
            return None
        try:
            info = t.fast_info
            spot = float(info.get("last_price") or info.get("lastPrice"))
        except Exception:
            bars = fetch_bars(ticker, period="5d")
            spot = float(bars["Close"].iloc[-1]) if not bars.empty else None
        if not spot:
            return None
        ci = float(calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]].get("impliedVolatility") or 0)
        pi = float(puts.iloc[(puts["strike"] - spot).abs().argsort().iloc[0]].get("impliedVolatility") or 0)
        return (ci + pi) / 2.0 if (ci and pi) else (ci or pi)
    except Exception as e:
        print(f"[data] front-month IV fetch failed for {ticker}: {e}")
        return None


# ----------------------------- Fundamentals --------------------------------

@dataclass
class Fundamentals:
    free_cashflow: Optional[float]
    debt_to_equity: Optional[float]
    revenue_growth: Optional[float]
    short_pct_of_float: Optional[float]
    market_cap: Optional[float]
    sector: Optional[str]
    next_earnings: Optional[datetime]
    forward_div_yield: Optional[float]


def fetch_fundamentals(ticker: str) -> Fundamentals:
    """Best-effort fundamentals pull. All fields may be None."""
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.get_info() if hasattr(t, "get_info") else t.info
        except Exception:
            info = {}

        # Free cash flow (TTM)
        fcf = info.get("freeCashflow")
        # Debt/Equity
        de = info.get("debtToEquity")
        if de is not None:
            de = de / 100.0 if de > 5 else de  # yfinance sometimes reports as pct
        # Revenue growth YoY
        rev_growth = info.get("revenueGrowth")
        short_pct = info.get("shortPercentOfFloat")
        mcap = info.get("marketCap")
        sector = info.get("sector")
        div_yield = info.get("dividendYield")

        # Next earnings date
        next_earn = None
        try:
            cal = t.calendar
            if cal is not None and not (hasattr(cal, "empty") and cal.empty):
                if isinstance(cal, dict):
                    val = cal.get("Earnings Date")
                    if isinstance(val, list) and val:
                        next_earn = pd.Timestamp(val[0]).to_pydatetime()
                else:
                    # Old-style DataFrame
                    try:
                        val = cal.loc["Earnings Date"].iloc[0]
                        next_earn = pd.Timestamp(val).to_pydatetime()
                    except Exception:
                        pass
        except Exception:
            pass

        return Fundamentals(
            free_cashflow=fcf,
            debt_to_equity=de,
            revenue_growth=rev_growth,
            short_pct_of_float=short_pct,
            market_cap=mcap,
            sector=sector,
            next_earnings=next_earn,
            forward_div_yield=div_yield,
        )
    except Exception as e:
        print(f"[data] fundamentals fetch failed for {ticker}: {e}")
        return Fundamentals(None, None, None, None, None, None, None, None)


# ----------------------------- Polygon (optional) ---------------------------

class PolygonClient:
    """Throttled Polygon free-tier client (5 calls/min). Optional supplement."""

    BASE = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None, calls_per_min: int = 5):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.interval = 60.0 / max(1, calls_per_min)
        self._last = 0.0

    def enabled(self) -> bool:
        return bool(self.api_key)

    def _throttle(self):
        dt = time.time() - self._last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self._last = time.time()

    def daily_bars(self, ticker: str, days: int = 365) -> pd.DataFrame:
        if not self.enabled():
            return pd.DataFrame()
        end = datetime.now().date()
        start = end - timedelta(days=days)
        url = (
            f"{self.BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        self._throttle()
        try:
            r = requests.get(url, params={"apiKey": self.api_key, "adjusted": "true"}, timeout=15)
            r.raise_for_status()
            data = r.json().get("results") or []
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df = df.set_index("date")
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            print(f"[polygon] daily bars failed for {ticker}: {e}")
            return pd.DataFrame()
