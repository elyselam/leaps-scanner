"""Market-wide scanner — S&P 500 + Nasdaq-100 prescreen → deep weekly scan.

Why this exists:
  The regular weekly scanner runs against ~85 hand-picked names, so anything
  outside that list (mid-caps, idiosyncratic movers, sector rotations) is
  invisible. This module casts a wider net.

Pipeline:
  1. Build universe   — S&P 500 + Nasdaq-100 from Wikipedia, cached daily
  2. Cheap prescreen  — batch fetch 1mo bars for ~550 names, compute volume
                        ratio / 1d return / 5d return / breakout / range
                        expansion, keep only "interesting" survivors
  3. Deep scan        — hand survivors to weekly_scanner.scan_weekly() which
                        does the option-chain + news + scoring work
  4. Top 10 calls     — main.py applies the same calls-only / top-10 cap
                        used by the regular weekly scanner

Universe size: ~550 unique tickers (S&P 500 has ~500, Nasdaq-100 overlaps
heavily but adds ~50 names like ASML, MELI, MSTR, ARM, etc.)

Cache: daily JSON at market_universe_cache.json. Refresh on schema mismatch
or after 24h. Wikipedia is the source of truth for both indices and updates
membership in real time as additions/deletions happen.
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from data import fetch_bars_batch
from weekly_scanner import scan_weekly, WeeklyAlert

_WIKI_HEADERS = {
    # Wikipedia 403s pandas's default UA — needs a browser-ish identifier
    "User-Agent": "Mozilla/5.0 leaps-scanner/1.0 (contact: yplam90@gmail.com)",
}
_WIKI_TIMEOUT = 15


def _read_wiki_tables(url: str) -> List[pd.DataFrame]:
    """Fetch a Wikipedia page and parse all <table> elements into DataFrames."""
    r = requests.get(url, headers=_WIKI_HEADERS, timeout=_WIKI_TIMEOUT)
    r.raise_for_status()
    # pd.read_html on a raw string is deprecated — wrap in StringIO
    return pd.read_html(io.StringIO(r.text))

HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(HERE, "market_universe_cache.json")
_CACHE_TTL_SECONDS = 24 * 3600

SP500_URL    = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


# ── universe builders ────────────────────────────────────────────────────────

def _normalize_ticker(t: str) -> str:
    """Wikipedia uses 'BRK.B'; yfinance wants 'BRK-B'. Same for BF.B etc."""
    return t.strip().upper().replace(".", "-")


def _fetch_sp500() -> List[str]:
    """Pull S&P 500 constituents from Wikipedia."""
    try:
        tables = _read_wiki_tables(SP500_URL)
    except Exception as e:
        print(f"[market] S&P 500 fetch failed: {e}")
        return []
    # First table on the page is the constituents
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return [_normalize_ticker(t) for t in df[col].dropna().tolist()]


def _fetch_nasdaq100() -> List[str]:
    """Pull Nasdaq-100 constituents from Wikipedia."""
    try:
        tables = _read_wiki_tables(NASDAQ100_URL)
    except Exception as e:
        print(f"[market] Nasdaq-100 fetch failed: {e}")
        return []
    # The constituents table varies in position year-to-year; find the one
    # with a Ticker/Symbol column
    for df in tables:
        cols = [c for c in df.columns if str(c).lower() in ("ticker", "symbol")]
        if cols and len(df) >= 90:   # Nasdaq-100 should have ~100 rows
            return [_normalize_ticker(t) for t in df[cols[0]].dropna().tolist()]
    print("[market] Nasdaq-100 — couldn't find constituents table")
    return []


def _build_universe() -> List[str]:
    """Combine indices and dedupe, preserving insertion order."""
    sp500     = _fetch_sp500()
    nasdaq100 = _fetch_nasdaq100()
    seen, out = set(), []
    for t in sp500 + nasdaq100:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def get_market_universe(force_refresh: bool = False) -> List[str]:
    """Return cached universe if fresh, else refresh from Wikipedia."""
    if not force_refresh and os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH) as f:
                cache = json.load(f)
            age = time.time() - cache.get("ts", 0)
            if age < _CACHE_TTL_SECONDS and cache.get("tickers"):
                print(f"[market] using cached universe ({len(cache['tickers'])} tickers, "
                      f"age={age/3600:.1f}h)")
                return cache["tickers"]
        except Exception as e:
            print(f"[market] cache read failed: {e}")

    print("[market] building universe from Wikipedia (S&P 500 + Nasdaq-100)...")
    tickers = _build_universe()
    if not tickers:
        # Fall back to stale cache rather than empty — better something than nothing
        if os.path.exists(_CACHE_PATH):
            try:
                with open(_CACHE_PATH) as f:
                    cache = json.load(f)
                if cache.get("tickers"):
                    print(f"[market] Wikipedia failed, using stale cache "
                          f"({len(cache['tickers'])} tickers)")
                    return cache["tickers"]
            except Exception:
                pass
        return []

    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump({"ts": time.time(), "tickers": tickers}, f)
    except Exception as e:
        print(f"[market] cache write failed: {e}")

    print(f"[market] universe built: {len(tickers)} unique tickers")
    return tickers


# ── prescreen ────────────────────────────────────────────────────────────────

def _prescreen_signals(bars: pd.DataFrame) -> Optional[Dict]:
    """Cheap signals from bars only — no API calls. Returns None if too short."""
    if bars is None or bars.empty or len(bars) < 25:
        return None

    close   = bars["Close"]
    open_   = bars["Open"]
    high    = bars["High"]
    low     = bars["Low"]
    volume  = bars["Volume"]

    last_close = float(close.iloc[-1])
    today_vol  = int(volume.iloc[-1])
    avg_vol20  = int(volume.iloc[-20:].mean())
    vol_ratio  = today_vol / avg_vol20 if avg_vol20 > 0 else 1.0

    ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0.0
    ret_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0.0

    recent_hi20 = float(high.iloc[-20:-1].max()) if len(high) > 20 else None
    breakout_20d = bool(recent_hi20 and last_close > recent_hi20)

    today_range  = (high.iloc[-1] - low.iloc[-1]) / last_close if last_close else 0
    avg_range_20 = ((high - low) / close).iloc[-20:].mean()
    range_exp    = float(today_range / avg_range_20) if avg_range_20 > 0 else 1.0

    today_green = bool(close.iloc[-1] > open_.iloc[-1])

    return {
        "price":        last_close,
        "vol_ratio":    vol_ratio,
        "ret_1d":       ret_1d,
        "ret_5d":       ret_5d,
        "breakout_20d": breakout_20d,
        "range_exp":    range_exp,
        "today_green":  today_green,
    }


def _is_interesting(sig: Dict) -> Tuple[bool, str]:
    """Filter rule: keep names where SOMETHING noteworthy is happening today.

    We're prescreening for the deep weekly scan, which is calls-only and
    looks for bullish setups. So we bias toward up-side tells:

      - heavy volume (regardless of direction — could be accumulation or
        capitulation flushing the float)
      - sharp 1d move (any direction; reversal candidates count)
      - 20d-high breakout on a green day
      - volatility blowout on a green day
      - 5d momentum already running

    Returns (keep, one-line reason).
    """
    if sig["vol_ratio"] >= 2.5:
        return True, f"vol {sig['vol_ratio']:.1f}x avg"
    if sig["ret_1d"] >= 0.05:
        return True, f"+{sig['ret_1d']*100:.1f}% today"
    if sig["ret_1d"] <= -0.05 and sig["vol_ratio"] >= 1.5:
        return True, f"{sig['ret_1d']*100:.1f}% today on {sig['vol_ratio']:.1f}x vol"
    if sig["breakout_20d"] and sig["today_green"]:
        return True, "20d breakout, green"
    if sig["range_exp"] >= 2.0 and sig["today_green"]:
        return True, f"range {sig['range_exp']:.1f}x avg"
    if sig["ret_5d"] >= 0.10 and sig["today_green"]:
        return True, f"+{sig['ret_5d']*100:.1f}% over 5d"
    return False, ""


# ── main ─────────────────────────────────────────────────────────────────────

def scan_market(dry_run: bool = False,
                max_survivors: int = 60,
                min_score: int = 7,
                news_cap: int = 50) -> List[WeeklyAlert]:
    """Run the market-wide scan: prescreen ~550 → deep scan survivors.

    `max_survivors` caps how many names enter the deep scan. The deep scan
    fetches news + chains per ticker and the slowest leg is Polygon news at
    12.5s/req. 60 tickers ≈ 12 minutes worst-case for Polygon, but parallel
    sources and the inner technical filter inside scan_weekly trim it down.

    Tuning vs the curated weekly scanner:
      - `min_score=7` (vs default 8): mid-cap S&P/Nasdaq names often have
        sparse Polygon/Finnhub coverage. Without news points, even a clean
        technical setup tops out at 6-7. Drop the bar by 1 so genuine bullish
        setups aren't filtered out by missing news data.
      - `news_cap=50` (vs default 25): the prescreen already trims ~550 to
        ~60. The default cap of 25 silently drops half the survivors before
        they even get news enrichment.
    """
    universe = get_market_universe()
    if not universe:
        print("[market] empty universe — aborting")
        return []

    # Need ≥25 bars for the 20d rolling stats — yfinance "1mo" returns only
    # ~20 trading days, so use 3mo to be safe (still cheap as a single batch)
    print(f"[market] prescreen: fetching 3mo bars for {len(universe)} tickers...")
    bars_map = fetch_bars_batch(universe, period="3mo")

    survivors: List[Tuple[str, Dict, str]] = []
    for ticker in universe:
        bars = bars_map.get(ticker)
        sig = _prescreen_signals(bars)
        if not sig:
            continue
        keep, reason = _is_interesting(sig)
        if keep:
            survivors.append((ticker, sig, reason))

    # Rank survivors by a composite "noteworthiness" so the cap picks the
    # most interesting ones if we're over `max_survivors`. Score weights
    # today's move heavily (forward-looking) and adds volume confirmation.
    def _rank(item):
        _, s, _ = item
        score = 0.0
        score += abs(s["ret_1d"]) * 100               # any sharp move
        if s["today_green"]: score += s["ret_1d"] * 50   # bonus for green
        score += min(s["vol_ratio"], 10) * 2           # vol surge, capped
        if s["breakout_20d"] and s["today_green"]: score += 5
        if s["range_exp"] >= 2.0: score += 2
        return -score

    survivors.sort(key=_rank)
    if len(survivors) > max_survivors:
        print(f"[market] {len(survivors)} survivors — trimming to top {max_survivors} by noteworthiness")
        survivors = survivors[:max_survivors]
    else:
        print(f"[market] {len(survivors)} survivors out of {len(universe)} ({len(survivors)/len(universe)*100:.1f}%)")

    # Show a peek at what made it through
    for ticker, sig, reason in survivors[:15]:
        print(f"  ✓ {ticker:6s}  ${sig['price']:>8.2f}  "
              f"1d={sig['ret_1d']*100:+5.1f}%  vol={sig['vol_ratio']:>4.1f}x  — {reason}")
    if len(survivors) > 15:
        print(f"  ... and {len(survivors) - 15} more")

    if not survivors:
        return []

    # Hand the survivor list to the regular weekly scanner — it does the
    # option chain + news + scoring + dual-expiry work for us.
    survivor_tickers = [s[0] for s in survivors]
    print(f"[market] handing {len(survivor_tickers)} survivors to weekly deep-scan "
          f"(min_score={min_score}, news_cap={news_cap})...")
    alerts = scan_weekly(
        survivor_tickers,
        dry_run=dry_run,
        news_cap=news_cap,
        min_score=min_score,
    )
    print(f"[market] funnel: universe={len(universe)} → "
          f"prescreen_survivors={len(survivor_tickers)} → "
          f"emitted={len(alerts)} (CALL={sum(1 for a in alerts if a.direction == 'CALL')}, "
          f"PUT={sum(1 for a in alerts if a.direction == 'PUT')})")
    return alerts
