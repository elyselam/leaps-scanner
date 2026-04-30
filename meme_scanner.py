"""Meme / short-squeeze / unusual-volume scanner.

Designed to catch:
  - Unusual volume spikes (today vs 20d avg)
  - Short squeeze setups (high SI%, low days-to-cover, breaking out)
  - Meme momentum (consecutive up days, gap ups, range expansion)
  - WSB mention surges (ApeWisdom — free, no auth)

Three tiers:
  🚀 SQUEEZE   — 3+ signals stack incl. high SI + price action
  🔥 UNUSUAL   — clear volume surge + price/breakout confirmation
  👀 WATCH     — one strong signal, worth monitoring

Posts to #meme via MEME_WEBHOOK_URL.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from data import fetch_bars_batch
from gamma_exposure import compute_gex
from social_sources import fetch_stocktwits

APEWISDOM_URL    = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
APEWISDOM_TIMEOUT = 6

# Tier thresholds
SQUEEZE_MIN_SCORE = 8
UNUSUAL_MIN_SCORE = 5
WATCH_MIN_SCORE   = 3


# ── data class ───────────────────────────────────────────────────────────────

@dataclass
class MemeAlert:
    ticker:           str
    tier:             str         # "SQUEEZE" | "UNUSUAL" | "WATCH"
    score:            int
    price:            float
    reasons:          List[str] = field(default_factory=list)

    # volume
    vol_ratio:        float       = 1.0
    today_volume:     int         = 0
    avg_vol_20d:      int         = 0

    # short interest
    short_pct_float:  Optional[float] = None
    days_to_cover:    Optional[float] = None
    float_shares:     Optional[int]   = None

    # price action
    ret_1d:           Optional[float] = None
    ret_5d:           Optional[float] = None
    up_streak:        int             = 0

    # social — WSB (ApeWisdom)
    wsb_mentions_24h: Optional[int]   = None
    wsb_rank:         Optional[int]   = None
    wsb_mentions_change: Optional[float] = None   # pct change vs prior 24h

    # social — Stocktwits
    st_bull_count:        Optional[int]   = None
    st_bear_count:        Optional[int]   = None
    st_sentiment_score:   Optional[float] = None  # -1.0 .. +1.0
    st_message_velocity:  Optional[float] = None  # last-hour vs prior pace
    st_watchlist:         Optional[int]   = None
    st_top_message:       Optional[str]   = None

    # gamma exposure (Polygon GEX-style, computed from yfinance chain)
    gex_dollar:           Optional[float] = None
    gex_call_put_ratio:   Optional[float] = None
    gex_magnet_strike:    Optional[float] = None
    gex_magnet_pct:       Optional[float] = None  # (magnet - spot) / spot
    gex_setup:            bool             = False

    # composite — likelihood of rallying from here (NOT just "is it interesting")
    rally_score:          float            = 0.0


# ── ApeWisdom (WSB mentions) ─────────────────────────────────────────────────

_apewisdom_cache: Optional[Dict[str, Dict]] = None
_apewisdom_cache_ts: float = 0.0


def fetch_wsb_mentions() -> Dict[str, Dict]:
    """Fetch top WSB-mentioned tickers from ApeWisdom. Cached for 10min.

    Returns ticker -> {rank, mentions, mentions_24h_ago, change_pct}
    """
    global _apewisdom_cache, _apewisdom_cache_ts
    if _apewisdom_cache and (time.time() - _apewisdom_cache_ts) < 600:
        return _apewisdom_cache

    try:
        r = requests.get(APEWISDOM_URL, timeout=APEWISDOM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[meme] ApeWisdom fetch failed: {e}")
        return {}

    out: Dict[str, Dict] = {}
    for item in data.get("results", []):
        t = (item.get("ticker") or "").upper()
        if not t:
            continue
        mentions    = int(item.get("mentions") or 0)
        mentions_24 = int(item.get("mentions_24h_ago") or 0)
        change_pct  = ((mentions - mentions_24) / mentions_24 * 100) if mentions_24 else None
        out[t] = {
            "rank":        int(item.get("rank") or 0),
            "mentions":    mentions,
            "mentions_24": mentions_24,
            "change_pct":  change_pct,
        }
    _apewisdom_cache    = out
    _apewisdom_cache_ts = time.time()
    return out


# ── short interest (yfinance fundamentals) ───────────────────────────────────

def _fetch_short_interest(ticker: str) -> Dict[str, Optional[float]]:
    """Pull short-interest fields from yfinance .info.
    Returns dict with short_pct_float, days_to_cover, float_shares.
    """
    out = {"short_pct_float": None, "days_to_cover": None, "float_shares": None}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return out

    # yfinance keys (may not all be present per-ticker)
    spf = info.get("shortPercentOfFloat")
    if spf is not None:
        out["short_pct_float"] = float(spf)         # already a fraction (0.32 = 32%)
    dtc = info.get("shortRatio")
    if dtc is not None:
        out["days_to_cover"] = float(dtc)
    fs = info.get("floatShares") or info.get("sharesOutstanding")
    if fs is not None:
        out["float_shares"] = int(fs)
    return out


# ── signal computation ───────────────────────────────────────────────────────

def _compute_meme_signals(bars: pd.DataFrame) -> Dict:
    """Return dict of signals from price bars."""
    if bars.empty or len(bars) < 30:
        return {}

    close   = bars["Close"]
    open_   = bars["Open"]
    high    = bars["High"]
    low     = bars["Low"]
    volume  = bars["Volume"]

    last_close = float(close.iloc[-1])
    today_vol  = int(volume.iloc[-1])
    avg_vol20  = int(volume.iloc[-20:].mean())
    vol_ratio  = today_vol / avg_vol20 if avg_vol20 > 0 else 1.0

    # Returns
    ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else None
    ret_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else None

    # 20d high breakout
    recent_hi20 = float(high.iloc[-20:-1].max()) if len(high) > 20 else None
    breakout_20d = bool(recent_hi20 and last_close > recent_hi20)

    # 52w high proximity
    hi52 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    dist_from_high = float((hi52 - last_close) / hi52)

    # Gap
    gap_up = False
    if len(close) >= 2:
        gap = (open_.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        gap_up = bool(gap >= 0.03)   # ≥3% gap up — meme-tier

    # Up streak
    up_streak = 0
    for i in range(len(close) - 1, 0, -1):
        if close.iloc[i] > close.iloc[i - 1]:
            up_streak += 1
        else:
            break

    # Range expansion
    today_range  = (high.iloc[-1] - low.iloc[-1]) / last_close if last_close else 0
    avg_range_20 = ((high - low) / close).iloc[-20:].mean()
    range_exp    = float(today_range / avg_range_20) if avg_range_20 > 0 else 1.0

    # Today's candle direction
    today_green = bool(close.iloc[-1] > open_.iloc[-1])

    return {
        "price":          last_close,
        "today_volume":   today_vol,
        "avg_vol_20d":    avg_vol20,
        "vol_ratio":      vol_ratio,
        "ret_1d":         ret_1d,
        "ret_5d":         ret_5d,
        "breakout_20d":   breakout_20d,
        "dist_from_high": dist_from_high,
        "gap_up":         gap_up,
        "up_streak":      up_streak,
        "range_exp":      range_exp,
        "today_green":    today_green,
    }


# ── scoring ──────────────────────────────────────────────────────────────────

def _score_meme(sig: Dict, short: Dict, wsb: Optional[Dict],
                stocktwits: Optional[Dict] = None,
                gex: Optional[Dict] = None) -> Tuple[int, List[str], str]:
    """Return (score, reasons, tier)."""
    if not sig:
        return 0, [], ""

    score   = 0
    reasons: List[str] = []

    # ── VOLUME SURGE ──
    vr = sig.get("vol_ratio", 1.0)
    if   vr >= 5.0:
        score += 4; reasons.append(f"🔊🔊 volume {vr:.1f}x avg — extreme surge")
    elif vr >= 3.0:
        score += 3; reasons.append(f"🔊 volume {vr:.1f}x avg — heavy")
    elif vr >= 2.0:
        score += 2; reasons.append(f"🔊 volume {vr:.1f}x avg — elevated")
    elif vr >= 1.5:
        score += 1; reasons.append(f"volume {vr:.1f}x avg")

    # ── SHORT INTEREST (squeeze fuel) ──
    spf = short.get("short_pct_float")
    dtc = short.get("days_to_cover")
    if spf is not None:
        spf_pct = spf * 100
        if   spf_pct >= 30:
            score += 4; reasons.append(f"🩳 short interest {spf_pct:.0f}% of float — squeeze fuel")
        elif spf_pct >= 20:
            score += 3; reasons.append(f"🩳 short interest {spf_pct:.0f}% of float — heavy")
        elif spf_pct >= 10:
            score += 1; reasons.append(f"🩳 short interest {spf_pct:.0f}% of float")
    if dtc is not None:
        if   dtc >= 7:
            score += 2; reasons.append(f"⏳ {dtc:.1f} days to cover — shorts trapped")
        elif dtc >= 4:
            score += 1; reasons.append(f"⏳ {dtc:.1f} days to cover")

    # ── PRICE ACTION ──
    ret_1d = sig.get("ret_1d")
    if ret_1d is not None:
        if   ret_1d >= 0.15:
            score += 3; reasons.append(f"🚀 +{ret_1d*100:.1f}% today — parabolic")
        elif ret_1d >= 0.08:
            score += 2; reasons.append(f"🚀 +{ret_1d*100:.1f}% today — strong")
        elif ret_1d >= 0.04:
            score += 1; reasons.append(f"🚀 +{ret_1d*100:.1f}% today")

    ret_5d = sig.get("ret_5d")
    if ret_5d is not None and ret_5d >= 0.20:
        score += 2; reasons.append(f"🔥 +{ret_5d*100:.1f}% over 5d — hot run")
    elif ret_5d is not None and ret_5d >= 0.10:
        score += 1; reasons.append(f"🔥 +{ret_5d*100:.1f}% over 5d")

    if sig.get("breakout_20d") and sig.get("today_green"):
        score += 2; reasons.append("🎯 breaking above 20d high")

    if sig.get("gap_up"):
        score += 1; reasons.append("⬆️ gap up at open")

    up = sig.get("up_streak", 0)
    if up >= 5:
        score += 2; reasons.append(f"🔥 {up} green days in a row")
    elif up >= 3:
        score += 1; reasons.append(f"{up} green days in a row")

    re_exp = sig.get("range_exp", 1.0)
    if re_exp >= 2.0:
        score += 1; reasons.append(f"📏 range {re_exp:.1f}x avg — volatility blowout")

    dist_hi = sig.get("dist_from_high", 1.0)
    if dist_hi <= 0.02:
        score += 1; reasons.append("🚩 at 52w high")

    # ── WSB MENTIONS ──
    if wsb:
        m   = wsb.get("mentions", 0)
        ch  = wsb.get("change_pct")
        rk  = wsb.get("rank", 0)
        if rk and rk <= 10:
            score += 3; reasons.append(f"🦍 WSB rank #{rk} ({m} mentions/24h)")
        elif rk and rk <= 25:
            score += 2; reasons.append(f"🦍 WSB rank #{rk} ({m} mentions/24h)")
        elif rk and rk <= 50:
            score += 1; reasons.append(f"🦍 WSB rank #{rk}")
        if ch is not None and ch >= 100:
            score += 2; reasons.append(f"🦍 WSB mentions +{ch:.0f}% vs yesterday")
        elif ch is not None and ch >= 50:
            score += 1; reasons.append(f"🦍 WSB mentions +{ch:.0f}% vs yesterday")

    # ── STOCKTWITS ──
    if stocktwits:
        s_score = stocktwits.get("sentiment_score")
        s_vel   = stocktwits.get("message_velocity") or 1.0
        tagged  = stocktwits.get("tagged_total") or 0
        if tagged >= 5 and s_score is not None:
            if   s_score >=  0.6:
                score += 2; reasons.append(f"💬 Stocktwits {s_score*100:+.0f}% bull "
                                           f"({stocktwits.get('bull_count')}b/{stocktwits.get('bear_count')}b)")
            elif s_score >=  0.3:
                score += 1; reasons.append(f"💬 Stocktwits {s_score*100:+.0f}% bull")
            elif s_score <= -0.6:
                score += 2; reasons.append(f"💬 Stocktwits {s_score*100:+.0f}% bear "
                                           f"({stocktwits.get('bull_count')}b/{stocktwits.get('bear_count')}b)")
            elif s_score <= -0.3:
                score += 1; reasons.append(f"💬 Stocktwits {s_score*100:+.0f}% bear")
        if s_vel >= 3.0:
            score += 2; reasons.append(f"💬 Stocktwits chatter {s_vel:.1f}x normal pace")
        elif s_vel >= 2.0:
            score += 1; reasons.append(f"💬 Stocktwits chatter {s_vel:.1f}x normal pace")

    # ── GAMMA EXPOSURE ──
    if gex:
        cp     = gex.get("call_put_oi_ratio") or 0
        magnet = gex.get("magnet_distance_pct")
        if gex.get("gamma_setup"):
            score += 3
            mk = gex.get("max_oi_call_strike")
            reasons.append(f"⚡ gamma squeeze setup — call/put OI {cp:.1f}x, "
                           f"magnet ${mk:.0f} ({(magnet or 0)*100:+.1f}%)")
        else:
            if cp >= 3.0:
                score += 2; reasons.append(f"⚡ call OI {cp:.1f}x put OI — heavy call bias")
            elif cp >= 2.0:
                score += 1; reasons.append(f"⚡ call OI {cp:.1f}x put OI")
            if magnet is not None and 0 <= magnet <= 0.05:
                score += 1
                mk = gex.get("max_oi_call_strike")
                reasons.append(f"⚡ gamma magnet at ${mk:.0f} ({magnet*100:+.1f}% above spot)")

    # ── TIER ──
    has_short_signal = (spf is not None and spf >= 0.20) or (dtc is not None and dtc >= 5)
    has_price_signal = (ret_1d or 0) >= 0.04 or sig.get("breakout_20d") or up >= 3
    has_vol_signal   = vr >= 2.0

    if score >= SQUEEZE_MIN_SCORE and has_short_signal and (has_price_signal or has_vol_signal):
        tier = "SQUEEZE"
    elif score >= UNUSUAL_MIN_SCORE and has_vol_signal and has_price_signal:
        tier = "UNUSUAL"
    elif score >= WATCH_MIN_SCORE:
        tier = "WATCH"
    else:
        tier = ""

    return score, reasons, tier


# ── rally likelihood ─────────────────────────────────────────────────────────

def _rally_score(sig: Dict, short: Dict, wsb: Optional[Dict],
                 stocktwits: Optional[Dict], gex: Optional[Dict]) -> float:
    """Forward-looking probability that this name rallies from HERE.

    Differs from `_score_meme` in that it cares about *direction*. A name
    down 11% on heavy volume is "interesting" but not a rally setup —
    that's distribution. We want green-and-confirmed names with fuel.

    Rough scale 0..15. Negative possible if structure is bearish.
    """
    if not sig:
        return 0.0

    score = 0.0
    ret_1d   = sig.get("ret_1d") or 0.0
    vol_r    = sig.get("vol_ratio", 1.0)
    up_str   = sig.get("up_streak", 0)
    breakout = sig.get("breakout_20d")
    today_green = sig.get("today_green", False)

    # ── DIRECTIONAL BIAS (the thing CAR was missing) ──
    # Today's move drives the whole picture. Red day = penalize hard.
    if   ret_1d >=  0.10: score += 4.0
    elif ret_1d >=  0.05: score += 3.0
    elif ret_1d >=  0.02: score += 2.0
    elif ret_1d >=  0.00: score += 0.5
    elif ret_1d >= -0.02: score -= 1.0
    elif ret_1d >= -0.05: score -= 3.0
    else:                 score -= 5.0   # ≥5% red — falling knife

    # ── VOLUME × DIRECTION ──
    # High vol on a green day = accumulation. High vol on a red day = distribution.
    if vol_r >= 2.0:
        if today_green: score += 2.0
        else:           score -= 1.5
    elif vol_r >= 1.5:
        if today_green: score += 1.0
        else:           score -= 0.5

    # ── BREAKOUT IN PROGRESS ──
    if breakout and today_green:
        score += 2.0

    # ── SHORT-SQUEEZE FUEL — only counts when ALREADY moving up ──
    spf = short.get("short_pct_float") or 0
    dtc = short.get("days_to_cover")   or 0
    if ret_1d > 0 or up_str >= 2:
        if   spf >= 0.30: score += 3.0
        elif spf >= 0.20: score += 2.0
        elif spf >= 0.10: score += 1.0
        if dtc >= 7:      score += 1.0
        elif dtc >= 4:    score += 0.5
    else:
        # Heavy SI on a red day = squeeze fizzling, not building
        if spf >= 0.20:   score -= 0.5

    # ── UP-STREAK FRESHNESS (penalize parabolic exhaustion) ──
    if   up_str == 0:        pass
    elif up_str <= 2:        score += 0.5    # just starting
    elif up_str <= 4:        score += 1.5    # building momentum
    elif up_str <= 6:        score += 1.0    # mature run
    elif up_str <= 8:        score += 0.0    # late
    else:                    score -= 1.0    # too parabolic, mean-reversion risk

    # ── GAP UP ──
    if sig.get("gap_up"):
        score += 1.0

    # ── STOCKTWITS sentiment alignment ──
    if stocktwits:
        s_score = stocktwits.get("sentiment_score")
        s_vel   = stocktwits.get("message_velocity") or 1.0
        tagged  = stocktwits.get("tagged_total") or 0
        if tagged >= 5 and s_score is not None:
            # Aligned = sentiment matches today's direction
            if today_green and s_score >= 0.4:   score += 1.5
            elif today_green and s_score >= 0.2: score += 0.5
            elif not today_green and s_score >= 0.4:
                # bullish chatter despite red day — possibly still a setup
                score += 0.5
        if s_vel >= 3.0 and today_green:  score += 1.0
        elif s_vel >= 2.0 and today_green: score += 0.5

    # ── WSB MENTIONS ──
    if wsb:
        rk = wsb.get("rank") or 99
        ch = wsb.get("change_pct")
        if rk <= 10 and today_green:    score += 1.5
        elif rk <= 25 and today_green:  score += 0.5
        if ch is not None and ch >= 100 and today_green:
            score += 1.0

    # ── GAMMA EXPOSURE ──
    if gex:
        if gex.get("gamma_setup"):  # all 3 conditions met (cp_ratio≥2, magnet 0-10%, +GEX)
            score += 2.5
        else:
            cp     = gex.get("call_put_oi_ratio") or 0
            magnet = gex.get("magnet_distance_pct")
            if cp >= 3.0 and today_green:    score += 1.0
            elif cp >= 2.0 and today_green:  score += 0.5
            if magnet is not None and 0 <= magnet <= 0.05 and today_green:
                score += 1.0

    return score


# ── main scan ────────────────────────────────────────────────────────────────

def scan_meme(universe: List[str], dry_run: bool = False) -> List[MemeAlert]:
    """Scan universe for unusual volume / squeeze / meme setups."""
    print(f"[meme] scanning {len(universe)} tickers...")

    # Fetch bars in batch
    bars_map = fetch_bars_batch(universe, period="1y")

    # Fetch WSB mentions once (cached)
    wsb_data = fetch_wsb_mentions()
    if wsb_data:
        print(f"[meme] ApeWisdom: top {len(wsb_data)} WSB tickers loaded")

    # Add WSB-trending tickers that aren't already in our universe
    # (filter to plausible US tickers only — alphanumeric, ≤5 chars)
    extra_from_wsb = [
        t for t in list(wsb_data.keys())[:25]
        if t not in universe and t.isalpha() and len(t) <= 5
    ][:10]
    if extra_from_wsb:
        print(f"[meme] adding from WSB top-25: {', '.join(extra_from_wsb)}")
        extra_bars = fetch_bars_batch(extra_from_wsb, period="6mo")
        bars_map.update(extra_bars)
        scan_universe = universe + extra_from_wsb
    else:
        scan_universe = universe

    alerts: List[MemeAlert] = []
    for ticker in scan_universe:
        bars = bars_map.get(ticker)
        if bars is None or bars.empty or len(bars) < 30:
            continue

        sig = _compute_meme_signals(bars)
        if not sig:
            continue

        # Quick pre-filter: only fetch short interest if SOMETHING unusual is going on
        # (volume ≥ 1.5x OR 1d move ≥ 4% OR up_streak ≥ 3 OR breakout)
        worth_checking = (
            sig["vol_ratio"] >= 1.5
            or (sig.get("ret_1d") or 0) >= 0.04
            or sig.get("up_streak", 0) >= 3
            or sig.get("breakout_20d")
            or ticker in wsb_data
        )
        if not worth_checking:
            continue

        short = _fetch_short_interest(ticker)
        wsb   = wsb_data.get(ticker)

        # Stocktwits — pull for any ticker that already passed the worth_checking gate
        st = fetch_stocktwits(ticker, lookback_hours=24)

        # GEX — only compute when the bar is high enough to justify chain calls
        # (tickers showing real action: heavy vol, big move, or short-squeeze fuel)
        gex = None
        worth_gex = (
            sig["vol_ratio"] >= 2.0
            or (sig.get("ret_1d") or 0) >= 0.05
            or sig.get("breakout_20d")
            or (short.get("short_pct_float") or 0) >= 0.15
        )
        if worth_gex:
            gex = compute_gex(ticker, sig["price"])

        score, reasons, tier = _score_meme(sig, short, wsb, st, gex)
        if not tier:
            continue

        rally = _rally_score(sig, short, wsb, st, gex)

        alerts.append(MemeAlert(
            ticker             = ticker,
            tier               = tier,
            score              = score,
            price              = sig["price"],
            reasons            = reasons,
            vol_ratio          = sig["vol_ratio"],
            today_volume       = sig["today_volume"],
            avg_vol_20d        = sig["avg_vol_20d"],
            short_pct_float    = short.get("short_pct_float"),
            days_to_cover      = short.get("days_to_cover"),
            float_shares       = short.get("float_shares"),
            ret_1d             = sig.get("ret_1d"),
            ret_5d             = sig.get("ret_5d"),
            up_streak          = sig.get("up_streak", 0),
            wsb_mentions_24h   = wsb.get("mentions") if wsb else None,
            wsb_rank           = wsb.get("rank") if wsb else None,
            wsb_mentions_change= wsb.get("change_pct") if wsb else None,
            st_bull_count        = st.get("bull_count")        if st else None,
            st_bear_count        = st.get("bear_count")        if st else None,
            st_sentiment_score   = st.get("sentiment_score")   if st else None,
            st_message_velocity  = st.get("message_velocity")  if st else None,
            st_watchlist         = st.get("watchlist_count")   if st else None,
            st_top_message       = st.get("top_message")       if st else None,
            gex_dollar           = gex.get("dollar_gex")          if gex else None,
            gex_call_put_ratio   = gex.get("call_put_oi_ratio")   if gex else None,
            gex_magnet_strike    = gex.get("max_oi_call_strike")  if gex else None,
            gex_magnet_pct       = gex.get("magnet_distance_pct") if gex else None,
            gex_setup            = bool(gex and gex.get("gamma_setup")),
            rally_score          = rally,
        ))
        emoji = {"SQUEEZE": "🚀", "UNUSUAL": "🔥", "WATCH": "👀"}.get(tier, "")
        print(f"  {emoji} {ticker:6s}  {tier:8s}  score={score}  rally={rally:+.1f}  "
              f"vol={sig['vol_ratio']:.1f}x  1d={(sig.get('ret_1d') or 0)*100:+.1f}%  "
              f"SI={(short.get('short_pct_float') or 0)*100:.0f}%")

    # Rank by forward-looking rally likelihood (NOT just tier+score), then
    # break ties by raw score so SQUEEZE-tier wins close calls.
    alerts.sort(key=lambda a: (-a.rally_score, -a.score))
    return alerts
