"""Weekly options scanner.

Every run scans BOTH expiries:
  - This Friday       (0–4 DTE depending on day of week)
  - Following Friday  (7–11 DTE)

The same ticker can appear in both buckets — the setups differ because the
earnings-vs-expiry check and the liquidity/chain available are different.

Signals are momentum-focused (not LEAPS value logic):
  - trend alignment (20d / 50d SMA)
  - MACD crossover / histogram direction
  - RSI in the right zone
  - 5-day price + RS momentum
  - volume expansion
  - proximity to 52w high (calls) or 52w low (puts)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from data import fetch_bars_batch, fetch_fundamentals
from news_sources import fetch_all_news

# ── minimum score to generate an alert ──────────────────────────────────────
MIN_SCORE = 8          # raised — richer signal set makes easy points


# ── keyword-based news sentiment (simple but effective) ─────────────────────
BULLISH_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "jumps", "jump", "soar",
    "upgrade", "upgraded", "raise", "raised", "raises", "acquire", "acquires",
    "acquisition", "partnership", "record", "strong", "bullish", "outperform",
    "outperforms", "buy", "accumulate", "breakout", "expand", "expands",
    "growth", "profit", "profitable", "wins", "approved", "approval",
    "launches", "launch", "tops", "tops estimates", "beats estimates",
    "boost", "boosts", "milestone", "crushes",
}
BEARISH_WORDS = {
    "miss", "misses", "plunge", "plunges", "drops", "drop", "fall", "falls",
    "downgrade", "downgraded", "cut", "cuts", "sell", "bearish",
    "underperform", "warn", "warns", "warning", "lawsuit", "sued",
    "investigation", "probe", "recall", "fraud", "bankruptcy", "debt",
    "layoffs", "fired", "resigns", "resignation", "scandal", "loss",
    "losses", "declines", "decline", "weak", "slashes", "slash",
    "guides lower", "guidance lower", "delay", "delays",
}


def _score_headlines(headlines: List[str]) -> float:
    """Return sentiment score in [-1.0, +1.0] from a list of headlines.
    Simple keyword match weighted by position (earlier = more recent = more weight).
    """
    if not headlines:
        return 0.0
    total_pos, total_neg, total_weight = 0.0, 0.0, 0.0
    for idx, head in enumerate(headlines[:15]):
        weight = 1.0 / (1 + idx * 0.25)   # decay: 1.0, 0.8, 0.67, 0.57, …
        words  = head.lower().replace(",", " ").replace(".", " ").split()
        pos = sum(1 for w in words if w in BULLISH_WORDS)
        neg = sum(1 for w in words if w in BEARISH_WORDS)
        total_pos    += pos * weight
        total_neg    += neg * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    raw = (total_pos - total_neg) / max(total_weight, 1.0)
    # clip to [-1, 1]
    return max(-1.0, min(1.0, raw))


# News window for weekly scanner — tight because weeklies react to current flow.
# User override: set NEWS_HOURS_BACK in .env to widen.
_NEWS_HOURS = int(os.getenv("NEWS_HOURS_BACK", "48"))


def _fetch_news_and_earnings(ticker: str, expiry: str) -> Dict:
    """Fetch recent news + next earnings date for a single ticker.

    News is pulled from Polygon.io + Finnhub (+ yfinance fallback) via
    news_sources.fetch_all_news, with a 48h window by default.

    Final news_sentiment blends:
      - headline keyword score  (our BULLISH/BEARISH word match, time-decayed)
      - Finnhub pre-computed score  (from their own NLP, if available)
      - Polygon per-article sentiment  (if any articles have it)

    Returns dict with news_sentiment, news_hot, news_top_headline, headlines,
    news_sources, news_article_count, days_to_earnings, earnings_before_expiry.
    """
    out: Dict = {
        "news_sentiment":         None,
        "news_hot":               False,
        "news_top_headline":      None,
        "headlines":              [],
        "news_sources":           [],
        "news_article_count":     0,
        "days_to_earnings":       None,
        "earnings_before_expiry": False,
    }
    try:
        # ── news (multi-source) ──
        news = fetch_all_news(ticker, hours_back=_NEWS_HOURS)
        articles    = news["articles"]
        headlines   = [a["title"] for a in articles]
        fh_sent     = news.get("finnhub_sentiment")

        # Blend sentiment signals
        sub_scores: List[float] = []

        if headlines:
            sub_scores.append(_score_headlines(headlines))

        if fh_sent is not None:
            sub_scores.append(float(fh_sent["score"]))

        # Polygon per-article sentiment — average across articles that have it
        poly_scored = [a["sentiment"] for a in articles if a.get("sentiment") is not None]
        if poly_scored:
            sub_scores.append(sum(poly_scored) / len(poly_scored))

        if sub_scores:
            out["news_sentiment"] = max(-1.0, min(1.0, sum(sub_scores) / len(sub_scores)))

        if headlines:
            out["headlines"]         = headlines
            out["news_top_headline"] = headlines[0][:80]

        out["news_hot"]            = news.get("is_hot", False)
        out["news_sources"]        = news.get("sources_used", [])
        out["news_article_count"]  = news.get("article_count", 0)

        # ── earnings ──
        fund = fetch_fundamentals(ticker)
        if fund.next_earnings:
            dte_e = (fund.next_earnings.date() - date.today()).days
            out["days_to_earnings"] = int(dte_e)
            exp_d = date.fromisoformat(expiry)
            days_to_exp = (exp_d - date.today()).days
            out["earnings_before_expiry"] = bool(0 <= dte_e <= days_to_exp)
    except Exception as e:
        print(f"[weekly] news/earnings fetch failed for {ticker}: {e}")

    return out


# ── data class ───────────────────────────────────────────────────────────────

@dataclass
class WeeklyAlert:
    ticker:      str
    direction:   str          # "CALL" | "PUT"
    strike:      float
    expiry:      str          # YYYY-MM-DD
    dte:         int
    bid:         float
    ask:         float
    mid:         float        # suggested entry price
    stock_price: float
    reasons:     List[str] = field(default_factory=list)
    score:       int = 0
    spread_pct:  float = 0.0
    oi:          int = 0

    # catalyst context
    days_to_earnings:       Optional[int] = None
    earnings_before_expiry: bool          = False
    news_sentiment:         Optional[float] = None
    news_top_headline:      Optional[str]   = None
    news_hot:               bool            = False
    headlines:              List[str]       = field(default_factory=list)
    news_sources:           List[str]       = field(default_factory=list)
    news_article_count:     int             = 0


# ── expiry logic ─────────────────────────────────────────────────────────────

def _friday(ref: date, offset_weeks: int = 0) -> date:
    """Return the Friday of the current week (+offset_weeks)."""
    days_to_fri = (4 - ref.weekday()) % 7   # 0 if already Friday
    return ref + timedelta(days=days_to_fri + offset_weeks * 7)


def get_target_expiries(ref: date | None = None) -> List[Tuple[str, str, int]]:
    """Return BOTH target expiries as a list of (expiry_YYYY-MM-DD, mode, dte).

    Always returns two entries:
      [(this_friday,   "THIS_WEEK", dte), (next_friday,   "NEXT_WEEK", dte)]

    If today is Friday, "this Friday" is 0 DTE (same day). The chain lookup
    will naturally filter that out if no tradeable contracts remain.
    """
    ref = ref or date.today()
    this_fri = _friday(ref, offset_weeks=0)
    next_fri = _friday(ref, offset_weeks=1)
    return [
        (this_fri.strftime("%Y-%m-%d"), "THIS_WEEK",  (this_fri - ref).days),
        (next_fri.strftime("%Y-%m-%d"), "NEXT_WEEK",  (next_fri - ref).days),
    ]


# Backwards-compat shim (main.py still calls this; returns the nearer expiry)
def get_target_expiry(ref: date | None = None) -> Tuple[str, str, int]:
    return get_target_expiries(ref)[0]


# ── signal computation ────────────────────────────────────────────────────────

def _compute_signals(bars: pd.DataFrame, spy_bars: pd.DataFrame) -> Dict:
    """Compute momentum / trend signals from daily bars."""
    if bars.empty or len(bars) < 60:
        return {}

    close  = bars["Close"]
    open_  = bars["Open"]
    high   = bars["High"]
    low    = bars["Low"]
    volume = bars["Volume"]

    last_close = float(close.iloc[-1])

    # Moving averages
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else None
    dist_vs_20d  = float((last_close - sma20.iloc[-1])  / sma20.iloc[-1])  if not np.isnan(sma20.iloc[-1])  else None
    dist_vs_50d  = float((last_close - sma50.iloc[-1])  / sma50.iloc[-1])  if not np.isnan(sma50.iloc[-1])  else None
    dist_vs_200d = (float((last_close - sma200.iloc[-1]) / sma200.iloc[-1])
                    if sma200 is not None and not np.isnan(sma200.iloc[-1]) else None)
    above_20d  = dist_vs_20d  is not None and dist_vs_20d  > 0
    above_50d  = dist_vs_50d  is not None and dist_vs_50d  > 0
    above_200d = dist_vs_200d is not None and dist_vs_200d > 0
    stacked_bull = bool(above_20d and above_50d and above_200d
                        and sma20.iloc[-1] > sma50.iloc[-1])
    stacked_bear = bool((not above_20d) and (not above_50d)
                        and sma20.iloc[-1] < sma50.iloc[-1])

    # 52-week range
    hi52 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    lo52 = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    dist_from_high = float((hi52 - last_close) / hi52)
    dist_from_low  = float((last_close - lo52) / lo52)

    # 20-day high/low breakout
    recent_hi20 = float(high.iloc[-20:-1].max()) if len(high) > 20 else None
    recent_lo20 = float(low.iloc[-20:-1].min())  if len(low)  > 20 else None
    breakout_20d_high = bool(recent_hi20 and last_close > recent_hi20)
    breakdown_20d_low = bool(recent_lo20 and last_close < recent_lo20)

    # RSI-14
    delta_p = close.diff()
    gain    = delta_p.clip(lower=0).rolling(14).mean()
    loss    = (-delta_p.clip(upper=0)).rolling(14).mean()
    rsi_raw = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi     = float(rsi_raw.iloc[-1]) if not np.isnan(rsi_raw.iloc[-1]) else None
    rsi_prev = float(rsi_raw.iloc[-2]) if len(rsi_raw) >= 2 and not np.isnan(rsi_raw.iloc[-2]) else None
    rsi_crossed_50_up   = bool(rsi is not None and rsi_prev is not None and rsi_prev < 50 <= rsi)
    rsi_crossed_50_down = bool(rsi is not None and rsi_prev is not None and rsi_prev > 50 >= rsi)

    # MACD (12/26/9)
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    h = hist.dropna()
    macd_bullish    = bool(len(h) >= 2 and h.iloc[-1] > 0 and h.iloc[-1] > h.iloc[-2])
    macd_bearish    = bool(len(h) >= 2 and h.iloc[-1] < 0 and h.iloc[-1] < h.iloc[-2])
    macd_bull_cross = bool(len(h) >= 2 and macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] >= signal.iloc[-1])
    macd_bear_cross = bool(len(h) >= 2 and macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] <= signal.iloc[-1])
    macd_hist_accel_up   = bool(len(h) >= 3 and h.iloc[-1] > h.iloc[-2] > h.iloc[-3])
    macd_hist_accel_down = bool(len(h) >= 3 and h.iloc[-1] < h.iloc[-2] < h.iloc[-3])

    # ATR-style volatility expansion: today's range vs 20d avg range
    today_range  = float((high.iloc[-1] - low.iloc[-1]) / last_close)
    avg_range_20 = float(((high - low) / close).iloc[-20:].mean())
    range_expansion = float(today_range / avg_range_20) if avg_range_20 > 0 else 1.0

    # Volume
    avg_vol20 = float(volume.iloc[-20:].mean())
    vol_ratio = float(volume.iloc[-1] / avg_vol20) if avg_vol20 > 0 else 1.0

    # Today's candle
    today_green = bool(close.iloc[-1] > open_.iloc[-1])
    today_red   = bool(close.iloc[-1] < open_.iloc[-1])

    # Gap up/down (open today vs prior close)
    gap_up   = False
    gap_down = False
    if len(close) >= 2:
        gap = (open_.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        gap_up   = bool(gap >=  0.01)   # ≥1% gap up
        gap_down = bool(gap <= -0.01)

    # Consecutive up/down days
    up_streak   = 0
    down_streak = 0
    for i in range(len(close) - 1, 0, -1):
        if close.iloc[i] > close.iloc[i - 1]:
            if down_streak: break
            up_streak += 1
        elif close.iloc[i] < close.iloc[i - 1]:
            if up_streak: break
            down_streak += 1
        else:
            break

    # Returns
    ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else None
    ret_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else None

    # Relative strength vs SPY
    rs_5d  = None
    rs_20d = None
    if not spy_bars.empty and ret_5d is not None:
        try:
            spy_5d  = float(spy_bars["Close"].iloc[-1] / spy_bars["Close"].iloc[-5]  - 1)
            rs_5d   = ret_5d - spy_5d
            if len(spy_bars) > 20 and len(close) > 20:
                ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1)
                spy_20d = float(spy_bars["Close"].iloc[-1] / spy_bars["Close"].iloc[-20] - 1)
                rs_20d  = ret_20d - spy_20d
        except Exception:
            pass

    return {
        "price":              last_close,
        "dist_vs_20d":        dist_vs_20d,
        "dist_vs_50d":        dist_vs_50d,
        "dist_vs_200d":       dist_vs_200d,
        "above_20d":          above_20d,
        "above_50d":          above_50d,
        "above_200d":         above_200d,
        "stacked_bull":       stacked_bull,
        "stacked_bear":       stacked_bear,
        "dist_from_high":     dist_from_high,
        "dist_from_low":      dist_from_low,
        "breakout_20d_high":  breakout_20d_high,
        "breakdown_20d_low":  breakdown_20d_low,
        "rsi":                rsi,
        "rsi_crossed_50_up":  rsi_crossed_50_up,
        "rsi_crossed_50_down": rsi_crossed_50_down,
        "macd_bullish":       macd_bullish,
        "macd_bearish":       macd_bearish,
        "macd_bull_cross":    macd_bull_cross,
        "macd_bear_cross":    macd_bear_cross,
        "macd_hist_accel_up":   macd_hist_accel_up,
        "macd_hist_accel_down": macd_hist_accel_down,
        "range_expansion":    range_expansion,
        "vol_ratio":          vol_ratio,
        "today_green":        today_green,
        "today_red":          today_red,
        "gap_up":             gap_up,
        "gap_down":           gap_down,
        "up_streak":          up_streak,
        "down_streak":        down_streak,
        "ret_1d":             ret_1d,
        "ret_5d":             ret_5d,
        "rs_5d":              rs_5d,
        "rs_20d":             rs_20d,
    }


# ── scoring ───────────────────────────────────────────────────────────────────

def _score(sig: Dict, context: Optional[Dict] = None) -> Tuple[str, int, List[str]]:
    """Score ticker for bullish (CALL) or bearish (PUT) weekly play.
    Returns (direction, score, reasons).

    context: optional dict with keys
      - days_to_earnings (int | None)  — None if unknown
      - earnings_before_expiry (bool)  — earnings fall before the weekly expiry
      - news_sentiment (float | None)  — -1.0 to +1.0
      - news_hot (bool)                — elevated headline volume
      - news_top_headline (str | None)
    """
    if not sig:
        return "", 0, []

    ctx       = context or {}
    rsi       = sig.get("rsi")
    vol_ratio = sig.get("vol_ratio", 1.0)
    ret_5d    = sig.get("ret_5d")
    rs_5d     = sig.get("rs_5d")
    rs_20d    = sig.get("rs_20d")

    call_score, call_reasons = 0, []
    put_score,  put_reasons  = 0, []

    # ══ TREND ══════════════════════════════════════════════════════════════
    if sig.get("stacked_bull"):
        call_score += 3
        call_reasons.append("📈 trend stacked bull (20d > 50d > 200d)")
    elif sig.get("above_50d"):
        d50 = sig.get("dist_vs_50d") or 0
        call_score += 2
        call_reasons.append(f"📈 +{d50*100:.1f}% above 50d SMA")
    if sig.get("above_200d") and not sig.get("stacked_bull"):
        call_score += 1
        call_reasons.append("📈 above 200d SMA (long-term bull)")

    if sig.get("stacked_bear"):
        put_score += 3
        put_reasons.append("📉 trend stacked bear (20d < 50d)")
    elif not sig.get("above_50d"):
        d50 = sig.get("dist_vs_50d") or 0
        put_score += 2
        put_reasons.append(f"📉 {d50*100:+.1f}% below 50d SMA")
    if sig.get("above_200d") is False and not sig.get("stacked_bear"):
        put_score += 1
        put_reasons.append("📉 below 200d SMA (long-term bear)")

    # ══ MOMENTUM ═══════════════════════════════════════════════════════════
    if ret_5d is not None:
        if ret_5d > 0.05:
            call_score += 2; call_reasons.append(f"🚀 5d price +{ret_5d*100:.1f}% (strong)")
        elif ret_5d > 0.02:
            call_score += 1; call_reasons.append(f"🚀 5d price +{ret_5d*100:.1f}%")
        elif ret_5d < -0.05:
            put_score += 2;  put_reasons.append(f"🔻 5d price {ret_5d*100:.1f}% (strong decline)")
        elif ret_5d < -0.02:
            put_score += 1;  put_reasons.append(f"🔻 5d price {ret_5d*100:.1f}%")

    # Consecutive day streaks
    up_s   = sig.get("up_streak", 0)
    down_s = sig.get("down_streak", 0)
    if up_s >= 3:
        call_score += 2; call_reasons.append(f"🔥 {up_s} green days in a row — persistent buying")
    if down_s >= 3:
        put_score += 2;  put_reasons.append(f"🔥 {down_s} red days in a row — persistent selling")

    # Gap today
    if sig.get("gap_up"):
        call_score += 1; call_reasons.append("⬆️ gap up at open")
    if sig.get("gap_down"):
        put_score += 1;  put_reasons.append("⬇️ gap down at open")

    # ══ OSCILLATORS ════════════════════════════════════════════════════════
    if rsi is not None and 45 <= rsi <= 72:
        call_score += 2; call_reasons.append(f"RSI {rsi:.0f} — trending, room to run")
    if rsi is not None and 28 <= rsi <= 55:
        put_score += 2;  put_reasons.append(f"RSI {rsi:.0f} — weak, room to fall")
    if sig.get("rsi_crossed_50_up"):
        call_score += 1; call_reasons.append("RSI crossed above 50 — bull regime")
    if sig.get("rsi_crossed_50_down"):
        put_score += 1;  put_reasons.append("RSI crossed below 50 — bear regime")

    if sig.get("macd_bull_cross"):
        call_score += 3; call_reasons.append("✨ MACD bullish crossover (fresh)")
    elif sig.get("macd_bullish"):
        call_score += 2; call_reasons.append("MACD bullish & rising")
    if sig.get("macd_hist_accel_up"):
        call_score += 1; call_reasons.append("MACD histogram accelerating up")

    if sig.get("macd_bear_cross"):
        put_score += 3; put_reasons.append("✨ MACD bearish crossover (fresh)")
    elif sig.get("macd_bearish"):
        put_score += 2; put_reasons.append("MACD bearish & falling")
    if sig.get("macd_hist_accel_down"):
        put_score += 1; put_reasons.append("MACD histogram accelerating down")

    # ══ RELATIVE STRENGTH vs SPY ═══════════════════════════════════════════
    if rs_5d is not None:
        if rs_5d > 0.03:
            call_score += 2; call_reasons.append(f"💪 beats SPY by +{rs_5d*100:.1f}% (5d)")
        elif rs_5d > 0.01:
            call_score += 1; call_reasons.append(f"💪 beats SPY by +{rs_5d*100:.1f}% (5d)")
        elif rs_5d < -0.03:
            put_score += 2;  put_reasons.append(f"💀 lags SPY by {rs_5d*100:.1f}% (5d)")
        elif rs_5d < -0.01:
            put_score += 1;  put_reasons.append(f"💀 lags SPY by {rs_5d*100:.1f}% (5d)")
    if rs_20d is not None:
        if rs_20d > 0.05:
            call_score += 1; call_reasons.append(f"💪 beats SPY by +{rs_20d*100:.1f}% (20d)")
        elif rs_20d < -0.05:
            put_score += 1;  put_reasons.append(f"💀 lags SPY by {rs_20d*100:.1f}% (20d)")

    # ══ VOLUME & VOLATILITY EXPANSION ══════════════════════════════════════
    today_green = sig.get("today_green")
    today_red   = sig.get("today_red")
    if vol_ratio >= 1.5:
        if today_green:
            call_score += 2; call_reasons.append(f"🔊 volume {vol_ratio:.1f}x avg on green day — real buying")
        elif today_red:
            put_score += 2;  put_reasons.append(f"🔊 volume {vol_ratio:.1f}x avg on red day — real selling")
        else:
            call_score += 1; put_score += 1
    elif vol_ratio >= 1.2:
        if today_green:
            call_score += 1; call_reasons.append(f"volume {vol_ratio:.1f}x avg — buyers present")
        elif today_red:
            put_score += 1;  put_reasons.append(f"volume {vol_ratio:.1f}x avg — sellers present")

    range_exp = sig.get("range_expansion", 1.0)
    if range_exp >= 1.5:
        if today_green:
            call_score += 1; call_reasons.append(f"📏 range {range_exp:.1f}x avg — volatility expanding up")
        elif today_red:
            put_score += 1;  put_reasons.append(f"📏 range {range_exp:.1f}x avg — volatility expanding down")

    # ══ BREAKOUT / 52W CONTEXT ═════════════════════════════════════════════
    if sig.get("breakout_20d_high"):
        call_score += 2; call_reasons.append("🎯 breaking above 20d high")
    if sig.get("breakdown_20d_low"):
        put_score += 2;  put_reasons.append("🎯 breaking below 20d low")

    dist_hi = sig.get("dist_from_high", 1.0)
    if dist_hi <= 0.01:
        call_score += 2; call_reasons.append("🚩 at 52w high — breakout setup")
    elif dist_hi <= 0.05:
        call_score += 1; call_reasons.append(f"🚩 {dist_hi*100:.1f}% from 52w high")

    dist_lo = sig.get("dist_from_low", 1.0)
    if dist_lo <= 0.01:
        put_score += 2; put_reasons.append("🚩 at 52w low — breakdown setup")
    elif dist_lo <= 0.05:
        put_score += 1; put_reasons.append(f"🚩 {dist_lo*100:.1f}% above 52w low")

    # ══ EARNINGS CATALYST ══════════════════════════════════════════════════
    dte_earn = ctx.get("days_to_earnings")
    earn_before_expiry = ctx.get("earnings_before_expiry")
    if dte_earn is not None:
        if earn_before_expiry and 0 <= dte_earn <= 7:
            # Earnings between now and the weekly expiry — big directional catalyst
            # Score both directions: the direction signal was already earned elsewhere
            if call_score >= put_score:
                call_score += 2
                call_reasons.append(f"📣 earnings in {dte_earn}d — catalyst BEFORE expiry")
            else:
                put_score += 2
                put_reasons.append(f"📣 earnings in {dte_earn}d — catalyst BEFORE expiry")
        elif 0 <= dte_earn <= 14:
            # Earnings within 2 weeks but after expiry — IV elevated going in,
            # LEAPS buyer paying for event risk they won't see realized
            note = f"⚠️ earnings in {dte_earn}d (after expiry) — IV inflated, premium overpaid"
            call_reasons.append(note)
            put_reasons.append(note)
            call_score -= 1
            put_score  -= 1

    # ══ NEWS SENTIMENT ══════════════════════════════════════════════════════
    news_sent = ctx.get("news_sentiment")
    news_hot  = ctx.get("news_hot")
    news_head = ctx.get("news_top_headline")
    if news_sent is not None:
        if news_sent >= 0.3:
            call_score += 2
            hd = f" ({news_head})" if news_head else ""
            call_reasons.append(f"📰 bullish news flow{hd}")
        elif news_sent >= 0.1:
            call_score += 1
            call_reasons.append("📰 mildly bullish news flow")
        elif news_sent <= -0.3:
            put_score += 2
            hd = f" ({news_head})" if news_head else ""
            put_reasons.append(f"📰 bearish news flow{hd}")
        elif news_sent <= -0.1:
            put_score += 1
            put_reasons.append("📰 mildly bearish news flow")
    if news_hot:
        # Hot news reinforces whichever direction is stronger
        note = "🔥 elevated news volume — story developing"
        if call_score >= put_score:
            call_score += 1; call_reasons.append(note)
        else:
            put_score += 1; put_reasons.append(note)

    # ══ DECIDE DIRECTION ═══════════════════════════════════════════════════
    if call_score >= put_score:
        return "CALL", call_score, call_reasons
    else:
        return "PUT", put_score, put_reasons


# ── contract lookup ───────────────────────────────────────────────────────────

def _find_contract(
    ticker: str,
    target_expiry: str,
    direction: str,
    stock_price: float,
) -> Optional[Tuple[str, Dict]]:
    """Find the best near-ATM weekly contract.
    Returns (actual_expiry, contract_dict) or None.
    """
    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        # Find closest available expiry within ±3 days of target
        target_date = date.fromisoformat(target_expiry)
        best_exp    = None
        best_diff   = 999
        for exp in exps:
            try:
                d    = date.fromisoformat(exp)
                diff = abs((d - target_date).days)
                if diff <= 3 and diff < best_diff:
                    best_exp  = exp
                    best_diff = diff
            except ValueError:
                continue

        if not best_exp:
            return None

        ch = t.option_chain(best_exp)
        df = ch.calls.copy() if direction == "CALL" else ch.puts.copy()
        if df.empty:
            return None

        # Near-ATM band: ±4% of stock price
        lo = stock_price * 0.96
        hi = stock_price * 1.04
        band = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
        if band.empty:
            # Fall back to single closest strike
            idx  = (df["strike"] - stock_price).abs().argsort().iloc[0]
            band = df.iloc[[idx]].copy()

        # Must have a real bid
        band = band[band["bid"] > 0.05]
        if band.empty:
            return None

        best = band.sort_values("openInterest", ascending=False).iloc[0]
        bid  = float(best.get("bid") or 0)
        ask  = float(best.get("ask") or 0)
        mid  = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else 999

        # Skip illiquid — spread > 20%
        if spread_pct > 0.20:
            return None

        return best_exp, {
            "strike":     float(best["strike"]),
            "bid":        bid,
            "ask":        ask,
            "mid":        mid,
            "spread_pct": spread_pct,
            "oi":         int(best.get("openInterest") or 0),
        }

    except Exception as e:
        print(f"[weekly] chain error {ticker}: {e}")
        return None


# ── main scan ─────────────────────────────────────────────────────────────────

def scan_weekly(
    universe: List[str],
    dry_run: bool = False,
    news_cap: int = 25,
    min_score: int = MIN_SCORE,
) -> List[WeeklyAlert]:
    """Scan universe for weekly options setups across BOTH expiries (this
    Friday AND next Friday). Returns alerts sorted by expiry, then score desc.

    Args:
      news_cap:  max tickers that get news+earnings enrichment. Beyond this,
                 candidates are dropped silently. Default 25 is tuned for the
                 ~85-name curated list. The market scanner (~60 prescreen
                 survivors) should raise this so its top survivors don't get
                 truncated.
      min_score: minimum score required to emit an alert. Default 8 fits the
                 curated list where most names have rich Polygon news. The
                 market scanner sees lots of mid-caps with sparse news, so it
                 passes a lower bar (e.g. 7) to compensate.

    Pipeline:
      1. Batch-fetch bars (once)
      2. Technical screen (once)
      3. News + earnings enrichment (once — news doesn't depend on expiry)
      4. For EACH of the two expiries:
         - Recompute earnings_before_expiry relative to that expiry
         - Re-score (earnings context can shift score)
         - Find ATM contract
         - Emit alert if score + liquidity pass
    """
    targets = get_target_expiries()
    print(f"[weekly] scanning {len(universe)} tickers for {len(targets)} expiries:")
    for expiry, mode, dte in targets:
        label = "this Friday" if mode == "THIS_WEEK" else "next Friday"
        print(f"[weekly]   {mode:<10s} {expiry}  ({dte} DTE, {label})")

    # ── 1. batch-fetch bars (fast) ────────────────────────────────────────
    all_tickers = list(dict.fromkeys(["SPY"] + universe))
    bars_map    = fetch_bars_batch(all_tickers, period="1y")
    spy_bars    = bars_map.get("SPY", pd.DataFrame())

    # ── 2. technical screen (no API calls, instant) ──────────────────────
    pre_candidates = []   # (ticker, sig, technical_score)
    for ticker in universe:
        bars = bars_map.get(ticker)
        if bars is None or bars.empty or len(bars) < 60:
            continue
        sig = _compute_signals(bars, spy_bars)
        direction, tech_score, _ = _score(sig)            # no context yet
        if tech_score >= min_score - 2 and direction:     # widen pre-filter
            pre_candidates.append((ticker, sig, tech_score))

    pre_candidates.sort(key=lambda x: -x[2])
    print(f"[weekly] {len(pre_candidates)} passed technical screen "
          f"(min_score={min_score}, news_cap={news_cap}) — fetching news + earnings...")

    # ── 3. enrich with earnings + news ONCE (expiry-independent) ─────────
    # We compute earnings_before_expiry separately per expiry below.
    far_expiry = targets[-1][0]   # use the farther Friday for the initial fetch
    enriched_tickers = []   # (ticker, sig, base_ctx)
    truncated = max(0, len(pre_candidates) - news_cap)
    if truncated:
        print(f"[weekly] news_cap={news_cap} — dropping {truncated} lower-tech-score candidates from enrichment")
    for ticker, sig, _ in pre_candidates[:news_cap]:
        ctx = _fetch_news_and_earnings(ticker, far_expiry)
        n_arts  = ctx.get("news_article_count") or 0
        n_srcs  = ",".join(ctx.get("news_sources") or []) or "-"
        n_sent  = ctx.get("news_sentiment")
        sent_s  = f"{n_sent:+.2f}" if n_sent is not None else " n/a"
        earn    = ctx.get("days_to_earnings")
        earn_s  = f"E{earn}d" if earn is not None else "E?"
        print(f"  {ticker:6s}  news[{n_srcs:<20s}] articles={n_arts:2d}  "
              f"sent={sent_s}  {earn_s}")
        enriched_tickers.append((ticker, sig, ctx))

    # ── 4. for each expiry, re-score + find contract ─────────────────────
    alerts: List[WeeklyAlert] = []
    for expiry, mode, _dte in targets:
        exp_date     = date.fromisoformat(expiry)
        days_to_exp  = (exp_date - date.today()).days
        print(f"\n[weekly] === {mode} ({expiry}, {days_to_exp} DTE) ===")

        per_expiry_survivors = []
        for ticker, sig, base_ctx in enriched_tickers:
            # Clone context + recompute earnings_before_expiry for THIS expiry
            ctx = dict(base_ctx)
            dte_e = ctx.get("days_to_earnings")
            ctx["earnings_before_expiry"] = bool(
                dte_e is not None and 0 <= dte_e <= days_to_exp
            )

            direction, score, reasons = _score(sig, ctx)
            if score >= min_score:
                per_expiry_survivors.append((ticker, direction, score, reasons, sig, ctx))

        per_expiry_survivors.sort(key=lambda x: -x[2])
        print(f"[weekly] {len(per_expiry_survivors)} passed score ≥{min_score} — fetching chains...")

        for ticker, direction, score, reasons, sig, ctx in per_expiry_survivors:
            result = _find_contract(ticker, expiry, direction, sig["price"])
            if not result:
                continue
            actual_expiry, contract = result
            actual_dte = (date.fromisoformat(actual_expiry) - date.today()).days

            alerts.append(WeeklyAlert(
                ticker      = ticker,
                direction   = direction,
                strike      = contract["strike"],
                expiry      = actual_expiry,
                dte         = actual_dte,
                bid         = contract["bid"],
                ask         = contract["ask"],
                mid         = contract["mid"],
                stock_price = sig["price"],
                reasons     = reasons,
                score       = score,
                spread_pct  = contract["spread_pct"],
                oi          = contract["oi"],
                days_to_earnings       = ctx.get("days_to_earnings"),
                earnings_before_expiry = ctx.get("earnings_before_expiry", False),
                news_sentiment         = ctx.get("news_sentiment"),
                news_top_headline      = ctx.get("news_top_headline"),
                news_hot               = ctx.get("news_hot", False),
                headlines              = ctx.get("headlines", []),
                news_sources           = ctx.get("news_sources", []),
                news_article_count     = ctx.get("news_article_count", 0),
            ))
            opt_type = "C" if direction == "CALL" else "P"
            print(f"  {ticker}: {direction} ${contract['strike']:.0f}{opt_type} "
                  f"{actual_expiry}  mid ${contract['mid']:.2f}  score={score}")

    # Sort: nearer expiry first, then by score descending
    alerts.sort(key=lambda a: (a.expiry, -a.score))
    return alerts
