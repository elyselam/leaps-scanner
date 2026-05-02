"""Earnings play scanner — find mispriced vol around upcoming earnings.

The edge: when the options market underprices the actual historical move,
buying a straddle or directional play is +EV. When it overprices, selling
premium (iron condor) is the play. This scanner finds both.

Pipeline:
  1. Pull universe (S&P 500 + Nasdaq-100 from cached market_universe)
  2. Check earnings dates — keep those reporting in 1–5 trading days
  3. For each upcoming ER:
     a. Get ATM straddle price → expected (implied) move
     b. Compute historical avg earnings-day move (last 4–8 quarters)
     c. Compare implied vs historical → cheap/fair/expensive vol
     d. Check pre-earnings drift (is stock extended into ER?)
     e. Check beat/miss track record
     f. Check put/call OI positioning on earnings-week expiry
  4. Score + classify play type (directional call/put, straddle, IV crush)
  5. Return top plays sorted by conviction

Posts to #earnings via EARNINGS_WEBHOOK_URL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from data import fetch_bars_batch, fetch_fundamentals
from market_scanner import get_market_universe


# ── config ──────────────────────────────────────────────────────────────────

# How many trading days ahead to look for earnings
EARNINGS_WINDOW_DAYS = 5

# Minimum historical quarters needed to compute avg move
MIN_HISTORY_QUARTERS = 4

# Implied vs historical thresholds
VOL_CHEAP_RATIO  = 0.85   # implied < 85% of historical → cheap vol (buy)
VOL_EXPENSIVE_RATIO = 1.25   # implied > 125% of historical → expensive (sell)


# ── data class ──────────────────────────────────────────────────────────────

@dataclass
class EarningsPlay:
    ticker:          str
    earnings_date:   str           # YYYY-MM-DD
    days_to_er:      int           # trading days until ER
    er_timing:       str           # "BMO" (before market open) | "AMC" (after close) | "unknown"
    stock_price:     float

    # Vol pricing
    implied_move_pct:   float      # expected move from straddle (as % of stock)
    historical_move_pct: float     # avg |% move| on ER day (last N quarters)
    iv_vs_hist_ratio:   float      # implied / historical — <0.85 = cheap, >1.25 = expensive
    num_quarters:       int        # how many ER events in history sample

    # Direction signals
    play_type:       str           # "CALL" | "PUT" | "STRADDLE" | "IV_CRUSH"
    direction_score: float         # -1.0 (bearish) to +1.0 (bullish)
    pre_er_drift_5d: float         # 5d return going into ER
    beat_rate:       float         # fraction of last N quarters that beat EPS (0.0–1.0)

    # Options positioning
    call_put_oi_ratio: Optional[float] = None   # call OI / put OI on ER-week expiry

    # Suggested contract
    suggested_strike:  Optional[float] = None
    suggested_expiry:  Optional[str]   = None
    suggested_type:    Optional[str]   = None   # "C" or "P" or "straddle"
    suggested_mid:     Optional[float] = None
    suggested_oi:      Optional[int]   = None

    # Scoring
    score:           float = 0.0
    reasons:         List[str] = field(default_factory=list)

    # Historical context
    last_moves:      List[float] = field(default_factory=list)  # last N ER % moves


# ── earnings date detection ─────────────────────────────────────────────────

def _get_earnings_date(ticker: str) -> Optional[Tuple[str, str]]:
    """Get next earnings date and timing for a ticker.
    Returns (date_str, timing) or None.
    timing: 'BMO', 'AMC', or 'unknown'
    """
    try:
        t = yf.Ticker(ticker)
        # yfinance .calendar gives next earnings
        cal = t.calendar
        if cal is None or cal.empty if hasattr(cal, 'empty') else not cal:
            return None

        # calendar can be a DataFrame or dict depending on yfinance version
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                er_dates = cal["Earnings Date"].tolist()
            elif "Earnings Date" in cal.index:
                er_dates = [cal.loc["Earnings Date"].iloc[0]]
            else:
                return None
        elif isinstance(cal, dict):
            er_dates = cal.get("Earnings Date", [])
            if not isinstance(er_dates, list):
                er_dates = [er_dates]
        else:
            return None

        if not er_dates:
            return None

        # Get the earliest upcoming date
        today = date.today()
        for ed in er_dates:
            if isinstance(ed, (datetime, pd.Timestamp)):
                er_date = ed.date() if hasattr(ed, 'date') else ed
            elif isinstance(ed, str):
                er_date = date.fromisoformat(ed[:10])
            elif isinstance(ed, date):
                er_date = ed
            else:
                continue

            if er_date >= today:
                # Try to determine BMO/AMC from the datetime hour
                timing = "unknown"
                if isinstance(ed, (datetime, pd.Timestamp)):
                    hour = ed.hour if hasattr(ed, 'hour') else 0
                    if hour < 12:
                        timing = "BMO"
                    elif hour >= 16:
                        timing = "AMC"
                return er_date.isoformat(), timing

    except Exception as e:
        # Silently skip — many tickers won't have calendar data
        pass
    return None


# ── historical earnings moves ───────────────────────────────────────────────

def _historical_earnings_moves(ticker: str, bars: pd.DataFrame,
                                num_quarters: int = 8) -> List[float]:
    """Compute absolute % move on each of the last N earnings days.

    Uses yfinance earnings_dates to get historical ER dates, then looks up
    the close-to-close move on that date from bars.
    """
    moves: List[float] = []
    try:
        t = yf.Ticker(ticker)
        # earnings_dates gives historical dates
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return moves

        # ed.index is DatetimeIndex of earnings dates
        er_dates = sorted(ed.index.date, reverse=True)[:num_quarters * 2]

        if bars is None or bars.empty:
            return moves

        close = bars["Close"]
        bar_dates = close.index.date if hasattr(close.index, 'date') else [
            d.date() for d in close.index
        ]

        for er_d in er_dates:
            # Find the bar index closest to this earnings date (within 1 day)
            for i, bd in enumerate(bar_dates):
                if abs((bd - er_d).days) <= 1 and i > 0:
                    # close-to-close move on ER day
                    move = float(close.iloc[i] / close.iloc[i - 1] - 1)
                    moves.append(move)
                    break
            if len(moves) >= num_quarters:
                break

    except Exception:
        pass
    return moves


# ── straddle pricing ────────────────────────────────────────────────────────

def _get_straddle_pricing(ticker: str, stock_price: float,
                          earnings_date: str) -> Optional[Dict]:
    """Get ATM straddle price from the expiry closest to (but after) earnings.

    Returns dict with: implied_move_pct, call_strike, put_strike, straddle_mid,
    call_put_oi_ratio, best_expiry, suggested contract info.
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        er_d = date.fromisoformat(earnings_date)

        # Find first expiry ON or AFTER earnings (so the straddle captures the event)
        best_exp = None
        for exp in sorted(exps):
            exp_d = date.fromisoformat(exp)
            if exp_d >= er_d:
                best_exp = exp
                break

        if not best_exp:
            return None

        ch = t.option_chain(best_exp)
        calls = ch.calls
        puts  = ch.puts

        if calls.empty or puts.empty:
            return None

        # Find ATM strike (closest to current price)
        call_strikes = calls["strike"].values
        atm_idx = int(np.abs(call_strikes - stock_price).argmin())
        atm_strike = float(call_strikes[atm_idx])

        # Get ATM call and put
        atm_call = calls[calls["strike"] == atm_strike]
        atm_put  = puts[puts["strike"] == atm_strike]

        if atm_call.empty or atm_put.empty:
            return None

        call_bid = float(atm_call["bid"].iloc[0] or 0)
        call_ask = float(atm_call["ask"].iloc[0] or 0)
        put_bid  = float(atm_put["bid"].iloc[0] or 0)
        put_ask  = float(atm_put["ask"].iloc[0] or 0)

        call_mid = (call_bid + call_ask) / 2
        put_mid  = (put_bid + put_ask) / 2
        straddle_mid = call_mid + put_mid

        if straddle_mid <= 0 or stock_price <= 0:
            return None

        implied_move_pct = straddle_mid / stock_price

        # Call/Put OI ratio across the whole chain (not just ATM)
        total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
        total_put_oi  = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
        cp_ratio = total_call_oi / total_put_oi if total_put_oi > 0 else None

        # Find best directional contract (slightly OTM for leverage)
        # For calls: first strike above current price with decent OI
        # For puts: first strike below current price with decent OI
        otm_call = calls[(calls["strike"] > stock_price) &
                         (calls["strike"] <= stock_price * 1.05) &
                         (calls["bid"] > 0.10)]
        otm_put  = puts[(puts["strike"] < stock_price) &
                        (puts["strike"] >= stock_price * 0.95) &
                        (puts["bid"] > 0.10)]

        # Pick best OTM call by OI
        suggested = {}
        if not otm_call.empty:
            best_c = otm_call.sort_values("openInterest", ascending=False).iloc[0]
            suggested["call_strike"] = float(best_c["strike"])
            suggested["call_mid"] = (float(best_c["bid"] or 0) + float(best_c["ask"] or 0)) / 2
            suggested["call_oi"] = int(best_c.get("openInterest") or 0)
        if not otm_put.empty:
            best_p = otm_put.sort_values("openInterest", ascending=False).iloc[0]
            suggested["put_strike"] = float(best_p["strike"])
            suggested["put_mid"] = (float(best_p["bid"] or 0) + float(best_p["ask"] or 0)) / 2
            suggested["put_oi"] = int(best_p.get("openInterest") or 0)

        return {
            "implied_move_pct": implied_move_pct,
            "straddle_mid":     straddle_mid,
            "atm_strike":       atm_strike,
            "call_put_oi_ratio": cp_ratio,
            "best_expiry":      best_exp,
            **suggested,
        }

    except Exception as e:
        print(f"[earnings] straddle error {ticker}: {e}")
        return None


# ── scoring + play classification ───────────────────────────────────────────

def _classify_play(
    implied_move: float,
    hist_avg_move: float,
    pre_drift: float,
    beat_rate: float,
    cp_ratio: Optional[float],
    hist_moves: List[float],
) -> Tuple[str, float, float, List[str]]:
    """Classify the optimal play type and score conviction.

    Returns (play_type, direction_score, conviction_score, reasons).
    """
    reasons: List[str] = []
    score = 0.0
    direction = 0.0  # -1 to +1

    ratio = implied_move / hist_avg_move if hist_avg_move > 0 else 1.0

    # ── VOL PRICING ──
    if ratio <= VOL_CHEAP_RATIO:
        score += 3.0
        reasons.append(f"📊 vol CHEAP — implied {implied_move*100:.1f}% vs "
                      f"historical {hist_avg_move*100:.1f}% (ratio {ratio:.2f})")
    elif ratio >= VOL_EXPENSIVE_RATIO:
        score += 2.0
        reasons.append(f"📊 vol EXPENSIVE — implied {implied_move*100:.1f}% vs "
                      f"historical {hist_avg_move*100:.1f}% (ratio {ratio:.2f})")
    else:
        reasons.append(f"📊 vol fair — implied {implied_move*100:.1f}% vs "
                      f"historical {hist_avg_move*100:.1f}% (ratio {ratio:.2f})")

    # ── BEAT/MISS TRACK RECORD ──
    if beat_rate >= 0.80:
        score += 2.0
        direction += 0.3
        reasons.append(f"✅ beats {beat_rate*100:.0f}% of last quarters — consistent winner")
    elif beat_rate >= 0.60:
        score += 1.0
        direction += 0.15
        reasons.append(f"✅ beats {beat_rate*100:.0f}% of quarters")
    elif beat_rate <= 0.30:
        direction -= 0.2
        reasons.append(f"❌ misses {(1-beat_rate)*100:.0f}% of quarters — unreliable")

    # ── PRE-EARNINGS DRIFT ──
    if pre_drift >= 0.08:
        direction -= 0.2   # extended → sell the news risk
        reasons.append(f"⚠️ run-up {pre_drift*100:+.1f}% into ER — priced for perfection")
    elif pre_drift >= 0.03:
        direction += 0.1   # mild bullish drift — positive anticipation
        reasons.append(f"📈 mild bullish drift {pre_drift*100:+.1f}% into ER")
    elif pre_drift <= -0.05:
        direction += 0.15   # sold off → upside surprise potential
        score += 0.5
        reasons.append(f"📉 sold off {pre_drift*100:+.1f}% into ER — low bar")
    elif pre_drift <= -0.02:
        reasons.append(f"📉 slight weakness {pre_drift*100:+.1f}% into ER")

    # ── OPTIONS POSITIONING ──
    if cp_ratio is not None:
        if cp_ratio >= 2.5:
            direction += 0.1
            score += 0.5
            reasons.append(f"📞 heavy call positioning (C/P OI {cp_ratio:.1f}x)")
        elif cp_ratio >= 1.5:
            direction += 0.05
            reasons.append(f"📞 call-biased positioning (C/P OI {cp_ratio:.1f}x)")
        elif cp_ratio <= 0.6:
            direction -= 0.1
            reasons.append(f"📋 heavy put hedging (C/P OI {cp_ratio:.1f}x)")

    # ── HISTORICAL MOVE CONSISTENCY ──
    if hist_moves:
        # If most past moves were in one direction, lean that way
        up_moves = sum(1 for m in hist_moves if m > 0)
        if up_moves >= len(hist_moves) * 0.75:
            direction += 0.2
            score += 1.0
            reasons.append(f"📊 rallied on {up_moves}/{len(hist_moves)} recent ERs")
        elif up_moves <= len(hist_moves) * 0.25:
            direction -= 0.2
            score += 1.0
            reasons.append(f"📊 sold off on {len(hist_moves)-up_moves}/{len(hist_moves)} recent ERs")

    # ── CLASSIFY PLAY TYPE ──
    direction = max(-1.0, min(1.0, direction))

    if ratio <= VOL_CHEAP_RATIO:
        # Vol is cheap — buy it
        if abs(direction) >= 0.25:
            play_type = "CALL" if direction > 0 else "PUT"
            score += 1.0
        else:
            play_type = "STRADDLE"
            score += 0.5
    elif ratio >= VOL_EXPENSIVE_RATIO:
        play_type = "IV_CRUSH"
        score += 1.0
    else:
        # Vol is fair — only play if directional conviction is strong
        if abs(direction) >= 0.3:
            play_type = "CALL" if direction > 0 else "PUT"
        elif abs(direction) >= 0.15:
            play_type = "CALL" if direction > 0 else "PUT"
            score -= 0.5   # lower conviction
        else:
            play_type = "STRADDLE"

    return play_type, direction, score, reasons


# ── main scan ───────────────────────────────────────────────────────────────

def scan_earnings(dry_run: bool = False, max_plays: int = 10) -> List[EarningsPlay]:
    """Scan market universe for upcoming earnings plays.

    Returns plays sorted by score (highest conviction first).
    """
    universe = get_market_universe()
    if not universe:
        print("[earnings] empty universe — aborting")
        return []

    print(f"[earnings] checking earnings dates for {len(universe)} tickers "
          f"(window: {EARNINGS_WINDOW_DAYS} trading days)...")

    # Phase 1: Find tickers with earnings in the next N days
    today = date.today()
    window_end = today + timedelta(days=EARNINGS_WINDOW_DAYS + 3)  # +3 for weekends

    upcoming: List[Tuple[str, str, int, str]] = []  # (ticker, date, days_to, timing)
    checked = 0
    for ticker in universe:
        result = _get_earnings_date(ticker)
        if result:
            er_date_str, timing = result
            er_date = date.fromisoformat(er_date_str)
            days_to = (er_date - today).days
            if 0 <= days_to <= EARNINGS_WINDOW_DAYS:
                upcoming.append((ticker, er_date_str, days_to, timing))
        checked += 1
        # Progress print every 100 tickers
        if checked % 100 == 0:
            print(f"[earnings]   checked {checked}/{len(universe)} "
                  f"— found {len(upcoming)} upcoming...")

    print(f"[earnings] found {len(upcoming)} tickers with earnings in next "
          f"{EARNINGS_WINDOW_DAYS} days")

    if not upcoming:
        return []

    # Phase 2: Fetch bars for all upcoming tickers (for drift + historical moves)
    er_tickers = [t for t, _, _, _ in upcoming]
    bars_map = fetch_bars_batch(er_tickers, period="2y")

    # Phase 3: Deep analysis for each
    plays: List[EarningsPlay] = []

    for ticker, er_date_str, days_to, timing in upcoming:
        bars = bars_map.get(ticker)
        if bars is None or bars.empty or len(bars) < 60:
            continue

        close = bars["Close"]
        stock_price = float(close.iloc[-1])

        # Pre-earnings drift (5d)
        pre_drift = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0.0

        # Historical earnings moves
        hist_moves = _historical_earnings_moves(ticker, bars)
        if len(hist_moves) < MIN_HISTORY_QUARTERS:
            # Not enough history — skip (can't price vol without baseline)
            continue

        hist_avg_move = float(np.mean([abs(m) for m in hist_moves]))

        # Straddle pricing
        straddle_info = _get_straddle_pricing(ticker, stock_price, er_date_str)
        if not straddle_info:
            continue

        implied_move = straddle_info["implied_move_pct"]
        cp_ratio = straddle_info.get("call_put_oi_ratio")

        # Beat rate (from historical moves — positive moves ≈ beats)
        # This is approximate: a positive ER-day move usually means beat
        beat_rate = sum(1 for m in hist_moves if m > 0) / len(hist_moves)

        # Classify
        iv_ratio = implied_move / hist_avg_move if hist_avg_move > 0 else 1.0
        play_type, direction, score, reasons = _classify_play(
            implied_move, hist_avg_move, pre_drift, beat_rate,
            cp_ratio, hist_moves,
        )

        # Build suggested contract
        suggested_strike = None
        suggested_type = None
        suggested_mid = None
        suggested_oi = None
        suggested_expiry = straddle_info.get("best_expiry")

        if play_type == "CALL" and "call_strike" in straddle_info:
            suggested_strike = straddle_info["call_strike"]
            suggested_type = "C"
            suggested_mid = straddle_info.get("call_mid")
            suggested_oi = straddle_info.get("call_oi")
        elif play_type == "PUT" and "put_strike" in straddle_info:
            suggested_strike = straddle_info["put_strike"]
            suggested_type = "P"
            suggested_mid = straddle_info.get("put_mid")
            suggested_oi = straddle_info.get("put_oi")
        elif play_type == "STRADDLE":
            suggested_strike = straddle_info.get("atm_strike")
            suggested_type = "straddle"
            suggested_mid = straddle_info.get("straddle_mid")

        play = EarningsPlay(
            ticker=ticker,
            earnings_date=er_date_str,
            days_to_er=days_to,
            er_timing=timing,
            stock_price=stock_price,
            implied_move_pct=implied_move,
            historical_move_pct=hist_avg_move,
            iv_vs_hist_ratio=iv_ratio,
            num_quarters=len(hist_moves),
            play_type=play_type,
            direction_score=direction,
            pre_er_drift_5d=pre_drift,
            beat_rate=beat_rate,
            call_put_oi_ratio=cp_ratio,
            suggested_strike=suggested_strike,
            suggested_expiry=suggested_expiry,
            suggested_type=suggested_type,
            suggested_mid=suggested_mid,
            suggested_oi=suggested_oi,
            score=score,
            reasons=reasons,
            last_moves=hist_moves[:8],
        )
        plays.append(play)

        # Log
        timing_str = f" ({timing})" if timing != "unknown" else ""
        print(f"  {ticker:6s}  ER {er_date_str}{timing_str}  "
              f"implied ±{implied_move*100:.1f}%  hist ±{hist_avg_move*100:.1f}%  "
              f"ratio={iv_ratio:.2f}  → {play_type}  score={score:.1f}")

    # Sort by score and cap
    plays.sort(key=lambda p: (-p.score, p.days_to_er))
    if len(plays) > max_plays:
        print(f"[earnings] {len(plays)} plays found — returning top {max_plays}")
        plays = plays[:max_plays]

    print(f"[earnings] final: {len(plays)} earnings plays")
    return plays
