"""Sector rotation scanner — detect where institutional money is flowing next.

Concept:
  Institutions rotate capital between sectors on multi-week cycles driven by
  macro regime changes (rates, growth expectations, risk appetite). The signals
  show up as relative-strength acceleration in sector ETFs days before
  individual names break out.

  This scanner catches:
    1. FRESH ROTATION IN  — sector was lagging (20d RS negative) but just
       started leading (5d RS positive). Earliest actionable signal.
    2. ACCELERATING       — sector already leading, and accelerating further.
       Confirms the rotation thesis; momentum chasers pile in here.
    3. ROTATION OUT       — sector was leading but 5d RS just flipped negative.
       Get out or avoid new entries.
    4. THEMATIC SHIFTS    — factor ETFs (growth/value, large/small, risk-on/off)
       to frame the macro context for the sector moves.

Sectors tracked:
  - 11 SPDR Select Sector ETFs (XLK, XLV, XLF, XLE, XLI, XLC, XLRE, XLU, XLP, XLB, XLY)
  - Thematic ETFs: SMH (semis), XBI (biotech), ARKK (disruptive), IWM (small-cap),
    KRE (regional banks), XHB (homebuilders), TAN (solar), KWEB (China tech)
  - Factor benchmarks: IWF (growth), IWD (value), QQQ (mega-tech)

Output:
  One Discord embed with sections for "Rotating INTO", "Accelerating",
  "Rotating OUT", and a one-line macro context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data import fetch_bars_batch


# ── ETF universe ────────────────────────────────────────────────────────────

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLV":  "Healthcare",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLC":  "Communication",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLP":  "Consumer Staples",
    "XLB":  "Materials",
    "XLY":  "Consumer Disc.",
}

THEMATIC_ETFS = {
    "SMH":  "Semiconductors",
    "XBI":  "Biotech",
    "ARKK": "Disruptive/Growth",
    "IWM":  "Small Caps",
    "KRE":  "Regional Banks",
    "XHB":  "Homebuilders",
    "TAN":  "Solar/Clean",
    "KWEB": "China Tech",
}

FACTOR_ETFS = {
    "IWF":  "Growth Factor",
    "IWD":  "Value Factor",
    "QQQ":  "Mega-Cap Tech",
}

ALL_ETFS = {**SECTOR_ETFS, **THEMATIC_ETFS, **FACTOR_ETFS}


# ── data class ──────────────────────────────────────────────────────────────

@dataclass
class RotationSignal:
    ticker:     str
    name:       str
    category:   str       # "sector" | "thematic" | "factor"
    signal:     str       # "ROTATING_IN" | "ACCELERATING" | "ROTATING_OUT" | "DECELERATING"

    # Relative strength vs SPY at each timeframe
    rs_5d:      float = 0.0
    rs_10d:     float = 0.0
    rs_20d:     float = 0.0
    rs_60d:     float = 0.0

    # Acceleration: rs_5d - rs_20d (positive = improving faster than trend)
    rs_accel:   float = 0.0

    # Volume signals
    vol_ratio_5d:  float = 1.0   # avg vol last 5d / avg vol last 20d

    # Price context
    price:      float = 0.0
    ret_5d:     float = 0.0
    ret_20d:    float = 0.0
    dist_from_high: float = 0.0  # distance from 52w high

    # Composite score for ranking (higher = stronger rotation signal)
    strength:   float = 0.0

    # Human-readable one-liner
    reason:     str = ""


# ── signal computation ──────────────────────────────────────────────────────

def _compute_rs(etf_close: pd.Series, spy_close: pd.Series, days: int) -> Optional[float]:
    """Relative strength = ETF return - SPY return over `days` trading days."""
    if len(etf_close) < days + 1 or len(spy_close) < days + 1:
        return None
    etf_ret = float(etf_close.iloc[-1] / etf_close.iloc[-days] - 1)
    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-days] - 1)
    return etf_ret - spy_ret


def _analyze_etf(
    ticker: str,
    bars: pd.DataFrame,
    spy_close: pd.Series,
) -> Optional[RotationSignal]:
    """Compute rotation metrics for a single ETF."""
    if bars is None or bars.empty or len(bars) < 70:
        return None

    close  = bars["Close"]
    volume = bars["Volume"]
    high   = bars["High"]

    # Relative strength at multiple timeframes
    rs_5  = _compute_rs(close, spy_close, 5)
    rs_10 = _compute_rs(close, spy_close, 10)
    rs_20 = _compute_rs(close, spy_close, 20)
    rs_60 = _compute_rs(close, spy_close, 60)

    if rs_5 is None or rs_20 is None:
        return None

    # Acceleration: how fast is RS changing? (positive = improving)
    rs_accel = rs_5 - rs_20

    # Volume: are institutions building positions?
    avg_vol_5d  = float(volume.iloc[-5:].mean())
    avg_vol_20d = float(volume.iloc[-20:].mean())
    vol_ratio   = avg_vol_5d / avg_vol_20d if avg_vol_20d > 0 else 1.0

    # Price context
    price   = float(close.iloc[-1])
    ret_5d  = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0.0
    ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0.0

    hi52 = float(high.iloc[-252:].max()) if len(high) >= 252 else float(high.max())
    dist_from_high = (hi52 - price) / hi52 if hi52 > 0 else 0.0

    # ── classify signal type ────────────────────────────────────────────
    # ROTATING_IN: was lagging (20d RS < 0), now leading short-term (5d RS > 0)
    # + acceleration confirms the flip is real (not a 1-day dead-cat bounce)
    if rs_20 < -0.005 and rs_5 > 0.005 and rs_accel > 0.01:
        signal = "ROTATING_IN"
    elif rs_5 > 0.01 and rs_20 > 0.005 and rs_accel > 0.005:
        signal = "ACCELERATING"
    elif rs_20 > 0.005 and rs_5 < -0.005 and rs_accel < -0.01:
        signal = "ROTATING_OUT"
    elif rs_5 < -0.005 and rs_20 < -0.005 and rs_accel < -0.005:
        signal = "DECELERATING"
    else:
        return None   # no clear rotation signal — skip

    # ── strength score (for ranking) ────────────────────────────────────
    # Fresh rotations rank highest (actionable NOW).
    # Acceleration of existing trend ranks next.
    # Rotation-out is also important (avoid/exit signal).
    strength = 0.0
    if signal == "ROTATING_IN":
        # Reward: magnitude of RS flip + volume confirmation
        strength = abs(rs_accel) * 100 + abs(rs_5) * 50
        if vol_ratio >= 1.2:
            strength += (vol_ratio - 1.0) * 20
        # Bonus if near 52w high (breakout) or far from high (mean-reversion)
        if dist_from_high <= 0.05:
            strength += 5   # breaking out to new highs
    elif signal == "ACCELERATING":
        strength = abs(rs_accel) * 80 + abs(rs_5) * 30
        if vol_ratio >= 1.2:
            strength += (vol_ratio - 1.0) * 15
    elif signal == "ROTATING_OUT":
        strength = abs(rs_accel) * 100 + abs(rs_5) * 50
    elif signal == "DECELERATING":
        strength = abs(rs_accel) * 60

    # ── reason one-liner ────────────────────────────────────────────────
    vol_note = f"  vol {vol_ratio:.1f}x" if vol_ratio >= 1.2 else ""
    hi_note  = " (near 52w high)" if dist_from_high <= 0.03 else ""

    if signal == "ROTATING_IN":
        reason = (f"RS flipping: 20d {rs_20*100:+.1f}% → 5d {rs_5*100:+.1f}% "
                  f"(accel {rs_accel*100:+.1f}%){vol_note}{hi_note}")
    elif signal == "ACCELERATING":
        reason = (f"Leading & gaining: 5d RS {rs_5*100:+.1f}%, "
                  f"accel {rs_accel*100:+.1f}%{vol_note}{hi_note}")
    elif signal == "ROTATING_OUT":
        reason = (f"RS fading: 20d {rs_20*100:+.1f}% → 5d {rs_5*100:+.1f}% "
                  f"(accel {rs_accel*100:+.1f}%){vol_note}")
    else:
        reason = (f"Lagging & fading: 5d RS {rs_5*100:+.1f}%, "
                  f"accel {rs_accel*100:+.1f}%{vol_note}")

    # Determine category
    if ticker in SECTOR_ETFS:
        category = "sector"
    elif ticker in THEMATIC_ETFS:
        category = "thematic"
    else:
        category = "factor"

    return RotationSignal(
        ticker=ticker,
        name=ALL_ETFS.get(ticker, ticker),
        category=category,
        signal=signal,
        rs_5d=rs_5,
        rs_10d=rs_10 or 0.0,
        rs_20d=rs_20,
        rs_60d=rs_60 or 0.0,
        rs_accel=rs_accel,
        vol_ratio_5d=vol_ratio,
        price=price,
        ret_5d=ret_5d,
        ret_20d=ret_20d,
        dist_from_high=dist_from_high,
        strength=strength,
        reason=reason,
    )


# ── macro context ───────────────────────────────────────────────────────────

def _macro_context(signals: List[RotationSignal], spy_bars: pd.DataFrame) -> str:
    """One-line macro read from factor ETFs + SPY trend."""
    # SPY trend
    if spy_bars is None or spy_bars.empty or len(spy_bars) < 50:
        spy_ctx = "SPY data insufficient"
    else:
        spy_close = spy_bars["Close"]
        sma20 = float(spy_close.rolling(20).mean().iloc[-1])
        sma50 = float(spy_close.rolling(50).mean().iloc[-1])
        price = float(spy_close.iloc[-1])
        ret_5d = float(spy_close.iloc[-1] / spy_close.iloc[-5] - 1)

        if price > sma20 > sma50:
            regime = "📈 uptrend"
        elif price < sma20 < sma50:
            regime = "📉 downtrend"
        elif price > sma20:
            regime = "↗️ recovering"
        else:
            regime = "↘️ weakening"
        spy_ctx = f"SPY {regime} (5d {ret_5d*100:+.1f}%)"

    # Factor read
    factors = [s for s in signals if s.category == "factor"]
    growth_leading = any(s.ticker == "IWF" and s.signal in ("ROTATING_IN", "ACCELERATING")
                        for s in factors)
    value_leading  = any(s.ticker == "IWD" and s.signal in ("ROTATING_IN", "ACCELERATING")
                        for s in factors)
    small_leading  = any(s.ticker == "IWM" and s.signal in ("ROTATING_IN", "ACCELERATING")
                        for s in signals)

    factor_notes = []
    if growth_leading:  factor_notes.append("growth > value")
    if value_leading:   factor_notes.append("value > growth")
    if small_leading:   factor_notes.append("small > large (risk-on)")

    factor_str = " · ".join(factor_notes) if factor_notes else "no clear factor tilt"

    return f"{spy_ctx}  ·  {factor_str}"


# ── main scan ───────────────────────────────────────────────────────────────

def scan_rotation(dry_run: bool = False) -> Tuple[List[RotationSignal], str]:
    """Run the rotation scan.

    Returns:
      (signals, macro_context_string)
      Signals are sorted by strength descending.
    """
    universe = list(ALL_ETFS.keys()) + ["SPY"]
    print(f"[rotation] scanning {len(universe) - 1} ETFs for sector rotation signals...")

    bars_map = fetch_bars_batch(universe, period="1y")
    spy_bars = bars_map.get("SPY")
    if spy_bars is None or spy_bars.empty:
        print("[rotation] ERROR: SPY bars empty — cannot compute relative strength")
        return [], "SPY data unavailable"

    spy_close = spy_bars["Close"]

    signals: List[RotationSignal] = []
    for ticker in ALL_ETFS:
        bars = bars_map.get(ticker)
        sig = _analyze_etf(ticker, bars, spy_close)
        if sig:
            signals.append(sig)

    # Sort by strength
    signals.sort(key=lambda s: -s.strength)

    # Print summary
    rotating_in  = [s for s in signals if s.signal == "ROTATING_IN"]
    accelerating = [s for s in signals if s.signal == "ACCELERATING"]
    rotating_out = [s for s in signals if s.signal == "ROTATING_OUT"]
    decelerating = [s for s in signals if s.signal == "DECELERATING"]

    macro = _macro_context(signals, spy_bars)

    print(f"[rotation] results: {len(rotating_in)} rotating-in, "
          f"{len(accelerating)} accelerating, {len(rotating_out)} rotating-out, "
          f"{len(decelerating)} decelerating")
    print(f"[rotation] macro: {macro}")

    for s in signals[:10]:
        emoji = {"ROTATING_IN": "🔄➡️", "ACCELERATING": "⚡",
                 "ROTATING_OUT": "🔄⬅️", "DECELERATING": "📉"}.get(s.signal, "")
        print(f"  {emoji} {s.ticker:5s} ({s.name:18s})  {s.signal:14s}  "
              f"str={s.strength:.1f}  5dRS={s.rs_5d*100:+.1f}%  "
              f"accel={s.rs_accel*100:+.1f}%  vol={s.vol_ratio_5d:.1f}x")

    return signals, macro
