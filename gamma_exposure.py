"""Gamma exposure (GEX) — detect potential gamma squeeze setups.

When call OI piles up at near-the-money strikes, market-makers who sold
those calls are short gamma — they must BUY stock as price rises to stay
delta-neutral. That mechanical buying amplifies up moves. Roughly 30% of
retail squeezes (GME, AMC) are partial gamma squeezes.

Method (per ticker):
  1. Pull nearest 1-2 weekly expiries via yfinance (already in deps)
  2. For each strike, compute gamma via Black-Scholes
  3. Aggregate dollar gamma:  Σ OI × γ × 100 × S²/100
     (calls add positive, puts subtract — assumes dealers short calls / long puts)
  4. Find max-OI call strike (the "gamma magnet")
  5. Compute call/put OI ratio (skew toward calls = retail FOMO)

Output signals:
  - dollar_gex            : aggregate dollar gamma exposure
  - max_oi_call_strike    : strike with most call OI (price magnet)
  - magnet_distance_pct   : (max_oi_strike - spot) / spot
  - call_put_oi_ratio     : sum(call OI) / sum(put OI) on near-term chain
  - gamma_setup           : True if all three: heavy call OI bias,
                            magnet within +5% above spot, positive GEX
"""
from __future__ import annotations

import math
import os
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import yfinance as yf
from scipy.stats import norm

RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.045"))


def _safe_int(x, default: int = 0) -> int:
    """Coerce to int, handling None and NaN (yfinance returns NaN floats)."""
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        f = float(x)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma. Same for calls & puts.
    T in years; sigma annualized. Returns 0 on degenerate inputs.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def compute_gex(ticker: str, stock_price: float,
                max_expiries: int = 2) -> Optional[Dict]:
    """Compute gamma exposure across the nearest `max_expiries` expiries.

    Returns None if no chain available. Returns dict with the fields
    documented in the module docstring.
    """
    try:
        t = yf.Ticker(ticker)
        exps = list(t.options or [])[:max_expiries]
    except Exception as e:
        print(f"[gex] chain list error for {ticker}: {e}")
        return None
    if not exps or stock_price <= 0:
        return None

    today = date.today()
    total_gex   = 0.0
    call_oi_sum = 0
    put_oi_sum  = 0
    call_oi_by_strike: Dict[float, int] = {}

    for exp in exps:
        try:
            ch = t.option_chain(exp)
        except Exception as e:
            print(f"[gex] chain fetch error {ticker} {exp}: {e}")
            continue

        try:
            d_exp = date.fromisoformat(exp)
        except Exception:
            continue
        T = max((d_exp - today).days, 0) / 365.0
        if T <= 0:
            T = 1 / 365.0   # treat 0-DTE as 1 day to avoid div-by-zero

        # Restrict to ATM ±15% — beyond that gamma is negligible
        lo = stock_price * 0.85
        hi = stock_price * 1.15

        for df, sign in ((ch.calls, +1), (ch.puts, -1)):
            if df is None or df.empty:
                continue
            band = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
            for _, row in band.iterrows():
                K   = _safe_float(row.get("strike"))
                oi  = _safe_int(row.get("openInterest"))
                iv  = _safe_float(row.get("impliedVolatility"))
                if K <= 0 or oi <= 0 or iv <= 0:
                    continue

                gamma = _bs_gamma(stock_price, K, T, RISK_FREE, iv)
                # Dollar gamma per 1% move in spot:
                #   contracts × 100 shares/contract × gamma × spot²/100
                dollar_gamma = oi * 100.0 * gamma * (stock_price ** 2) / 100.0
                total_gex   += sign * dollar_gamma

                if sign > 0:
                    call_oi_sum += oi
                    call_oi_by_strike[K] = call_oi_by_strike.get(K, 0) + oi
                else:
                    put_oi_sum += oi

    if call_oi_sum == 0 and put_oi_sum == 0:
        return None

    # Gamma magnet = strike with most call OI in the ±15% band
    if call_oi_by_strike:
        max_strike  = max(call_oi_by_strike, key=call_oi_by_strike.get)
        max_oi      = call_oi_by_strike[max_strike]
        magnet_pct  = (max_strike - stock_price) / stock_price
    else:
        max_strike, max_oi, magnet_pct = None, 0, None

    cp_ratio = (call_oi_sum / put_oi_sum) if put_oi_sum > 0 else (
        float(call_oi_sum) if call_oi_sum > 0 else 0.0
    )

    # Gamma squeeze setup heuristic: all three must hold
    #   (1) call/put OI ratio ≥ 2.0 (real call bias)
    #   (2) magnet within 0%–10% above spot (room to run TO it)
    #   (3) positive GEX (overall short-gamma dealer pressure on upside)
    setup = bool(
        cp_ratio >= 2.0
        and magnet_pct is not None
        and 0.0 <= magnet_pct <= 0.10
        and total_gex > 0
    )

    return {
        "dollar_gex":          total_gex,            # in $
        "call_oi_sum":         call_oi_sum,
        "put_oi_sum":          put_oi_sum,
        "call_put_oi_ratio":   cp_ratio,
        "max_oi_call_strike":  max_strike,
        "max_oi_call_oi":      max_oi,
        "magnet_distance_pct": magnet_pct,
        "gamma_setup":         setup,
        "expiries_used":       exps,
    }
