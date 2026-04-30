"""Signal computations. Each function returns a dict the scorer can consume.

All functions tolerate partial/empty inputs by returning sensible defaults
(None or False) so downstream scoring can still run.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import OptionsChain


# ----------------------------- Math helpers --------------------------------

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    dn = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def bs_delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta. T in years, sigma annualized."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


# ----------------------------- Trend ---------------------------------------

def compute_trend(bars: pd.DataFrame, spy_bars: pd.DataFrame, rs_lookback: int = 90) -> Dict:
    """Trend / structure signals."""
    if bars.empty or len(bars) < 210:
        return {"above_200d": False, "weekly_hhhl": False, "rs_vs_spy": None,
                "dist_from_52w_high": None, "price": None}

    close = bars["Close"]
    sma200 = close.rolling(200).mean()
    above_200d = bool(close.iloc[-1] > sma200.iloc[-1])

    # Weekly HH/HL: resample to weekly, check last 3 weeks
    wk = close.resample("W-FRI").last().dropna()
    weekly_hhhl = False
    if len(wk) >= 4:
        w = wk.iloc[-4:].values
        weekly_hhhl = bool(w[-1] > w[-2] > w[-3])

    # RS vs SPY over lookback
    rs_vs_spy = None
    if not spy_bars.empty and len(spy_bars) > rs_lookback:
        try:
            ticker_ret = close.iloc[-1] / close.iloc[-rs_lookback] - 1
            spy_ret = spy_bars["Close"].iloc[-1] / spy_bars["Close"].iloc[-rs_lookback] - 1
            rs_vs_spy = float(ticker_ret - spy_ret)
        except Exception:
            pass

    # Distance from 52-week high
    dist = None
    if len(close) >= 252:
        hi52 = close.iloc[-252:].max()
        dist = float((hi52 - close.iloc[-1]) / hi52)

    return {
        "above_200d": above_200d,
        "weekly_hhhl": weekly_hhhl,
        "rs_vs_spy": rs_vs_spy,
        "dist_from_52w_high": dist,
        "price": float(close.iloc[-1]),
    }


# ----------------------------- Momentum ------------------------------------

def compute_momentum(bars: pd.DataFrame, rsi_oversold: float = 35) -> Dict:
    if bars.empty or len(bars) < 50:
        return {"rsi": None, "rsi_oversold": False, "bullish_divergence": False,
                "macd_curling_up": False, "macd_hist_rising": False}

    close = bars["Close"]
    rsi = _rsi(close)
    macd, sig, hist = _macd(close)

    cur_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None
    rsi_os = bool(cur_rsi is not None and cur_rsi < rsi_oversold)

    # Bullish divergence over last 30 bars: price made lower low, RSI made higher low
    bull_div = False
    if len(close) >= 30 and not rsi.iloc[-30:].isna().all():
        recent_lows_price = close.iloc[-30:].rolling(5).min().dropna()
        recent_lows_rsi = rsi.iloc[-30:].rolling(5).min().dropna()
        if len(recent_lows_price) >= 2 and len(recent_lows_rsi) >= 2:
            if (recent_lows_price.iloc[-1] < recent_lows_price.iloc[0]
                    and recent_lows_rsi.iloc[-1] > recent_lows_rsi.iloc[0]):
                bull_div = True

    # MACD curling up: hist negative but rising for 2+ sessions, or bullish cross
    macd_curling = False
    macd_hist_rising = False
    if len(hist.dropna()) >= 3:
        h = hist.iloc[-3:].values
        macd_hist_rising = bool(h[-1] > h[-2])
        if h[-1] < 0 and h[-1] > h[-2] > h[-3]:
            macd_curling = True
        if macd.iloc[-2] < sig.iloc[-2] and macd.iloc[-1] > sig.iloc[-1]:
            macd_curling = True

    return {
        "rsi": cur_rsi,
        "rsi_oversold": rsi_os,
        "bullish_divergence": bull_div,
        "macd_curling_up": macd_curling,
        "macd_hist_rising": macd_hist_rising,
    }


# ----------------------------- Volume --------------------------------------

def compute_volume(bars: pd.DataFrame, dryup_ratio: float = 0.85) -> Dict:
    if bars.empty or len(bars) < 30:
        return {"volume_ratio_20d": None, "drying_up": False}
    vol = bars["Volume"]
    avg20 = vol.iloc[-20:].mean()
    if avg20 == 0 or np.isnan(avg20):
        return {"volume_ratio_20d": None, "drying_up": False}
    ratio = float(vol.iloc[-1] / avg20)
    # "Drying up on selloff": ratio < dryup_ratio AND last close < close 5 days ago
    selling_off = bars["Close"].iloc[-1] < bars["Close"].iloc[-5] if len(bars) > 5 else False
    drying_up = bool(ratio < dryup_ratio and (selling_off or ratio < 0.7))
    return {"volume_ratio_20d": ratio, "drying_up": drying_up}


# ----------------------------- Volatility ----------------------------------

def _hist_vol(close: pd.Series, window: int = 30) -> float:
    """Annualized historical volatility."""
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < window:
        return float("nan")
    return float(rets.iloc[-window:].std() * np.sqrt(252))


def _hv_series(close: pd.Series, window: int = 30) -> pd.Series:
    rets = np.log(close / close.shift(1))
    return rets.rolling(window).std() * np.sqrt(252)


def compute_volatility(
    bars: pd.DataFrame,
    chain: Optional[OptionsChain],
    front_iv: Optional[float],
    atm_iv_history: Optional[pd.Series],
) -> Dict:
    """IV rank (from stored history if available, else HV rank as proxy),
    skew check, term-structure check, IV/HV ratio."""
    out: Dict = {
        "hv30": None,
        "hv_rank": None,            # 0-100 pct-rank of current HV over 252d
        "ivr": None,                # 0-100 true IVR from stored history (or None)
        "ivr_basis": "hv_proxy",    # "true" once history is sufficient
        "iv_cheap": False,          # in [ivr_min, ivr_max] OR HV-rank says so
        "iv_near_52w_low": False,
        "atm_iv": chain.atm_iv if chain else None,
        "skew_25d": None,           # 25d put IV - 25d call IV
        "skew_ok": False,
        "term_structure_ok": False,
        "iv_hv_ratio": None,
    }

    if bars.empty or len(bars) < 60:
        return out

    close = bars["Close"]
    hv30 = _hist_vol(close, 30)
    out["hv30"] = hv30

    # HV rank over last ~252d
    hv_series = _hv_series(close, 30).dropna()
    if len(hv_series) >= 60:
        window = hv_series.iloc[-252:] if len(hv_series) >= 252 else hv_series
        if not np.isnan(hv30):
            rank = float((window < hv30).mean() * 100.0)
            out["hv_rank"] = rank

    # True IVR if we have >=60 samples of stored ATM IV
    if atm_iv_history is not None and len(atm_iv_history.dropna()) >= 60:
        hist = atm_iv_history.dropna().iloc[-252:]
        cur = float(chain.atm_iv) if (chain and chain.atm_iv) else float(hist.iloc[-1])
        if hist.max() > hist.min():
            ivr = float((cur - hist.min()) / (hist.max() - hist.min()) * 100.0)
            out["ivr"] = ivr
            out["ivr_basis"] = "true"
            out["iv_near_52w_low"] = bool(ivr <= 20)

    # Cheap IV decision — prefer IVR, fall back to HV rank
    basis_rank = out["ivr"] if out["ivr"] is not None else out["hv_rank"]
    if basis_rank is not None:
        out["iv_cheap"] = bool(30 <= basis_rank <= 50 or basis_rank < 30)

    # IV/HV ratio (current ATM IV vs 30d HV). <1.2 = not expensive.
    if chain and chain.atm_iv and not np.isnan(hv30) and hv30 > 0:
        ratio = chain.atm_iv / hv30
        out["iv_hv_ratio"] = float(ratio)

    # 25-delta skew: put IV - call IV at the 25d strike in LEAPS expiry.
    # Approx by pulling IV at strikes whose delta ≈ 0.25 (call) and -0.25 (put=0.75 call-delta-equiv).
    if chain and not chain.calls.empty and not chain.puts.empty:
        try:
            S = chain.spot
            T = chain.dte / 365.0
            r = float(__import__("os").environ.get("RISK_FREE_RATE", 0.045))
            # 25-delta call: find call whose BS delta is ~0.25
            def pick_by_delta(df: pd.DataFrame, target: float) -> Optional[pd.Series]:
                best = None
                best_diff = 1e9
                for _, row in df.iterrows():
                    k = float(row["strike"])
                    iv = float(row.get("impliedVolatility") or 0)
                    if iv <= 0:
                        continue
                    d = bs_delta_call(S, k, T, r, iv)
                    if np.isnan(d):
                        continue
                    diff = abs(d - target)
                    if diff < best_diff:
                        best_diff = diff
                        best = row
                return best
            c25 = pick_by_delta(chain.calls, 0.25)
            p25 = pick_by_delta(chain.puts, 0.75)  # put with call-delta-equiv 0.75 ≈ 25d put
            if c25 is not None and p25 is not None:
                call_iv = float(c25.get("impliedVolatility") or 0)
                put_iv = float(p25.get("impliedVolatility") or 0)
                if call_iv and put_iv:
                    skew = put_iv - call_iv
                    out["skew_25d"] = float(skew)
                    # Normal skew: puts 2-8 vol points above calls. "Heavily tilted
                    # against calls" would be skew > 0.10 (very fear-bid puts) or call IV > put IV
                    # meaning calls expensive relative to puts (blow-off top). We want moderate.
                    out["skew_ok"] = bool(-0.02 <= skew <= 0.08)
        except Exception:
            pass

    # Term structure: back-month (LEAPS) IV not much higher than front-month.
    if chain and chain.atm_iv and front_iv:
        # Healthy: front >= back (normal backwardation-ish for single stocks)
        # Bad for LEAPS buyers: back >> front (contango blown out).
        out["term_structure_ok"] = bool(front_iv >= chain.atm_iv * 0.95)

    return out


# ----------------------------- Options liquidity ---------------------------

def compute_options(
    chain: Optional[OptionsChain],
    thresholds: Dict,
) -> Dict:
    if chain is None or chain.calls.empty:
        return {"leaps_expiry": None, "leaps_strike": None, "oi": None,
                "spread_pct": None, "delta": None, "liquid_ok": False, "dte": None}

    S = chain.spot
    T = chain.dte / 365.0
    import os
    r = float(os.environ.get("RISK_FREE_RATE", 0.045))
    dmin = thresholds.get("delta_target_min", 0.55)
    dmax = thresholds.get("delta_target_max", 0.85)

    # For each call, compute delta; pick one in target band with highest OI
    best = None
    for _, row in chain.calls.iterrows():
        iv = float(row.get("impliedVolatility") or 0)
        if iv <= 0:
            continue
        k = float(row["strike"])
        d = bs_delta_call(S, k, T, r, iv)
        if np.isnan(d) or not (dmin <= d <= dmax):
            continue
        oi = int(row.get("openInterest") or 0)
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        mid = (bid + ask) / 2.0 if (bid and ask) else 0
        spread_pct = (ask - bid) / mid if mid > 0 else float("inf")
        score = oi  # maximize OI among those in delta band
        if best is None or score > best["score"]:
            best = dict(strike=k, delta=d, iv=iv, oi=oi,
                        spread_pct=spread_pct, bid=bid, ask=ask, score=score)

    if best is None:
        return {"leaps_expiry": chain.expiry, "leaps_strike": None, "oi": None,
                "spread_pct": None, "delta": None, "liquid_ok": False, "dte": chain.dte}

    liquid_ok = bool(
        best["oi"] >= thresholds.get("options_oi_min", 500)
        and best["spread_pct"] <= thresholds.get("options_spread_max_pct", 0.08)
    )
    return {
        "leaps_expiry": chain.expiry,
        "leaps_strike": best["strike"],
        "oi": best["oi"],
        "spread_pct": best["spread_pct"],
        "delta": best["delta"],
        "iv": best["iv"],
        "bid": best["bid"],
        "ask": best["ask"],
        "dte": chain.dte,
        "liquid_ok": liquid_ok,
    }


# ----------------------------- Fundamentals --------------------------------

def compute_fundamentals(fund, thresholds: Dict) -> Dict:
    de_max = thresholds.get("debt_to_equity_max", 2.0)
    fcf_pos = fund.free_cashflow is not None and fund.free_cashflow > 0
    de_ok = fund.debt_to_equity is None or fund.debt_to_equity <= de_max
    rev_ok = fund.revenue_growth is None or fund.revenue_growth > -0.02
    short_ok = fund.short_pct_of_float is None or fund.short_pct_of_float < 0.15
    overall = bool(fcf_pos and de_ok and rev_ok and short_ok)
    return {
        "fcf_positive": fcf_pos,
        "de_ok": de_ok,
        "revenue_growth_ok": rev_ok,
        "short_ok": short_ok,
        "fundamentals_ok": overall,
        "free_cashflow": fund.free_cashflow,
        "debt_to_equity": fund.debt_to_equity,
        "revenue_growth": fund.revenue_growth,
        "short_pct_of_float": fund.short_pct_of_float,
    }


# ----------------------------- Events --------------------------------------

def compute_events(fund, thresholds: Dict) -> Dict:
    days_to_earn = None
    if fund.next_earnings:
        dd = (fund.next_earnings.date() - datetime.now().date()).days
        days_to_earn = int(dd)
    far_from_earn = bool(
        days_to_earn is None or days_to_earn >= thresholds.get("days_to_earnings_min", 21)
    )
    return {
        "days_to_earnings": days_to_earn,
        "far_from_earnings": far_from_earn,
        "next_earnings": fund.next_earnings.isoformat() if fund.next_earnings else None,
    }


# ----------------------------- Macro ---------------------------------------

def compute_macro(
    spy_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
    hyg_bars: pd.DataFrame,
    sector_bars: pd.DataFrame,
    thresholds: Dict,
) -> Dict:
    out = {
        "vix": None, "vix_elevated": False, "vix_calming": False,
        "spy_above_200d": False, "hyg_stable": False, "sector_rs": None,
        "regime_ok": False,
    }
    # VIX elevated but calming
    if not vix_bars.empty and len(vix_bars) >= 10:
        vix = vix_bars["Close"]
        cur = float(vix.iloc[-1])
        sma5 = float(vix.iloc[-thresholds.get("vix_calming_lookback", 5):].mean())
        out["vix"] = cur
        out["vix_elevated"] = bool(cur >= thresholds.get("vix_elevated", 18))
        out["vix_calming"] = bool(cur < sma5)
    # SPY above 200d
    if not spy_bars.empty and len(spy_bars) >= 210:
        sma200 = spy_bars["Close"].rolling(200).mean()
        out["spy_above_200d"] = bool(spy_bars["Close"].iloc[-1] > sma200.iloc[-1])
    # HYG stable (not in a crash): price within 3% of 20d high
    if not hyg_bars.empty and len(hyg_bars) >= 20:
        hi20 = float(hyg_bars["Close"].iloc[-20:].max())
        cur = float(hyg_bars["Close"].iloc[-1])
        out["hyg_stable"] = bool((hi20 - cur) / hi20 < 0.03)
    # Sector RS (sector ETF vs SPY over 90d)
    if not sector_bars.empty and not spy_bars.empty and len(sector_bars) > 90 and len(spy_bars) > 90:
        try:
            s_ret = sector_bars["Close"].iloc[-1] / sector_bars["Close"].iloc[-90] - 1
            b_ret = spy_bars["Close"].iloc[-1] / spy_bars["Close"].iloc[-90] - 1
            out["sector_rs"] = float(s_ret - b_ret)
        except Exception:
            pass
    # Regime: SPY above 200d AND HYG stable (don't require vix calming — that's upside)
    out["regime_ok"] = bool(out["spy_above_200d"] and (out["hyg_stable"] or not hyg_bars.empty is False))
    return out
