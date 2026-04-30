"""Offline smoke test: builds synthetic bars + a fake options chain and runs
every signal + the scorer end-to-end. Catches regressions without needing
network access. Run: python _smoketest.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data import OptionsChain
from scoring import score_ticker
from signals import (
    compute_events,
    compute_fundamentals,
    compute_macro,
    compute_momentum,
    compute_options,
    compute_trend,
    compute_volatility,
    compute_volume,
    bs_delta_call,
)


def make_bars(n: int = 500, trend: float = 0.0003, vol: float = 0.02, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic OHLCV frame with a pullback near the end."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, vol, n)
    # Inject a ~10% pullback in the last 20 bars
    rets[-20:] = rng.normal(-0.003, vol * 0.8, 20)
    price = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(end="2026-04-17", periods=n)
    df = pd.DataFrame({
        "Open": price * (1 + rng.normal(0, 0.001, n)),
        "High": price * (1 + rng.uniform(0.001, 0.01, n)),
        "Low": price * (1 - rng.uniform(0.001, 0.01, n)),
        "Close": price,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    # Drying volume in last 5 bars
    df.loc[df.index[-5:], "Volume"] = df["Volume"].iloc[-25:-5].mean() * 0.6
    return df


def make_chain(spot: float) -> OptionsChain:
    strikes = np.arange(round(spot * 0.7), round(spot * 1.3), 5.0)
    def iv_for(k):
        # Mild skew: puts higher, ATM lowest, calls slightly higher far OTM
        moneyness = k / spot
        return 0.30 + 0.12 * (1 - moneyness) + 0.03 * max(0, moneyness - 1.0)
    calls = pd.DataFrame({
        "strike": strikes,
        "impliedVolatility": [iv_for(k) for k in strikes],
        "openInterest": np.random.randint(500, 20000, len(strikes)),
        "bid": [max(0.5, spot - k + 8) for k in strikes],
        "ask": [max(0.6, spot - k + 8.5) for k in strikes],
    })
    puts = calls.copy()
    puts["impliedVolatility"] = [iv_for(k) + 0.04 for k in strikes]
    return OptionsChain(
        ticker="TEST", expiry="2027-04-16", dte=363,
        spot=spot, calls=calls, puts=puts, atm_iv=0.32,
    )


class FakeFundamentals:
    free_cashflow = 5_000_000_000
    debt_to_equity = 0.4
    revenue_growth = 0.15
    short_pct_of_float = 0.03
    market_cap = 500_000_000_000
    sector = "Technology"
    next_earnings = None
    forward_div_yield = None


def main() -> int:
    with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
        cfg = json.load(f)

    bars = make_bars()
    spy_bars = make_bars(seed=11, trend=0.0004, vol=0.012)
    vix_bars = make_bars(seed=13, trend=-0.0001, vol=0.03)
    # VIX elevated but calming
    vix_bars["Close"] = 20 + 3 * np.sin(np.linspace(0, 10, len(vix_bars)))
    vix_bars.loc[vix_bars.index[-5:], "Close"] = [22, 21.5, 21, 20.5, 20.2]
    hyg_bars = make_bars(seed=17, trend=0.0001, vol=0.005)
    sector_bars = make_bars(seed=19, trend=0.0005, vol=0.015)

    spot = float(bars["Close"].iloc[-1])
    chain = make_chain(spot)
    front_iv = 0.35  # front >= back → term structure ok
    iv_hist = pd.Series(
        0.28 + 0.08 * np.random.default_rng(21).random(100),
        index=pd.bdate_range(end="2026-04-17", periods=100),
    )

    trend = compute_trend(bars, spy_bars, cfg["thresholds"]["rs_lookback_days"])
    mom = compute_momentum(bars, cfg["thresholds"]["rsi_oversold"])
    vol = compute_volume(bars, cfg["thresholds"]["volume_dryup_ratio"])
    vola = compute_volatility(bars, chain, front_iv, iv_hist)
    opts = compute_options(chain, cfg["thresholds"])
    fund = compute_fundamentals(FakeFundamentals(), cfg["thresholds"])
    events = compute_events(FakeFundamentals(), cfg["thresholds"])
    macro = compute_macro(spy_bars, vix_bars, hyg_bars, sector_bars, cfg["thresholds"])

    r = score_ticker(
        "TEST", trend, mom, vol, vola, opts, fund, events, macro,
        cfg["weights"], cfg["thresholds"], cfg["tiers"],
    )

    # Assertions
    assert r.max_possible == sum(cfg["weights"].values()), "max_possible mismatch"
    assert r.tier in {"A", "B", "reject"}, f"bad tier {r.tier}"
    assert opts["leaps_strike"] is not None, "options picker returned no contract"
    assert 0.55 <= opts["delta"] <= 0.85, f"delta out of band: {opts['delta']}"
    assert vola["hv_rank"] is not None or vola["ivr"] is not None, "no IV ranking"
    assert trend["above_200d"] in (True, False), "above_200d must be boolean"

    # BS delta sanity: deep ITM ~1.0, deep OTM ~0.0
    assert bs_delta_call(100, 50, 1.0, 0.045, 0.3) > 0.95
    assert bs_delta_call(100, 200, 1.0, 0.045, 0.3) < 0.05

    print(f"[smoke] score={r.total}/{r.max_possible} tier={r.tier}")
    print(f"[smoke] chose call: strike={opts['leaps_strike']} Δ={opts['delta']:.2f} "
          f"OI={opts['oi']} spread={opts['spread_pct']*100:.1f}%")
    print(f"[smoke] IV rank basis={vola['ivr_basis']} IVR={vola['ivr']} HVR={vola['hv_rank']}")
    print("[smoke] PROS:")
    for p in r.reasons_pro:
        print("  " + p)
    if r.reasons_con:
        print("[smoke] CONS:")
        for c in r.reasons_con:
            print("  " + c)
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
