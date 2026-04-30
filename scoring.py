"""Weighted scoring + A/B tier classification.

Inputs: dicts of computed signals. Output: total_score, tier, reasons[].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScoreResult:
    ticker: str
    total: int
    max_possible: int
    tier: str                  # "A", "B", or "reject"
    reasons_pro: List[str] = field(default_factory=list)
    reasons_con: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


def score_ticker(
    ticker: str,
    trend: Dict,
    momentum: Dict,
    volume: Dict,
    volatility: Dict,
    options: Dict,
    fundamentals: Dict,
    events: Dict,
    macro: Dict,
    weights: Dict,
    thresholds: Dict,
    tiers: Dict,
) -> ScoreResult:
    total = 0
    pros: List[str] = []
    cons: List[str] = []

    def award(key: str, condition: bool, pro_msg: str, con_msg: str = "") -> None:
        nonlocal total
        w = int(weights.get(key, 0))
        if condition:
            total += w
            pros.append(f"+{w} {pro_msg}")
        elif con_msg:
            cons.append(f"  0 {con_msg}")

    # Trend
    award("trend_above_200d", bool(trend.get("above_200d")),
          "above 200d SMA", "below 200d SMA")
    award("weekly_hh_hl", bool(trend.get("weekly_hhhl")),
          "weekly HH/HL intact")
    rs = trend.get("rs_vs_spy")
    award("rs_vs_spy_positive", rs is not None and rs > 0,
          f"RS vs SPY +{(rs or 0)*100:.1f}%", "RS vs SPY negative" if rs is not None else "")
    d52 = trend.get("dist_from_52w_high")
    healthy = (
        d52 is not None
        and thresholds["dist_from_52w_high_min"] <= d52 <= thresholds["dist_from_52w_high_max"]
    )
    award("dist_from_52w_high_healthy", healthy,
          f"{(d52 or 0)*100:.0f}% off 52w high (healthy)",
          f"{(d52 or 0)*100:.0f}% off 52w high (too far / too close)" if d52 is not None else "")

    # Momentum
    rsi_or_div = bool(momentum.get("rsi_oversold") or momentum.get("bullish_divergence"))
    rsi_val = momentum.get("rsi")
    award("rsi_oversold_or_div", rsi_or_div,
          f"RSI {rsi_val:.0f} oversold/divergent" if rsi_val else "RSI setup",
          f"RSI {rsi_val:.0f} not oversold" if rsi_val else "")
    award("macd_curling_up",
          bool(momentum.get("macd_curling_up") or momentum.get("macd_hist_rising")),
          "MACD curling up")

    # Volume
    award("volume_dryup", bool(volume.get("drying_up")),
          f"volume drying up ({(volume.get('volume_ratio_20d') or 0):.2f}x avg)")

    # Volatility
    iv_cheap = bool(volatility.get("iv_cheap"))
    basis = volatility.get("ivr_basis", "hv_proxy")
    ivr = volatility.get("ivr")
    hvr = volatility.get("hv_rank")
    rank_str = f"IVR {ivr:.0f}" if ivr is not None else f"HV-rank {hvr:.0f}" if hvr is not None else "?"
    award("iv_cheap", iv_cheap,
          f"{rank_str} ({basis}) — IV cheap/moderate",
          f"{rank_str} ({basis}) — IV too rich")
    award("iv_skew_ok", bool(volatility.get("skew_ok")),
          f"skew ok (25d P-C IV diff {(volatility.get('skew_25d') or 0):.2f})")
    award("term_structure_ok", bool(volatility.get("term_structure_ok")),
          "term structure ok")

    # Options liquidity
    award("options_liquid", bool(options.get("liquid_ok")),
          f"LEAPS liquid (OI {options.get('oi')}, sp {(options.get('spread_pct') or 0)*100:.1f}%)",
          f"LEAPS illiquid (OI {options.get('oi')}, sp {(options.get('spread_pct') or 0)*100:.1f}%)")

    # Fundamentals
    award("fundamentals_ok", bool(fundamentals.get("fundamentals_ok")),
          "fundamentals ok",
          "fundamentals weak")

    # Events
    dte_earn = events.get("days_to_earnings")
    ff_earn = bool(events.get("far_from_earnings"))
    award("no_near_earnings", ff_earn,
          f"no earnings risk (next in {dte_earn}d)" if dte_earn else "no earnings risk known",
          f"earnings in {dte_earn}d — IV crush risk" if dte_earn is not None else "")

    # Sector RS
    sector_rs = macro.get("sector_rs")
    award("sector_rs_positive", sector_rs is not None and sector_rs > 0,
          f"sector RS +{(sector_rs or 0)*100:.1f}%")

    # Macro regime
    award("macro_regime_ok", bool(macro.get("regime_ok")),
          "macro regime ok (SPY>200d, HYG stable)",
          "macro regime weak")

    max_possible = sum(int(w) for w in weights.values())

    # Tier decision. Also require a minimum set of "primary" confirmations.
    primary_confirmed = sum([
        bool(trend.get("above_200d")),
        bool(iv_cheap),
        bool(options.get("liquid_ok")),
        bool(events.get("far_from_earnings")),
    ])

    if total >= tiers["A_min_score"] and primary_confirmed >= 4:
        tier = "A"
    elif total >= tiers["B_min_score"] and primary_confirmed >= 3:
        tier = "B"
    else:
        tier = "reject"

    return ScoreResult(
        ticker=ticker,
        total=total,
        max_possible=max_possible,
        tier=tier,
        reasons_pro=pros,
        reasons_con=cons,
        details={
            "trend": trend,
            "momentum": momentum,
            "volume": volume,
            "volatility": volatility,
            "options": options,
            "fundamentals": fundamentals,
            "events": events,
            "macro": macro,
        },
    )
