"""Exit signal logic for open LEAPS positions.

Given the same signal dicts already computed by the entry scanner,
returns an ExitAlert if any close/trim condition is triggered.

Urgency levels:
  CLOSE NOW      — stop loss hit, or a primary entry condition has broken
  CONSIDER       — profit target hit, extended / risk is rising
  WATCH          — early warning, nothing actionable yet
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ExitAlert:
    ticker: str
    urgency: str                        # "CLOSE NOW" | "CONSIDER" | "WATCH"
    reasons: List[str] = field(default_factory=list)
    price: Optional[float] = None       # stock price
    dte: Optional[int] = None           # DTE on your specific contract
    entry_price: Optional[float] = None # what you paid per contract
    current_option_mid: Optional[float] = None
    pnl_pct: Optional[float] = None
    contracts: Optional[int] = None


def compute_exit_signals(
    ticker: str,
    trend: Dict,
    mom: Dict,
    vola: Dict,
    events: Dict,
    opts: Dict,
    position: Optional[Dict] = None,       # entry from positions.json
    current_option: Optional[Dict] = None, # live price from fetch_specific_option_price
) -> Optional[ExitAlert]:
    """Return an ExitAlert if any exit condition fires, else None."""

    close_now: List[str] = []
    consider: List[str] = []
    watch: List[str] = []

    stock_price = trend.get("price")
    rsi = mom.get("rsi")
    ivr = vola.get("ivr")
    hv_rank = vola.get("hv_rank")
    iv_rank = ivr if ivr is not None else hv_rank
    iv_basis = "IVR" if ivr is not None else "HV-rank"
    dist = trend.get("dist_from_52w_high")
    days_to_earn = events.get("days_to_earnings")

    # ------------------------------------------------------------------ #
    #  P&L signals — only when position details are provided              #
    # ------------------------------------------------------------------ #
    entry_price = None
    current_mid = None
    pnl_pct = None
    contracts = None
    position_dte = None

    if position:
        entry_price = position.get("entry_price")
        contracts = position.get("contracts", 1)
        profit_target_pct = position.get("profit_target_pct", 75)
        stop_loss_pct = position.get("stop_loss_pct", 40)
        dte_alert = position.get("dte_alert", 90)

        # Live P&L from current option price
        if current_option and entry_price and entry_price > 0:
            current_mid = current_option.get("mid", 0)
            if current_mid > 0:
                pnl_pct = (current_mid - entry_price) / entry_price * 100
                dollar_pnl = (current_mid - entry_price) * 100 * contracts
                pnl_str = f"+{pnl_pct:.0f}%" if pnl_pct >= 0 else f"{pnl_pct:.0f}%"
                dollar_str = f"+${dollar_pnl:,.0f}" if dollar_pnl >= 0 else f"-${abs(dollar_pnl):,.0f}"

                if pnl_pct >= profit_target_pct:
                    consider.append(
                        f"💰 up {pnl_str} ({dollar_str}) — profit target of {profit_target_pct:.0f}% hit"
                    )
                elif pnl_pct <= -stop_loss_pct:
                    close_now.append(
                        f"🛑 down {pnl_str} ({dollar_str}) — stop loss of {stop_loss_pct:.0f}% hit"
                    )

        # DTE on your specific contract
        expiry = position.get("expiry")
        if expiry:
            try:
                position_dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.now().date()).days
                if position_dte <= dte_alert:
                    watch.append(
                        f"👁 DTE {position_dte}d on your {expiry} contract — theta accelerating, plan to roll or close"
                    )
            except ValueError:
                pass

    # ------------------------------------------------------------------ #
    #  CLOSE NOW — a core entry thesis has broken                         #
    # ------------------------------------------------------------------ #

    if trend.get("above_200d") is False:
        close_now.append("❌ broke below 200d SMA — entry thesis gone")

    if days_to_earn is not None and 0 <= days_to_earn <= 14:
        close_now.append(f"❌ earnings in {days_to_earn}d — close before IV crush")

    if trend.get("weekly_hhhl") is False and trend.get("above_200d") is False:
        close_now.append("❌ weekly HH/HL structure broken")

    # ------------------------------------------------------------------ #
    #  CONSIDER CLOSING — extended or risk rising                         #
    # ------------------------------------------------------------------ #

    if dist is not None and dist <= 0.02:
        consider.append(f"📈 within {dist*100:.1f}% of 52w high — consider booking gains")

    if rsi is not None and rsi >= 80:
        consider.append(f"📈 RSI {rsi:.0f} — overbought, momentum likely to stall")

    if iv_rank is not None and iv_rank >= 65:
        consider.append(f"📈 {iv_basis} {iv_rank:.0f} — IV elevated, LEAPS premium rich, sell into it")

    if trend.get("weekly_hhhl") is False and trend.get("above_200d") is True:
        consider.append("⚠️ weekly HH/HL no longer intact — uptrend structure weakening")

    if days_to_earn is not None and 14 < days_to_earn <= 21:
        consider.append(f"⚠️ earnings in {days_to_earn}d — plan your close or hedge now")

    # ------------------------------------------------------------------ #
    #  WATCH — early warning, nothing actionable yet                      #
    # ------------------------------------------------------------------ #

    if mom.get("macd_hist_rising") is False and rsi is not None and rsi >= 70:
        watch.append(f"👁 MACD histogram declining (RSI {rsi:.0f}) — momentum fading at highs")

    # Fallback DTE check from scanner opts (if no position detail)
    if position_dte is None:
        scanner_dte = opts.get("dte")
        if scanner_dte is not None and scanner_dte <= 90:
            watch.append(f"👁 DTE {scanner_dte}d — theta accelerating, plan to roll or close")

    rs = trend.get("rs_vs_spy")
    if rs is not None and rs < 0:
        watch.append(f"👁 RS vs SPY turned negative ({rs*100:+.1f}%) — underperforming market")

    # ------------------------------------------------------------------ #
    #  Decide urgency                                                      #
    # ------------------------------------------------------------------ #
    if not (close_now or consider or watch):
        return None

    if close_now:
        urgency = "CLOSE NOW"
        reasons = close_now + consider + watch
    elif consider:
        urgency = "CONSIDER"
        reasons = consider + watch
    else:
        urgency = "WATCH"
        reasons = watch

    return ExitAlert(
        ticker=ticker,
        urgency=urgency,
        reasons=reasons,
        price=stock_price,
        dte=position_dte if position_dte is not None else opts.get("dte"),
        entry_price=entry_price,
        current_option_mid=current_mid,
        pnl_pct=pnl_pct,
        contracts=contracts,
    )
