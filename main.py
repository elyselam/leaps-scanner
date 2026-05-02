"""LEAPS Recommender — main scanner.

Usage:
  python main.py                    # scan once, post to Discord if webhook set
  python main.py --dry-run          # scan once, print only (no Discord)
  python main.py --loop             # run continuously on scan_interval_minutes
  python main.py --tickers NVDA MU  # override watchlist ad-hoc

Reads config.json for watchlist, thresholds, weights, tiers.
Reads .env for DISCORD_WEBHOOK_URL and optional POLYGON_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import pytz
from dotenv import load_dotenv

from data import (
    fetch_bars_batch,
    fetch_leaps_chain,
    fetch_front_month_iv,
    fetch_fundamentals,
    fetch_specific_option_price,
    PolygonClient,
)
from discord_sender import (
    send_alerts, send_digest, send_meme_alerts,
    send_rotation_alerts, send_sell_alerts, send_weekly_alerts,
)
from exit_signals import compute_exit_signals, ExitAlert
from market_scanner import scan_market
from meme_scanner import scan_meme
from rotation_scanner import scan_rotation
from weekly_scanner import scan_weekly
from scoring import ScoreResult, score_ticker
from signals import (
    compute_events,
    compute_fundamentals,
    compute_macro,
    compute_momentum,
    compute_options,
    compute_trend,
    compute_volatility,
    compute_volume,
)
from state import (
    compute_channel_hash,
    compute_setup_hash,
    load_iv_history,
    mark_alerted,
    record_iv,
    was_recently_alerted,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
ET = pytz.timezone("America/New_York")


def load_config() -> Dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def resolve_watchlist(cfg: Dict, override: List[str] | None) -> List[str]:
    if override:
        return [t.upper() for t in override]
    wl = list(cfg.get("watchlist", []))
    if cfg.get("use_suggested_additions"):
        wl += cfg.get("suggested_additions", [])
    # Dedupe preserving order
    seen, out = set(), []
    for t in wl:
        tu = t.upper()
        if tu not in seen:
            seen.add(tu)
            out.append(tu)
    return out


def _check_webhook(name: str, url: str | None, channel: str, dry_run: bool) -> None:
    """Loud warning if a Discord webhook env var is missing on a real run.

    Without this, a missing/typo'd secret produces a green workflow with no
    Discord posts and no error — symptom is "channel went silent" with no
    obvious cause. Print a banner at the top of the scanner branch so the
    misconfiguration is the first thing you see in the run log.
    """
    if dry_run:
        return
    if not url:
        print(f"[{channel}] ⚠️  {name} not set — Discord posts will be SKIPPED. "
              f"Set it as a GitHub Actions secret (Settings → Secrets → Actions).")
    else:
        # Last 6 chars only — enough to confirm in logs without leaking the URL
        tail = url[-6:] if len(url) >= 6 else "***"
        print(f"[{channel}] {name} configured (...{tail})")


def is_market_hours() -> bool:
    """Roughly 9:30–16:00 ET, Mon–Fri. Doesn't account for holidays."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= mins <= 16 * 60


def scan_once(cfg: Dict, watchlist: List[str], dry_run: bool) -> List[ScoreResult]:
    # Surface webhook config status up-front so a missing secret is the first
    # thing visible in the workflow log (not buried after a 5-min scan).
    leaps_webhook_early = os.getenv("LEAPS_ALERT_WEBHOOK_URL")
    sell_webhook_early  = os.getenv("SELL_ALERT_WEBHOOK_URL")
    _check_webhook("LEAPS_ALERT_WEBHOOK_URL", leaps_webhook_early, "leaps", dry_run)
    _check_webhook("SELL_ALERT_WEBHOOK_URL",  sell_webhook_early,  "leaps", dry_run)

    thresholds = cfg["thresholds"]
    weights = cfg["weights"]
    tiers = cfg["tiers"]
    sector_map = cfg.get("sector_map", {})

    # Context tickers: SPY for trend/RS, VIX for macro, HYG for credit, + every sector ETF we need
    context = ["SPY", "^VIX", "HYG"] + sorted(set(sector_map.get(t) for t in watchlist if sector_map.get(t)))
    all_tickers = list(dict.fromkeys(watchlist + context))

    print(f"[scan] fetching bars for {len(all_tickers)} tickers...")
    bars_map = fetch_bars_batch(all_tickers, period="2y")
    spy_bars = bars_map.get("SPY", bars_map.get("SPY", None))
    vix_bars = bars_map.get("^VIX", None)
    hyg_bars = bars_map.get("HYG", None)
    if spy_bars is None or spy_bars.empty:
        print("[scan] WARNING: SPY bars empty — RS and regime checks will be skipped")
        spy_bars = _empty()
    if vix_bars is None:
        vix_bars = _empty()
    if hyg_bars is None:
        hyg_bars = _empty()

    # Load open positions (only these get sell-signal checks)
    positions_path = os.path.join(HERE, "positions.json")
    open_positions: Dict = {}   # ticker -> position dict
    if os.path.exists(positions_path):
        with open(positions_path) as f:
            raw_pos = json.load(f)
        # Support both plain list of tickers and list of dicts
        for item in raw_pos:
            if isinstance(item, str):
                open_positions[item.upper()] = {}
            elif isinstance(item, dict) and item.get("ticker"):
                open_positions[item["ticker"].upper()] = item
        print(f"[scan] open positions ({len(open_positions)}): {', '.join(sorted(open_positions))}")

    results: List[ScoreResult] = []
    exit_alerts: List[ExitAlert] = []
    for t in watchlist:
        bars = bars_map.get(t)
        if bars is None or bars.empty:
            print(f"[scan] {t}: no bars, skipping")
            continue

        sector_etf = sector_map.get(t)
        sector_bars = bars_map.get(sector_etf, _empty()) if sector_etf else _empty()

        try:
            chain = fetch_leaps_chain(
                t,
                min_dte=thresholds["leaps_dte_min"],
                max_dte=thresholds["leaps_dte_max"],
            )
        except Exception as e:
            print(f"[scan] {t}: chain fetch error {e}")
            chain = None
        front_iv = None
        try:
            front_iv = fetch_front_month_iv(t)
        except Exception:
            pass

        if chain and chain.atm_iv:
            record_iv(t, chain.atm_iv)
        iv_hist = load_iv_history(t)

        fund = fetch_fundamentals(t)

        trend = compute_trend(bars, spy_bars, thresholds.get("rs_lookback_days", 90))
        mom = compute_momentum(bars, thresholds.get("rsi_oversold", 35))
        vol = compute_volume(bars, thresholds.get("volume_dryup_ratio", 0.85))
        vola = compute_volatility(bars, chain, front_iv, iv_hist)
        opts = compute_options(chain, thresholds)
        fundamentals = compute_fundamentals(fund, thresholds)
        events = compute_events(fund, thresholds)
        macro = compute_macro(spy_bars, vix_bars, hyg_bars, sector_bars, thresholds)

        r = score_ticker(
            t, trend, mom, vol, vola, opts, fundamentals, events, macro,
            weights, thresholds, tiers,
        )
        results.append(r)
        print(f"  {t}: {r.total:>3}/{r.max_possible} tier={r.tier}")

        # Exit signal check — only for tickers you're actually holding
        if t in open_positions:
            position = open_positions[t]
            current_option = None
            if position.get("strike") and position.get("expiry"):
                current_option = fetch_specific_option_price(
                    t,
                    position["expiry"],
                    float(position["strike"]),
                    position.get("option_type", "call"),
                )
            ea = compute_exit_signals(t, trend, mom, vola, events, opts, position, current_option)
            if ea:
                exit_alerts.append(ea)
                print(f"  {t}: EXIT SIGNAL [{ea.urgency}] — {ea.reasons[0]}")

    # Filter to non-reject, dedupe, then cap to TOP 5 by confidence score.
    # Mark-as-alerted only AFTER the cap so a setup that ranked 6th today
    # can still alert tomorrow when it might be in the top 5.
    dedupe_hours = cfg.get("dedupe_hours", 12)
    candidates: List[tuple] = []   # (ScoreResult, setup_hash)
    for r in results:
        if r.tier == "reject":
            continue
        opt = r.details.get("options", {}) or {}
        h = compute_setup_hash(r.tier, opt.get("leaps_strike"), opt.get("leaps_expiry"))
        if was_recently_alerted(r.ticker, h, dedupe_hours):
            print(f"  {r.ticker}: suppressed (recent alert within {dedupe_hours}h)")
            continue
        candidates.append((r, h))

    # Sort by total score desc, take top 5
    candidates.sort(key=lambda x: -x[0].total)
    top5 = candidates[:5]
    to_alert = [r for r, _ in top5]

    if candidates:
        print(f"[leaps] {len(candidates)} candidates after dedup, posting top {len(top5)}:")
        for r, _ in top5:
            print(f"  {r.ticker:6s} score={r.total}/{r.max_possible} tier={r.tier}")

    # Mark only the alerts we're actually posting
    if not dry_run:
        for r, h in top5:
            mark_alerted(r.ticker, h)

    # Post entry alerts → #leaps
    leaps_webhook = os.getenv("LEAPS_ALERT_WEBHOOK_URL")
    sell_webhook = os.getenv("SELL_ALERT_WEBHOOK_URL")
    send_alerts(leaps_webhook, to_alert, dry_run=dry_run)

    # Digest → #leaps
    send_digest(leaps_webhook, results, dry_run=dry_run)

    # Exit/close alerts → #sell-alerts
    if exit_alerts:
        send_sell_alerts(sell_webhook, exit_alerts, dry_run=dry_run)
    else:
        print("[discord] no exit signals for open positions")

    return results


def _empty():
    import pandas as pd
    return pd.DataFrame()


def main() -> int:
    load_dotenv(os.path.join(HERE, ".env"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print results, don't post to Discord")
    ap.add_argument("--loop",    action="store_true", help="run continuously on interval")
    ap.add_argument("--tickers", nargs="+",           help="override watchlist")
    ap.add_argument("--force",   action="store_true", help="ignore market-hours gate")
    ap.add_argument("--weekly",  action="store_true", help="run weekly options scanner (curated list)")
    ap.add_argument("--market",   action="store_true", help="run market-wide weekly scanner (S&P 500 + Nasdaq-100)")
    ap.add_argument("--rotation", action="store_true", help="run sector rotation detector")
    ap.add_argument("--meme",     action="store_true", help="run meme/squeeze/unusual-volume scanner")
    args = ap.parse_args()

    cfg = load_config()

    # ── weekly scanner ────────────────────────────────────────────────────
    if args.weekly:
        universe = cfg.get("weekly_universe", [])
        if args.tickers:
            universe = [t.upper() for t in args.tickers]
        weekly_webhook = os.getenv("WEEKLY_WEBHOOK_URL")
        _check_webhook("WEEKLY_WEBHOOK_URL", weekly_webhook, "weekly", args.dry_run)
        alerts = scan_weekly(universe, dry_run=args.dry_run)
        # Calls only — bullish setups only, drop PUTs before ranking.
        before = len(alerts)
        alerts = [a for a in alerts if a.direction == "CALL"]
        if before != len(alerts):
            print(f"[weekly] dropped {before - len(alerts)} PUT setups (calls-only mode)")
        # Pick the 10 highest-conviction setups across BOTH expiries by score,
        # then re-sort the survivors by (expiry, -score) so the Discord embed
        # still groups this-Friday vs next-Friday in its header.
        top10 = sorted(alerts, key=lambda a: -a.score)[:10]
        top10.sort(key=lambda a: (a.expiry, -a.score))
        if alerts:
            print(f"[weekly] {len(alerts)} setups found, posting top {len(top10)}:")
            for a in top10:
                print(f"  {a.ticker:6s} {a.direction:4s} ${a.strike:>7.2f} "
                      f"{a.expiry}  score={a.score}")
        # mode="BOTH" tells the Discord sender to group by expiry in the header
        send_weekly_alerts(weekly_webhook, top10, mode="BOTH", dry_run=args.dry_run)
        return 0

    # ── market-wide weekly scanner ────────────────────────────────────────
    if args.market:
        market_webhook = os.getenv("MARKET_WEBHOOK_URL")
        _check_webhook("MARKET_WEBHOOK_URL", market_webhook, "market", args.dry_run)
        alerts = scan_market(dry_run=args.dry_run)
        # Calls-only — bullish setups only
        before = len(alerts)
        alerts = [a for a in alerts if a.direction == "CALL"]
        if before != len(alerts):
            print(f"[market] dropped {before - len(alerts)} PUT setups (calls-only mode)")

        # Dedup against last 12h: same (ticker, strike, expiry) combo posted
        # recently is suppressed so the 15-min cadence doesn't spam unchanged
        # setups. The hash is channel-namespaced ("market") so it doesn't
        # cross-suppress with the curated weekly scanner.
        dedupe_hours = cfg.get("dedupe_hours", 12)
        candidates = []   # (alert, hash)
        for a in alerts:
            h = compute_channel_hash("market", a.ticker, a.direction, a.strike, a.expiry)
            if was_recently_alerted(a.ticker, h, dedupe_hours):
                continue
            candidates.append((a, h))
        suppressed = len(alerts) - len(candidates)
        if suppressed:
            print(f"[market] suppressed {suppressed} unchanged setups (last {dedupe_hours}h)")

        # Top 5 by score, then re-sort by (expiry, -score) for the embed grouping
        candidates.sort(key=lambda x: -x[0].score)
        top5 = candidates[:5]
        top5.sort(key=lambda x: (x[0].expiry, -x[0].score))
        to_post = [a for a, _ in top5]

        if to_post:
            print(f"[market] posting top {len(to_post)}:")
            for a in to_post:
                print(f"  {a.ticker:6s} {a.direction:4s} ${a.strike:>7.2f} "
                      f"{a.expiry}  score={a.score}")
        elif alerts:
            print("[market] all setups already alerted recently — nothing new to post")

        # Mark only what we're actually posting
        if not args.dry_run:
            for a, h in top5:
                mark_alerted(a.ticker, h)

        send_weekly_alerts(market_webhook, to_post, mode="BOTH", dry_run=args.dry_run)
        return 0

    # ── sector rotation detector ──────────────────────────────────────────
    if args.rotation:
        market_webhook = os.getenv("MARKET_WEBHOOK_URL")
        _check_webhook("MARKET_WEBHOOK_URL", market_webhook, "rotation", args.dry_run)
        signals, macro = scan_rotation(dry_run=args.dry_run)

        # Dedup: only re-post if the rotation picture changes.
        # Hash on the set of (ticker, signal_type) for ROTATING_IN and ROTATING_OUT —
        # those are the actionable ones. If the same sectors are still rotating
        # in/out, don't re-post. An acceleration change or a new entrant re-triggers.
        dedupe_hours = cfg.get("dedupe_hours", 12)
        actionable = [s for s in signals if s.signal in ("ROTATING_IN", "ROTATING_OUT")]
        # Build a single composite hash from all actionable signals
        sig_key = "|".join(sorted(f"{s.ticker}:{s.signal}" for s in actionable))
        rotation_hash = compute_channel_hash("rotation", sig_key)

        if was_recently_alerted("__ROTATION__", rotation_hash, dedupe_hours):
            print("[rotation] same rotation picture as last alert — suppressed")
        elif actionable:
            # Post all signals (including accelerating/decelerating for context)
            send_rotation_alerts(market_webhook, signals, macro, dry_run=args.dry_run)
            if not args.dry_run:
                mark_alerted("__ROTATION__", rotation_hash)
            print(f"[rotation] posted {len(signals)} signals to #market-scan")
        else:
            print("[rotation] no clear rotation signals today")
        return 0

    # ── meme / squeeze scanner ────────────────────────────────────────────
    if args.meme:
        universe = cfg.get("meme_universe", [])
        if args.tickers:
            universe = [t.upper() for t in args.tickers]
        meme_webhook = os.getenv("MEME_WEBHOOK_URL")
        _check_webhook("MEME_WEBHOOK_URL", meme_webhook, "meme", args.dry_run)
        alerts = scan_meme(universe, dry_run=args.dry_run)

        # Dedup: same (ticker, tier) within the last 12h is suppressed.
        # Tier change (WATCH→UNUSUAL→SQUEEZE) re-alerts because the hash
        # changes — that's a real status upgrade worth surfacing.
        dedupe_hours = cfg.get("dedupe_hours", 12)
        candidates       = []   # (alert, hash)
        suppressed_names = []   # for logging — see "where did GME go?"
        for a in alerts:
            h = compute_channel_hash("meme", a.ticker, a.tier)
            if was_recently_alerted(a.ticker, h, dedupe_hours):
                suppressed_names.append(f"{a.ticker}({a.tier})")
                continue
            candidates.append((a, h))
        suppressed = len(alerts) - len(candidates)
        if suppressed:
            print(f"[meme] suppressed {suppressed} unchanged setups (last {dedupe_hours}h): "
                  f"{', '.join(suppressed_names)}")

        # Top 5 by rally_score (alerts are pre-sorted but candidates may
        # have been re-ordered by dedup — re-sort to be safe)
        candidates.sort(key=lambda x: -x[0].rally_score)
        top5 = candidates[:5]
        to_post = [a for a, _ in top5]

        if to_post:
            print(f"[meme] posting top {len(to_post)}:")
            for a in to_post:
                print(f"  {a.ticker:6s}  rally={a.rally_score:+.1f}  "
                      f"score={a.score}  tier={a.tier}  "
                      f"1d={(a.ret_1d or 0)*100:+.1f}%")
        elif alerts:
            print("[meme] all setups already alerted recently — nothing new to post")

        if not args.dry_run:
            for a, h in top5:
                mark_alerted(a.ticker, h)

        send_meme_alerts(meme_webhook, to_post, dry_run=args.dry_run)
        return 0

    # ── LEAPS scanner (default) ───────────────────────────────────────────
    watchlist = resolve_watchlist(cfg, args.tickers)
    print(f"[init] watchlist ({len(watchlist)}): {', '.join(watchlist)}")

    interval = max(5, int(cfg.get("scan_interval_minutes", 45))) * 60

    if not args.loop:
        scan_once(cfg, watchlist, dry_run=args.dry_run)
        return 0

    print(f"[init] loop mode every {interval//60}m; market-hours-only={cfg.get('market_hours_only', True)}")
    while True:
        try:
            if not args.force and cfg.get("market_hours_only", True) and not is_market_hours():
                print(f"[loop] {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} — market closed, sleeping")
            else:
                scan_once(cfg, watchlist, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print("\n[loop] interrupted")
            return 0
        except Exception as e:
            print(f"[loop] scan error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
