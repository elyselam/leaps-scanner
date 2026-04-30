"""Discord webhook poster. Formats rich embeds for each alert."""
from __future__ import annotations

from typing import Dict, List, Optional

import requests

from exit_signals import ExitAlert
from meme_scanner import MemeAlert
from scoring import ScoreResult
from weekly_scanner import WeeklyAlert

TIER_COLOR   = {"A": 0x2ecc71, "B": 0xf1c40f, "reject": 0x95a5a6}
URGENCY_COLOR = {"CLOSE NOW": 0xe74c3c, "CONSIDER": 0xe67e22, "WATCH": 0x3498db}
URGENCY_EMOJI = {"CLOSE NOW": "🚨", "CONSIDER": "⚠️", "WATCH": "👁"}


# ── helpers ────────────────────────────────────────────────────────────────

def _strip_weight(reason: str) -> str:
    """'+10 above 200d SMA' → 'above 200d SMA'"""
    parts = reason.strip().split(" ", 1)
    return parts[1] if len(parts) == 2 else reason


def _post(webhook_url: str, payload: Dict) -> None:
    try:
        resp = requests.post(webhook_url, json=payload, timeout=15)
        if resp.status_code >= 300:
            print(f"[discord] HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[discord] post failed: {e}")


def _embed_char_count(e: Dict) -> int:
    """Approximate an embed's Discord char count (sum of title + description
    + every field name + value). Discord sums these across ALL embeds in a
    request and caps the total at 6000."""
    n = len(e.get("title", "")) + len(e.get("description", ""))
    for f in e.get("fields", []):
        n += len(f.get("name", "")) + len(f.get("value", ""))
    return n


def _post_embeds(webhook_url: str, embeds: List[Dict]) -> None:
    """Pack embeds into messages respecting Discord's two caps:
      - max 10 embeds per request
      - max 6000 total chars across embeds in one request
    """
    MAX_EMBEDS = 10
    MAX_CHARS  = 5800   # small safety margin under 6000
    batch: List[Dict] = []
    batch_chars = 0
    for e in embeds:
        ec = _embed_char_count(e)
        if batch and (len(batch) >= MAX_EMBEDS or batch_chars + ec > MAX_CHARS):
            _post(webhook_url, {"embeds": batch})
            batch, batch_chars = [], 0
        batch.append(e)
        batch_chars += ec
    if batch:
        _post(webhook_url, {"embeds": batch})


# ── entry alerts (#leaps) ──────────────────────────────────────────────────

def _entry_embed(r: ScoreResult) -> Dict:
    opt   = r.details.get("options",    {}) or {}
    vol   = r.details.get("volatility", {}) or {}
    trend = r.details.get("trend",      {}) or {}
    ev    = r.details.get("events",     {}) or {}

    price  = trend.get("price")
    strike = opt.get("leaps_strike")
    exp    = opt.get("leaps_expiry")
    delta  = opt.get("delta")
    oi     = opt.get("oi")
    spread = opt.get("spread_pct")
    dte    = opt.get("dte")
    ivr    = vol.get("ivr")
    hvr    = vol.get("hv_rank")
    iv_str = (f"IVR {ivr:.0f}" if ivr is not None
              else f"HV-rank {hvr:.0f}" if hvr is not None else "IV n/a")
    earn   = ev.get("days_to_earnings")

    # ── description: one compact stats line ──
    stats = []
    if price:  stats.append(f"**${price:.2f}**")
    stats.append(iv_str)
    if earn is not None: stats.append(f"earnings in {earn}d")
    description = "  ·  ".join(stats)

    fields: List[Dict] = [
        {"name": "Score", "value": f"**{r.total} / {r.max_possible}**", "inline": True},
        {"name": "Tier",  "value": f"**{r.tier}**",                     "inline": True},
    ]

    # Contract row
    if strike and exp:
        contract_val = f"`${strike:.0f}C  {exp}  ({dte}d)`"
        if delta and spread is not None:
            contract_val += f"\nΔ {delta:.2f}  ·  OI {oi:,}  ·  spread {spread*100:.1f}%"
        fields.append({"name": "📋 Contract", "value": contract_val, "inline": False})

    # Bull case — join as bullets to save vertical space
    pros = "  ·  ".join(_strip_weight(p) for p in r.reasons_pro[:8])
    fields.append({"name": "✅ Bull case", "value": pros or "—", "inline": False})

    # Caveats
    if r.reasons_con:
        cons = "  ·  ".join(_strip_weight(c) for c in r.reasons_con[:4])
        fields.append({"name": "⚠️ Watch", "value": cons, "inline": False})

    return {
        "title":       f"{r.ticker} — Tier {r.tier} LEAPS setup",
        "description": description,
        "color":       TIER_COLOR.get(r.tier, 0x95a5a6),
        "fields":      fields,
    }


def send_alerts(webhook_url: str, results: List[ScoreResult], dry_run: bool = False) -> None:
    if not results:
        print("[discord] no entry alerts to send")
        return
    if dry_run or not webhook_url:
        for r in results:
            opt   = r.details.get("options",    {}) or {}
            trend = r.details.get("trend",      {}) or {}
            vol   = r.details.get("volatility", {}) or {}
            ev    = r.details.get("events",     {}) or {}
            price  = trend.get("price")
            strike = opt.get("leaps_strike")
            exp    = opt.get("leaps_expiry")
            dte    = opt.get("dte")
            delta  = opt.get("delta")
            oi     = opt.get("oi")
            spread = opt.get("spread_pct")
            ivr    = vol.get("ivr")
            hvr    = vol.get("hv_rank")
            iv_str = f"IVR {ivr:.0f}" if ivr is not None else f"HV-rank {hvr:.0f}" if hvr is not None else "n/a"
            print(f"\n[DRY entry] {r.ticker} — Tier {r.tier}  {r.total}/{r.max_possible}")
            print(f"  ${price:.2f}  ·  {iv_str}  ·  earnings {ev.get('days_to_earnings')}d")
            if strike and exp:
                liq = f"  Δ{delta:.2f} OI {oi} spread {spread*100:.1f}%" if delta and spread is not None else ""
                print(f"  ${strike:.0f}C {exp} ({dte}d){liq}")
            print("  ✅ " + "  ·  ".join(_strip_weight(p) for p in r.reasons_pro[:8]))
            if r.reasons_con:
                print("  ⚠️ " + "  ·  ".join(_strip_weight(c) for c in r.reasons_con[:4]))
        return
    _post_embeds(webhook_url, [_entry_embed(r) for r in results])


# ── digest (#leaps) ────────────────────────────────────────────────────────

def _digest_embed(r: ScoreResult) -> Dict:
    opt   = r.details.get("options",    {}) or {}
    vol   = r.details.get("volatility", {}) or {}
    trend = r.details.get("trend",      {}) or {}
    ev    = r.details.get("events",     {}) or {}

    price  = trend.get("price")
    ivr    = vol.get("ivr")
    hvr    = vol.get("hv_rank")
    iv_str = (f"IVR {ivr:.0f}" if ivr is not None
              else f"HV-rank {hvr:.0f}" if hvr is not None else "IV n/a")
    strike = opt.get("leaps_strike")
    exp    = opt.get("leaps_expiry")
    dte    = opt.get("dte")
    earn   = ev.get("days_to_earnings")

    # compact one-liner under the title
    stats = []
    if price:  stats.append(f"**${price:.2f}**")
    stats.append(iv_str)
    if strike and exp: stats.append(f"${strike:.0f}C {exp} ({dte}d)")
    if earn is not None: stats.append(f"earnings {earn}d")
    description = "  ·  ".join(stats)

    pros = "  ·  ".join(_strip_weight(p) for p in r.reasons_pro[:6])
    cons = "  ·  ".join(_strip_weight(c) for c in r.reasons_con[:3])

    fields = [
        {"name": "Score", "value": f"**{r.total}/{r.max_possible}**", "inline": True},
        {"name": "Tier",  "value": f"**{r.tier}**",                   "inline": True},
        {"name": "✅", "value": pros or "—", "inline": False},
    ]
    if cons:
        fields.append({"name": "⚠️", "value": cons, "inline": False})

    return {
        "title":       r.ticker,
        "description": description,
        "color":       TIER_COLOR.get(r.tier, 0x95a5a6),
        "fields":      fields,
    }


def send_digest(webhook_url: str, results: List[ScoreResult], dry_run: bool = False) -> None:
    qualified = sorted(
        (r for r in results if r.tier != "reject"),
        key=lambda x: -x.total,
    )
    if not qualified:
        return

    if dry_run or not webhook_url:
        print("[DRY digest]")
        for r in qualified:
            opt   = r.details.get("options",    {}) or {}
            trend = r.details.get("trend",      {}) or {}
            vol   = r.details.get("volatility", {}) or {}
            ev    = r.details.get("events",     {}) or {}
            price = trend.get("price")
            ivr   = vol.get("ivr")
            hvr   = vol.get("hv_rank")
            iv_str = f"IVR {ivr:.0f}" if ivr is not None else f"HV-rank {hvr:.0f}" if hvr is not None else "n/a"
            strike = opt.get("leaps_strike")
            exp    = opt.get("leaps_expiry")
            dte    = opt.get("dte")
            earn   = ev.get("days_to_earnings")
            print(f"\n  {r.ticker}  {r.total}/{r.max_possible}  Tier {r.tier}")
            line = f"  ${price:.2f}  {iv_str}"
            if strike and exp: line += f"  ${strike:.0f}C {exp} ({dte}d)"
            if earn is not None: line += f"  earnings {earn}d"
            print(line)
            print("  ✅ " + "  ·  ".join(_strip_weight(p) for p in r.reasons_pro[:6]))
            if r.reasons_con:
                print("  ⚠️ " + "  ·  ".join(_strip_weight(c) for c in r.reasons_con[:3]))
        return

    # Header message + one embed per ticker
    _post(webhook_url, {"content": "**📊 LEAPS scan digest**"})
    _post_embeds(webhook_url, [_digest_embed(r) for r in qualified])


# ── sell alerts (#sell-alerts) ─────────────────────────────────────────────

def _sell_embed(a: ExitAlert) -> Dict:
    emoji = URGENCY_EMOJI.get(a.urgency, "")

    # description: stock price + P&L on one line
    desc_parts = []
    if a.price:
        desc_parts.append(f"Stock **${a.price:.2f}**")
    if a.dte is not None:
        desc_parts.append(f"DTE **{a.dte}d**")
    if a.pnl_pct is not None and a.entry_price and a.current_option_mid:
        sign   = "+" if a.pnl_pct >= 0 else ""
        dollar = (a.current_option_mid - a.entry_price) * 100 * (a.contracts or 1)
        d_str  = f"+${dollar:,.0f}" if dollar >= 0 else f"-${abs(dollar):,.0f}"
        desc_parts.append(
            f"P&L **{sign}{a.pnl_pct:.0f}% ({d_str})**  "
            f"${a.entry_price:.2f} → ${a.current_option_mid:.2f}"
        )
    description = "  ·  ".join(desc_parts)

    fields = [{"name": f"{emoji} Signals",
               "value": "\n".join(a.reasons)[:1024] or "—",
               "inline": False}]

    return {
        "title":       f"{emoji} {a.ticker} — {a.urgency}",
        "description": description,
        "color":       URGENCY_COLOR.get(a.urgency, 0x95a5a6),
        "fields":      fields,
    }


def send_sell_alerts(webhook_url: str, alerts: List[ExitAlert], dry_run: bool = False) -> None:
    if not alerts:
        return

    if dry_run or not webhook_url:
        for a in alerts:
            emoji = URGENCY_EMOJI.get(a.urgency, "")
            print(f"\n[DRY sell] {emoji} {a.urgency} — {a.ticker}", end="")
            if a.price: print(f"  ${a.price:.2f}", end="")
            if a.dte:   print(f"  DTE {a.dte}d", end="")
            if a.pnl_pct is not None and a.entry_price and a.current_option_mid:
                sign   = "+" if a.pnl_pct >= 0 else ""
                dollar = (a.current_option_mid - a.entry_price) * 100 * (a.contracts or 1)
                d_str  = f"+${dollar:,.0f}" if dollar >= 0 else f"-${abs(dollar):,.0f}"
                print(f"  P&L {sign}{a.pnl_pct:.0f}% ({d_str})  ${a.entry_price:.2f}→${a.current_option_mid:.2f}", end="")
            print()
            for reason in a.reasons:
                print(f"    {reason}")
        return

    _post_embeds(webhook_url, [_sell_embed(a) for a in alerts])


# ── weekly alerts (#weekly-alerts) ────────────────────────────────────────────

DIRECTION_COLOR = {"CALL": 0x2ecc71, "PUT": 0xe74c3c}
DIRECTION_EMOJI = {"CALL": "🟢", "PUT": "🔴"}


def _weekly_embed(a: WeeklyAlert) -> Dict:
    emoji    = DIRECTION_EMOJI.get(a.direction, "")
    opt_type = "C" if a.direction == "CALL" else "P"

    description = (
        f"Stock **${a.stock_price:.2f}**  ·  "
        f"Expires **{a.expiry}** ({a.dte} DTE)"
    )

    contract_val = (
        f"`${a.strike:.0f}{opt_type}  {a.expiry}  ({a.dte}d)`\n"
        f"Bid **${a.bid:.2f}**  ·  Ask **${a.ask:.2f}**  ·  Mid **${a.mid:.2f}**\n"
        f"OI {a.oi:,}  ·  spread {a.spread_pct*100:.1f}%"
    )

    reasons_val = "\n".join(f"• {r}" for r in a.reasons) or "—"

    fields = [
        {"name": "Score",     "value": f"**{a.score}**",          "inline": True},
        {"name": "Direction", "value": f"**{emoji} {a.direction}**", "inline": True},
        {"name": f"📋 Buy {a.direction}", "value": contract_val, "inline": False},
        {"name": "✅ Why this trade", "value": reasons_val[:1024], "inline": False},
    ]

    # ── Earnings catalyst ──
    if a.days_to_earnings is not None:
        if a.earnings_before_expiry:
            earn_val = f"📣 **{a.days_to_earnings}d** away — BEFORE expiry (binary catalyst)"
        elif a.days_to_earnings <= 14:
            earn_val = f"⚠️ **{a.days_to_earnings}d** away — after expiry (IV inflated)"
        else:
            earn_val = f"{a.days_to_earnings}d away — far out, no IV impact"
        fields.append({"name": "📅 Earnings", "value": earn_val, "inline": False})

    # ── News ──
    if a.news_sentiment is not None or a.headlines:
        sent = a.news_sentiment or 0
        if   sent >=  0.3: tag = "🟢 Bullish"
        elif sent >=  0.1: tag = "🟢 Mildly bullish"
        elif sent <= -0.3: tag = "🔴 Bearish"
        elif sent <= -0.1: tag = "🔴 Mildly bearish"
        else:              tag = "⚪ Neutral"
        if a.news_hot:
            tag += "  ·  🔥 hot"

        # Data-source attribution so you know where the signal came from
        src_str = ""
        if a.news_sources:
            src_str = f"  ·  *sources: {', '.join(a.news_sources)}*"
        header_bits = [f"**{tag}** (score {sent:+.2f})"]
        if a.news_article_count:
            header_bits.append(f"{a.news_article_count} articles")
        head_lines = ["  ·  ".join(header_bits) + src_str]
        for h in a.headlines[:3]:
            head_lines.append(f"• {h[:120]}")
        fields.append({"name": "📰 Recent news", "value": "\n".join(head_lines)[:1024], "inline": False})

    return {
        "title":       f"{emoji} {a.ticker} — Weekly {a.direction}",
        "description": description,
        "color":       DIRECTION_COLOR.get(a.direction, 0x95a5a6),
        "fields":      fields,
    }


def send_weekly_alerts(
    webhook_url: str,
    alerts: List[WeeklyAlert],
    mode: str = "",
    dry_run: bool = False,
) -> None:
    if not alerts:
        print("[discord] no weekly alerts to send")
        return

    # Group alerts by expiry so the header reads naturally
    by_expiry: Dict[str, List[WeeklyAlert]] = {}
    for a in alerts:
        by_expiry.setdefault(a.expiry, []).append(a)
    # Sorted ascending: nearer Friday first
    expiries_sorted = sorted(by_expiry.keys())

    if dry_run or not webhook_url:
        print(f"\n[DRY weekly] {'─'*60}")
        print(f"  {len(alerts)} setups across {len(expiries_sorted)} expir"
              f"{'y' if len(expiries_sorted) == 1 else 'ies'}")
        for exp in expiries_sorted:
            group = by_expiry[exp]
            print(f"\n  ━━ Expiring {exp} ({group[0].dte} DTE) · {len(group)} setups ━━")
            for a in group:
                opt_type = "C" if a.direction == "CALL" else "P"
                print(f"\n  {a.ticker}  {a.direction}  ${a.strike:.0f}{opt_type}  score={a.score}")
                print(f"  Stock ${a.stock_price:.2f}  ·  Bid ${a.bid:.2f}  Ask ${a.ask:.2f}  "
                      f"Mid ${a.mid:.2f}  OI {a.oi:,}  spread {a.spread_pct*100:.1f}%")
                print("  Why:")
                for r in a.reasons:
                    print(f"    • {r}")
                if a.days_to_earnings is not None:
                    tag = "BEFORE expiry (catalyst)" if a.earnings_before_expiry else "after expiry"
                    print(f"  📅 Earnings: in {a.days_to_earnings}d — {tag}")
                if a.news_sentiment is not None:
                    sent = a.news_sentiment
                    tag = "🟢 bull" if sent > 0.1 else "🔴 bear" if sent < -0.1 else "⚪ neutral"
                    hot = "  🔥 hot" if a.news_hot else ""
                    src = f"  [{', '.join(a.news_sources)}]" if a.news_sources else ""
                    print(f"  📰 News: {tag} (score {sent:+.2f}) · {a.news_article_count} articles{hot}{src}")
                    for h in a.headlines[:3]:
                        print(f"      • {h[:100]}")
        return

    # Post one section per expiry so Discord renders a clean separator
    for exp in expiries_sorted:
        group = by_expiry[exp]
        header = (f"**📅 Weekly options — expiring {exp} "
                  f"({group[0].dte} DTE) · {len(group)} setups**")
        _post(webhook_url, {"content": header})
        _post_embeds(webhook_url, [_weekly_embed(a) for a in group])


# ── meme / squeeze alerts (#meme) ──────────────────────────────────────────

MEME_TIER_COLOR = {"SQUEEZE": 0xff3b30, "UNUSUAL": 0xff9500, "WATCH": 0x5ac8fa}
MEME_TIER_EMOJI = {"SQUEEZE": "🚀", "UNUSUAL": "🔥", "WATCH": "👀"}


def _fmt_vol(n: int) -> str:
    if n >= 1_000_000_000: return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:     return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:         return f"{n / 1_000:.1f}K"
    return str(n)


def _meme_embed(a: MemeAlert) -> Dict:
    emoji = MEME_TIER_EMOJI.get(a.tier, "")

    desc_parts = [f"Stock **${a.price:.2f}**"]
    if a.ret_1d is not None:
        desc_parts.append(f"1d **{a.ret_1d*100:+.1f}%**")
    if a.ret_5d is not None:
        desc_parts.append(f"5d **{a.ret_5d*100:+.1f}%**")
    description = "  ·  ".join(desc_parts)

    vol_val = (
        f"Today **{_fmt_vol(a.today_volume)}**  ·  "
        f"20d avg {_fmt_vol(a.avg_vol_20d)}  ·  "
        f"**{a.vol_ratio:.1f}x**"
    )

    short_bits = []
    if a.short_pct_float is not None:
        short_bits.append(f"**{a.short_pct_float*100:.0f}%** of float short")
    if a.days_to_cover is not None:
        short_bits.append(f"**{a.days_to_cover:.1f}** days to cover")
    if a.float_shares:
        short_bits.append(f"float {_fmt_vol(a.float_shares)}")
    short_val = "  ·  ".join(short_bits) if short_bits else "no SI data"

    fields = [
        {"name": "Tier",  "value": f"**{emoji} {a.tier}**", "inline": True},
        {"name": "Score", "value": f"**{a.score}**",        "inline": True},
        {"name": "🔊 Volume",         "value": vol_val,   "inline": False},
        {"name": "🩳 Short interest", "value": short_val, "inline": False},
        {"name": "Signals",
         "value": "\n".join(f"• {r}" for r in a.reasons)[:1024] or "—",
         "inline": False},
    ]

    if a.wsb_rank is not None:
        wsb_bits = [f"WSB rank **#{a.wsb_rank}**"]
        if a.wsb_mentions_24h is not None:
            wsb_bits.append(f"{a.wsb_mentions_24h} mentions/24h")
        if a.wsb_mentions_change is not None:
            wsb_bits.append(f"{a.wsb_mentions_change:+.0f}% vs yesterday")
        fields.append({"name": "🦍 WSB", "value": "  ·  ".join(wsb_bits), "inline": False})

    # Stocktwits
    if a.st_sentiment_score is not None or a.st_message_velocity:
        s = a.st_sentiment_score
        if   s is None:        st_tag = "—"
        elif s >=  0.3:        st_tag = f"🟢 {s*100:+.0f}% bull"
        elif s <= -0.3:        st_tag = f"🔴 {s*100:+.0f}% bear"
        else:                  st_tag = f"⚪ {s*100:+.0f}% mixed"
        st_bits = [st_tag]
        if a.st_bull_count is not None and a.st_bear_count is not None:
            st_bits.append(f"{a.st_bull_count}🟢 / {a.st_bear_count}🔴")
        if a.st_message_velocity:
            st_bits.append(f"chatter **{a.st_message_velocity:.1f}x** pace")
        if a.st_watchlist:
            st_bits.append(f"{_fmt_vol(a.st_watchlist)} watching")
        st_val = "  ·  ".join(st_bits)
        if a.st_top_message:
            st_val += f"\n> {a.st_top_message[:140]}"
        fields.append({"name": "💬 Stocktwits", "value": st_val[:1024], "inline": False})

    # Gamma exposure
    if a.gex_call_put_ratio is not None:
        gex_bits = []
        if a.gex_setup:
            gex_bits.append("⚡ **gamma squeeze setup**")
        if a.gex_call_put_ratio:
            gex_bits.append(f"call/put OI **{a.gex_call_put_ratio:.1f}x**")
        if a.gex_magnet_strike and a.gex_magnet_pct is not None:
            gex_bits.append(f"magnet **${a.gex_magnet_strike:.0f}** ({a.gex_magnet_pct*100:+.1f}%)")
        if a.gex_dollar is not None:
            sign = "+" if a.gex_dollar >= 0 else "−"
            absv = abs(a.gex_dollar)
            if   absv >= 1e9: gex_bits.append(f"GEX {sign}${absv/1e9:.2f}B")
            elif absv >= 1e6: gex_bits.append(f"GEX {sign}${absv/1e6:.0f}M")
        fields.append({"name": "⚡ Gamma exposure", "value": "  ·  ".join(gex_bits), "inline": False})

    return {
        "title":       f"{emoji} {a.ticker} — {a.tier}",
        "description": description,
        "color":       MEME_TIER_COLOR.get(a.tier, 0x95a5a6),
        "fields":      fields,
    }


def send_meme_alerts(webhook_url: str, alerts: List[MemeAlert], dry_run: bool = False) -> None:
    if not alerts:
        print("[discord] no meme alerts to send")
        return

    if dry_run or not webhook_url:
        print(f"\n[DRY meme] {'─'*60}")
        for a in alerts:
            emoji = MEME_TIER_EMOJI.get(a.tier, "")
            print(f"\n  {emoji} {a.ticker}  {a.tier}  score={a.score}  ${a.price:.2f}")
            print(f"  vol {a.vol_ratio:.1f}x ({_fmt_vol(a.today_volume)} vs avg {_fmt_vol(a.avg_vol_20d)})")
            if a.short_pct_float is not None:
                print(f"  SI {a.short_pct_float*100:.0f}% of float  ·  DTC {a.days_to_cover or 0:.1f}d")
            if a.wsb_rank:
                print(f"  WSB #{a.wsb_rank}  {a.wsb_mentions_24h} mentions")
            for r in a.reasons:
                print(f"    • {r}")
        return

    counts: Dict[str, int] = {}
    for a in alerts:
        counts[a.tier] = counts.get(a.tier, 0) + 1
    summary = "  ·  ".join(f"{MEME_TIER_EMOJI.get(t,'')} {n} {t}"
                           for t, n in counts.items())
    _post(webhook_url, {"content": f"**🎰 Meme / squeeze scan**  —  {summary}"})
    _post_embeds(webhook_url, [_meme_embed(a) for a in alerts])
