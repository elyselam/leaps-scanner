# LEAPS Recommender

Scans a watchlist intraday for long-dated call setups (LEAPS) and pings a Discord channel on A/B-tier candidates. Primary data source is `yfinance` (free, no key); Polygon free tier is wired in as an optional supplement for EOD equity bars.

## What it checks

Each ticker is scored on ~15 weighted criteria grouped into:

- **Trend / structure** — above 200d SMA, weekly HH/HL, RS vs SPY, healthy distance from 52w high.
- **Momentum** — RSI oversold or bullish divergence, MACD curling / histogram rising.
- **Volume** — volume drying up on the pullback.
- **Volatility** — IV Rank (true IVR once 60+ days of ATM-IV history are stored; HV-rank proxy before that), 25-delta put/call skew, term-structure sanity, current IV/HV ratio.
- **Options liquidity** — picks the LEAPS call in your delta band (default 0.55–0.85) with highest OI at ~1yr expiry; requires OI ≥ 500 and bid-ask ≤ 8% by default.
- **Fundamentals** — positive free cash flow, debt/equity, revenue growth, short interest sanity.
- **Events** — next earnings ≥ 21 days out (avoid IV crush).
- **Macro** — SPY above 200d, HYG stable, VIX elevated-but-calming, sector ETF RS.

Tiers: **A** (score ≥ 70, all 4 primary confirmations), **B** (score ≥ 50, 3+ primary), else `reject`.

## Setup

```bash
cd leaps_scanner
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, paste DISCORD_WEBHOOK_URL (and POLYGON_API_KEY if you want)
```

### Get a Discord webhook

In Discord: channel settings → **Integrations** → **Webhooks** → **New Webhook** → copy URL → paste into `.env`.

## Run

```bash
# Single scan, no Discord posting:
python main.py --dry-run

# Single scan, post A/B-tier alerts + digest to Discord:
python main.py

# Continuous loop (reads scan_interval_minutes in config.json, default 45m):
python main.py --loop

# Override watchlist ad-hoc:
python main.py --tickers NVDA MU PLTR --dry-run

# Force scan outside market hours:
python main.py --loop --force
```

## Config

Edit `config.json`:

- `watchlist` — your tickers.
- `suggested_additions` + `use_suggested_additions: true` — flip to fold in AVGO/TSM/AMAT/ANET/MSFT.
- `sector_map` — maps each ticker to its sector ETF for the sector-RS check.
- `thresholds` — tune numeric cut-offs (IVR range, RSI level, delta band, OI floor, etc).
- `weights` — per-criterion points that sum into the total score.
- `tiers` — A/B minimum scores.
- `scan_interval_minutes` — loop interval.
- `market_hours_only` — skip scans outside 9:30–4pm ET.
- `dedupe_hours` — don't re-alert the same (ticker, tier, strike, expiry) within this many hours.

## Data notes & known limits

- **Polygon free tier has no options data.** All options chain data (IV, OI, bid/ask) comes from `yfinance`, which is free but occasionally flaky. If a chain fetch fails for a ticker on one scan, it'll just get skipped that round and try again next interval.
- **True IV Rank requires history.** The scanner stores daily ATM IV in `leaps_state.db` (SQLite). Until ~60 days of history accumulate per ticker, IVR falls back to **HV rank** (percentile of current 30d realized vol over the last 252d) as a proxy. This is noted in the Discord embed so you know what you're reading.
- **Rate limits.** yfinance has no hard rate limit but gets occasional timeouts; the scanner handles them gracefully. Polygon free tier is 5 calls/min — only used if `POLYGON_API_KEY` is set, and currently just for EOD bar cross-check.
- **No IV computation of exotic tenors or mid-curve skew.** If you want a more rigorous vol surface, wire in an options data vendor (ORATS, Cboe LiveVol) — the `data.py` layer is the only file that would need to change.
- **Market-hours gate is US only and doesn't check holidays.** Pass `--force` to scan outside those hours.

## Interpreting a Discord alert

```
NVDA — Tier A LEAPS setup
Score: 78/104 — Tier A
Price: $892.41
IV context: 34 (HV-rank proxy)
Suggested LEAPS: $900C 2027-01-15 (360d)
                 Δ 0.52 · OI 12483 · spread 1.8%
Next earnings: in 45d
Why:
+10 above 200d SMA
+8 RS vs SPY +12.3%
+6 18% off 52w high (healthy)
+10 RSI 32 oversold/divergent
+8 MACD curling up
...
```

The "suggested LEAPS" is **the contract in your delta band with the highest OI at the ~1yr expiry** — it's a starting point, not a buy recommendation. Always check spreads live before entering.

## Files

```
leaps_scanner/
├── main.py              # entry point + scan loop
├── config.json          # watchlist, thresholds, weights, tiers
├── data.py              # yfinance + Polygon data fetching
├── signals.py           # trend/momentum/volatility/options/etc computations
├── scoring.py           # weighted score + A/B tier logic
├── discord_sender.py    # webhook formatter
├── state.py             # SQLite IV history + alert dedupe
├── requirements.txt
├── .env.example
└── leaps_state.db       # (created on first run)
```
