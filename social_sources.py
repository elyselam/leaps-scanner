"""Stocktwits — second social signal alongside ApeWisdom (WSB).

Stocktwits skews more retail/speculative than Reddit, and users self-tag
their messages Bullish/Bearish — so we get a clean sentiment without NLP
guessing. Free, no auth required, ~200 req/hr soft cap.

Endpoint:
  https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json
Returns:
  symbol.watchlist_count           — total followers
  messages[].entities.sentiment.basic  — "Bullish" | "Bearish" | None
  messages[].created_at            — ISO timestamp
  messages[].user.followers        — message author followers
  messages[].likes.total           — engagement
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

ST_URL          = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"
ST_TIMEOUT      = 6
ST_MIN_INTERVAL = 0.4   # ~150 req/min headroom under their 200/hr cap

_last_call_ts = 0.0


def _pace() -> None:
    global _last_call_ts
    wait = ST_MIN_INTERVAL - (time.time() - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()


def fetch_stocktwits(ticker: str, lookback_hours: int = 24) -> Optional[Dict]:
    """Fetch recent Stocktwits stream for a ticker.

    Returns dict with:
      bull_count          — # messages tagged Bullish in window
      bear_count          — # messages tagged Bearish
      tagged_total        — bull + bear (rest are untagged)
      sentiment_score     — (bull - bear) / (bull + bear),  -1.0..+1.0
      message_count       — total messages seen in window
      message_velocity    — msgs/hr in last 1h vs prior 23h average (ratio)
      watchlist_count     — total followers of the symbol
      top_message         — body of the most-liked message in window
    None if no data or fetch fails.
    """
    _pace()
    try:
        r = requests.get(
            ST_URL.format(ticker.upper()),
            timeout=ST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 leaps-scanner"},
        )
        if r.status_code == 429:
            print(f"[social] Stocktwits rate-limited for {ticker}")
            return None
        if r.status_code == 404:
            return None      # ticker not on Stocktwits
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[social] Stocktwits error for {ticker}: {e}")
        return None

    msgs = data.get("messages") or []
    if not msgs:
        return None

    cutoff_ts = time.time() - lookback_hours * 3600
    one_hr_ts = time.time() - 3600

    bull = bear = 0
    in_window: List[Dict] = []
    last_hour = 0
    prior     = 0
    top_msg   = None
    top_likes = -1

    for m in msgs:
        # Parse timestamp
        try:
            ts = datetime.fromisoformat(
                (m.get("created_at") or "").replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            continue
        if ts < cutoff_ts:
            continue

        in_window.append(m)
        if ts >= one_hr_ts: last_hour += 1
        else:               prior     += 1

        sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        if   sent == "Bullish": bull += 1
        elif sent == "Bearish": bear += 1

        likes = (m.get("likes") or {}).get("total") or 0
        if likes > top_likes:
            top_likes = likes
            top_msg   = (m.get("body") or "").strip()

    tagged = bull + bear
    score  = ((bull - bear) / tagged) if tagged > 0 else None

    # Velocity: last 1h pace vs prior (lookback-1)h pace
    prior_hours = max(1, lookback_hours - 1)
    prior_rate  = prior / prior_hours
    velocity    = (last_hour / prior_rate) if prior_rate > 0 else (
        float(last_hour) if last_hour else 1.0
    )

    return {
        "bull_count":       bull,
        "bear_count":       bear,
        "tagged_total":     tagged,
        "sentiment_score":  score,
        "message_count":    len(in_window),
        "message_velocity": velocity,        # >2.0 = unusually busy in last hour
        "watchlist_count":  ((data.get("symbol") or {}).get("watchlist_count")),
        "top_message":      (top_msg or "")[:140],
    }
