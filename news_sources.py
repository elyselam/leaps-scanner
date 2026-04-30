"""Multi-source news fetcher for the weekly options scanner.

Pulls from (in priority order):
  1. Polygon.io  /v2/reference/news     — fast, ticker-tagged, has insights
  2. Finnhub     /company-news          — broad coverage, hourly granularity
  3. Finnhub     /news-sentiment        — pre-computed sentiment + buzz
  4. yfinance    Ticker.news            — fallback only

Dedupes across sources by normalized title.
Default window: last 48 hours (configurable). Tighter than yfinance's 7d
because the user explicitly wants current data for short-DTE weeklies.

Env vars (read at call time):
  POLYGON_API_KEY   (already in your .env)
  FINNHUB_API_KEY   (add to .env — get free key at finnhub.io)
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

POLYGON_NEWS_URL  = "https://api.polygon.io/v2/reference/news"
FINNHUB_NEWS_URL  = "https://finnhub.io/api/v1/company-news"
FINNHUB_SENT_URL  = "https://finnhub.io/api/v1/news-sentiment"

# How long to wait per HTTP call before giving up (network hiccups shouldn't
# stall the whole scanner)
HTTP_TIMEOUT = 6

# ── rate limiting ───────────────────────────────────────────────────────────
# Polygon free tier = 5 req/min → spacing ~12s between calls keeps us safe.
# Finnhub free tier = 60 req/min → 1s spacing is plenty.
_POLYGON_MIN_INTERVAL = 12.5
_FINNHUB_MIN_INTERVAL = 1.1
_last_polygon_call_ts = 0.0
_last_finnhub_call_ts = 0.0


def _pace(last_ts: float, min_interval: float) -> float:
    """Sleep as needed so it's been >= min_interval since last_ts. Returns new ts."""
    now = time.time()
    wait = min_interval - (now - last_ts)
    if wait > 0:
        time.sleep(wait)
    return time.time()


# ── normalization & dedupe ──────────────────────────────────────────────────

def _normalize_title(s: str) -> str:
    """Lowercase, strip punctuation/whitespace, take first 60 chars.
    Used as a dedup key — different publishers reword the same story."""
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60]


def _dedupe_articles(articles: List[Dict]) -> List[Dict]:
    """Remove articles with same normalized title. Keep the newest one."""
    seen: Dict[str, Dict] = {}
    for art in articles:
        key = _normalize_title(art.get("title", ""))
        if not key:
            continue
        existing = seen.get(key)
        if existing is None or art.get("ts", 0) > existing.get("ts", 0):
            seen[key] = art
    # Newest first
    return sorted(seen.values(), key=lambda a: -a.get("ts", 0))


# ── Polygon.io ───────────────────────────────────────────────────────────────

def fetch_polygon_news(ticker: str, hours_back: int = 48,
                       limit: int = 20) -> List[Dict]:
    """Polygon's news endpoint. Returns articles with publisher + insights.
    Some Polygon plans include sentiment in `insights[].sentiment`.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    params = {
        "ticker":              ticker,
        "published_utc.gte":   cutoff,
        "order":               "desc",
        "limit":               limit,
        "apiKey":              api_key,
    }
    global _last_polygon_call_ts
    _last_polygon_call_ts = _pace(_last_polygon_call_ts, _POLYGON_MIN_INTERVAL)
    try:
        r = requests.get(POLYGON_NEWS_URL, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            # Try once more after a full minute
            time.sleep(60)
            r = requests.get(POLYGON_NEWS_URL, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                print(f"[news] Polygon rate-limited for {ticker} (after retry)")
                return []
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[news] Polygon error for {ticker}: {e}")
        return []

    out: List[Dict] = []
    for item in data.get("results", [])[:limit]:
        title = item.get("title") or ""
        if not title:
            continue
        # Publish time → unix ts
        pub = item.get("published_utc")
        try:
            ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = time.time()

        # Polygon "insights" may include per-ticker sentiment
        sent = None
        for ins in (item.get("insights") or []):
            if (ins.get("ticker") or "").upper() == ticker.upper():
                s = (ins.get("sentiment") or "").lower()
                if   s == "positive": sent =  0.5
                elif s == "negative": sent = -0.5
                elif s == "neutral":  sent =  0.0
                break

        out.append({
            "title":     title,
            "ts":        ts,
            "publisher": (item.get("publisher") or {}).get("name", "Polygon"),
            "url":       item.get("article_url"),
            "sentiment": sent,                # may be None
            "source":    "polygon",
        })
    return out


# ── Finnhub ──────────────────────────────────────────────────────────────────

def fetch_finnhub_news(ticker: str, hours_back: int = 48,
                       limit: int = 25) -> List[Dict]:
    """Finnhub company news. Free tier = 60 req/min.
    Date range is full days, but we filter by hour client-side.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []

    today    = datetime.now(timezone.utc).date()
    from_day = today - timedelta(days=max(1, hours_back // 24 + 1))
    params = {
        "symbol": ticker,
        "from":   from_day.isoformat(),
        "to":     today.isoformat(),
        "token":  api_key,
    }
    global _last_finnhub_call_ts
    _last_finnhub_call_ts = _pace(_last_finnhub_call_ts, _FINNHUB_MIN_INTERVAL)
    try:
        r = requests.get(FINNHUB_NEWS_URL, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            print(f"[news] Finnhub rate-limited for {ticker}")
            return []
        r.raise_for_status()
        items = r.json() or []
    except Exception as e:
        print(f"[news] Finnhub news error for {ticker}: {e}")
        return []

    cutoff_ts = time.time() - hours_back * 3600
    out: List[Dict] = []
    for item in items[:limit * 2]:
        title = item.get("headline") or ""
        ts    = float(item.get("datetime") or 0)
        if not title or ts < cutoff_ts:
            continue
        out.append({
            "title":     title,
            "ts":        ts,
            "publisher": item.get("source", "Finnhub"),
            "url":       item.get("url"),
            "sentiment": None,             # comes from /news-sentiment endpoint
            "source":    "finnhub",
        })
        if len(out) >= limit:
            break
    return out


def fetch_finnhub_sentiment(ticker: str) -> Optional[Dict]:
    """Finnhub's pre-computed news-sentiment score for the company.
    Returns dict with:
      score                — overall company score in [-1, 1] (we normalize)
      bullish_pct          — % articles bullish
      bearish_pct          — % articles bearish
      articles_last_week   — buzz volume
      sector_avg           — sector benchmark for comparison
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return None

    global _last_finnhub_call_ts
    _last_finnhub_call_ts = _pace(_last_finnhub_call_ts, _FINNHUB_MIN_INTERVAL)
    try:
        r = requests.get(
            FINNHUB_SENT_URL,
            params={"symbol": ticker, "token": api_key},
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code == 429:
            return None
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        print(f"[news] Finnhub sentiment error for {ticker}: {e}")
        return None

    sent = d.get("sentiment") or {}
    buzz = d.get("buzz") or {}
    bullish = float(sent.get("bullishPercent") or 0)
    bearish = float(sent.get("bearishPercent") or 0)
    # Normalize to a -1..+1 score
    score = bullish - bearish     # both are 0..1 → diff is -1..+1

    # Treat empty buzz as missing data
    articles = int(buzz.get("articlesInLastWeek") or 0)
    if articles == 0 and bullish == 0 and bearish == 0:
        return None

    return {
        "score":              max(-1.0, min(1.0, score)),
        "bullish_pct":        bullish,
        "bearish_pct":        bearish,
        "articles_last_week": articles,
        "sector_avg":         float(d.get("sectorAverageNewsScore") or 0),
        "company_score_raw":  float(d.get("companyNewsScore") or 0),
    }


# ── yfinance fallback ────────────────────────────────────────────────────────

def fetch_yfinance_news(ticker: str, hours_back: int = 72,
                        limit: int = 15) -> List[Dict]:
    """Fallback. yfinance lags real wires by minutes-to-hours.
    Window defaults wider here (72h) since yfinance tends to be sparse."""
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    cutoff_ts = time.time() - hours_back * 3600
    out: List[Dict] = []
    for item in items[:limit * 2]:
        title = (item.get("title")
                 or (item.get("content") or {}).get("title")
                 or "")
        ts = item.get("providerPublishTime")
        if ts is None:
            content = item.get("content") or {}
            pub_date = content.get("pubDate")
            if pub_date:
                try:
                    ts = datetime.fromisoformat(
                        pub_date.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = None
        if not title or not isinstance(ts, (int, float)):
            continue
        if ts < cutoff_ts:
            continue
        out.append({
            "title":     title,
            "ts":        float(ts),
            "publisher": (item.get("publisher")
                          or (item.get("content") or {}).get("provider", {}).get("displayName")
                          or "Yahoo"),
            "url":       item.get("link"),
            "sentiment": None,
            "source":    "yfinance",
        })
        if len(out) >= limit:
            break
    return out


# ── unified entry point ──────────────────────────────────────────────────────

def fetch_all_news(ticker: str, hours_back: int = 48) -> Dict:
    """Pull from all configured sources, dedupe, return summary dict.

    Returns:
      {
        articles:           [ {title, ts, publisher, url, sentiment, source}, ... ]
        sources_used:       ["polygon", "finnhub", ...]
        finnhub_sentiment:  {score, bullish_pct, ...} | None
        top_headline:       str | None
        article_count:      int
        is_hot:             bool   # ≥4 articles in window
      }
    """
    all_articles: List[Dict] = []
    sources_used: List[str]  = []

    poly = fetch_polygon_news(ticker, hours_back=hours_back)
    if poly:
        all_articles.extend(poly)
        sources_used.append("polygon")

    fin = fetch_finnhub_news(ticker, hours_back=hours_back)
    if fin:
        all_articles.extend(fin)
        sources_used.append("finnhub")

    # Only fall back to yfinance if both primary sources came up empty.
    if not all_articles:
        yfn = fetch_yfinance_news(ticker, hours_back=max(hours_back, 72))
        if yfn:
            all_articles.extend(yfn)
            sources_used.append("yfinance")

    deduped = _dedupe_articles(all_articles)

    # Finnhub's pre-computed company-level sentiment (independent of headlines)
    fh_sent = fetch_finnhub_sentiment(ticker)
    if fh_sent and "finnhub" not in sources_used:
        sources_used.append("finnhub-sentiment")

    return {
        "articles":          deduped,
        "sources_used":      sources_used,
        "finnhub_sentiment": fh_sent,
        "top_headline":      deduped[0]["title"] if deduped else None,
        "article_count":     len(deduped),
        "is_hot":            len(deduped) >= 4,
    }
