from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests


YAHOO_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"}


def _period_seconds(value: str) -> int:
    number = int(value[:-1])
    unit = value[-1].lower()
    return number * {"d": 86400, "h": 3600, "m": 60}.get(unit, 86400)


def yahoo_interval_for(timeframe: str) -> str:
    tf = timeframe.lower()
    if tf in YAHOO_INTERVALS:
        return tf
    if tf.endswith("h") and int(tf[:-1]) % 1 == 0:
        return "60m"
    if tf.endswith("min"):
        minutes = int(tf[:-3])
        if f"{minutes}m" in YAHOO_INTERVALS:
            return f"{minutes}m"
    raise ValueError("Unsupported timeframe. Use 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1wk, or 1mo")


def _resample_rule(timeframe: str) -> str | None:
    tf = timeframe.lower()
    if tf.endswith("h") and tf not in {"1h", "60m"}:
        return f"{int(tf[:-1])}h"
    if tf.endswith("min"):
        return f"{int(tf[:-3])}min"
    return None


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def fetch_yahoo_bars(symbol: str, start: str, end: str, timeframe: str = "1d", session: str = "regular", timeout: int = 30) -> pd.DataFrame:
    """Fetch OHLCV bars from Yahoo's chart endpoint; intended for research/smoke tests."""
    if session not in {"regular", "extended"}:
        raise ValueError("session must be 'regular' or 'extended'")
    interval = yahoo_interval_for(timeframe)
    period1 = int(_parse_utc(start).timestamp())
    period2 = int(_parse_utc(end).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
    response = requests.get(url, params={"period1": period1, "period2": period2, "interval": interval, "events": "div,splits", "includePrePost": "true" if session == "extended" else "false"}, headers={"User-Agent": "ema-sr-backtester/0.1"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise ValueError(f"Yahoo returned no data for {symbol}: {error}")
    result = result[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    frame = pd.DataFrame({key: quote.get(key, []) for key in ["open", "high", "low", "close", "volume"]}, index=pd.to_datetime(timestamps, unit="s", utc=True))
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    if frame.empty:
        raise ValueError(f"Yahoo returned no complete OHLCV bars for {symbol}")
    rule = _resample_rule(timeframe)
    if rule:
        frame = frame.resample(rule, label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    frame.attrs["symbol"] = symbol.upper()
    frame.attrs["timezone"] = result.get("meta", {}).get("exchangeTimezoneName", "UTC")
    frame.attrs["source"] = "Yahoo Finance chart API"
    frame.attrs["session"] = session
    return frame
