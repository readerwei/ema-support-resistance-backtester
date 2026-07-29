from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class AnalysisResult:
    bars: pd.DataFrame
    interactions: pd.DataFrame
    summary: dict[str, float | int | None]


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(bars.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if len(bars) < 3:
        raise ValueError("At least 3 bars are required")
    result = bars.copy()
    result.index = pd.to_datetime(result.index, utc=True)
    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError("Bar timestamps must be unique")
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=list(REQUIRED_COLUMNS))
    if len(result) < 3:
        raise ValueError("Not enough complete bars after removing missing values")
    return result


def _atr(bars: pd.DataFrame, period: int) -> pd.Series:
    previous_close = bars["close"].shift(1)
    true_range = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - previous_close).abs(), (bars["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _bar_relation(row: pd.Series) -> Literal["above", "below", "inside", "mixed"]:
    """Classify a candle body relative to the ATR bands."""
    open_price = float(row["open"])
    close = float(row["close"])
    upper = float(row["upper_band"])
    lower = float(row["lower_band"])
    if open_price > upper and close > upper:
        return "above"
    if open_price < lower and close < lower:
        return "below"
    if lower <= open_price <= upper and lower <= close <= upper:
        return "inside"
    return "mixed"


def analyze_ema_interactions(
    bars: pd.DataFrame,
    ema_period: int = 70,
    atr_period: int = 200,
    atr_multiple: float = 0.5,
    mode: Literal["close", "body"] = "close",
) -> AnalysisResult:
    """Classify EMA interactions using only information available at each bar close."""
    if ema_period < 1 or atr_period < 1 or atr_multiple <= 0:
        raise ValueError("Periods must be positive and atr_multiple must be greater than zero")
    if mode not in {"close", "body"}:
        raise ValueError("mode must be 'close' or 'body'")
    data = _validate_bars(bars)
    data = data.copy()
    data["ema"] = data["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    data["atr"] = _atr(data, atr_period)
    data["upper_band"] = data["ema"] + atr_multiple * data["atr"]
    data["lower_band"] = data["ema"] - atr_multiple * data["atr"]
    data["regime"] = np.where(data["close"] >= data["ema"], "support", "resistance")

    rows: list[dict] = []
    active: dict | None = None
    previous: pd.Series | None = None
    for timestamp, row in data.iterrows():
        if pd.isna(row["upper_band"]) or pd.isna(row["lower_band"]):
            previous = row
            continue
        close = float(row["close"])
        upper = float(row["upper_band"])
        lower = float(row["lower_band"])
        if active is None:
            if previous is not None and not pd.isna(previous["upper_band"]):
                if mode == "body":
                    previous_relation = _bar_relation(previous)
                    relation = _bar_relation(row)
                    if previous_relation == "above" and relation == "inside":
                        active = {"side": "support", "start": timestamp, "entry_price": close}
                    elif previous_relation == "below" and relation == "inside":
                        active = {"side": "resistance", "start": timestamp, "entry_price": close}
                    elif previous_relation == "above" and relation == "below":
                        rows.append({"timestamp": timestamp, "side": "support", "outcome": "penetration", "entry_price": float(previous["close"]), "exit_price": close})
                    elif previous_relation == "below" and relation == "above":
                        rows.append({"timestamp": timestamp, "side": "resistance", "outcome": "penetration", "entry_price": float(previous["close"]), "exit_price": close})
                else:
                    prev_close = float(previous["close"])
                    prev_upper = float(previous["upper_band"])
                    prev_lower = float(previous["lower_band"])
                    inside_band = lower <= close <= upper
                    if prev_close > prev_upper and inside_band:
                        active = {"side": "support", "start": timestamp, "entry_price": close}
                    elif prev_close < prev_lower and inside_band:
                        active = {"side": "resistance", "start": timestamp, "entry_price": close}
                    elif prev_close > prev_upper and close < lower:
                        rows.append({"timestamp": timestamp, "side": "support", "outcome": "penetration", "entry_price": prev_close, "exit_price": close})
                    elif prev_close < prev_lower and close > upper:
                        rows.append({"timestamp": timestamp, "side": "resistance", "outcome": "penetration", "entry_price": prev_close, "exit_price": close})
        else:
            side = active["side"]
            if mode == "body":
                relation = _bar_relation(row)
                if side == "support":
                    if relation == "above":
                        rows.append({"timestamp": timestamp, **active, "outcome": "bounce", "exit_price": close})
                        active = None
                    elif relation == "below":
                        rows.append({"timestamp": timestamp, **active, "outcome": "penetration", "exit_price": close})
                        active = None
                else:
                    if relation == "below":
                        rows.append({"timestamp": timestamp, **active, "outcome": "bounce", "exit_price": close})
                        active = None
                    elif relation == "above":
                        rows.append({"timestamp": timestamp, **active, "outcome": "penetration", "exit_price": close})
                        active = None
            elif side == "support":
                if close > upper:
                    rows.append({"timestamp": timestamp, **active, "outcome": "bounce", "exit_price": close})
                    active = None
                elif close < lower:
                    rows.append({"timestamp": timestamp, **active, "outcome": "penetration", "exit_price": close})
                    active = None
            else:
                if close < lower:
                    rows.append({"timestamp": timestamp, **active, "outcome": "bounce", "exit_price": close})
                    active = None
                elif close > upper:
                    rows.append({"timestamp": timestamp, **active, "outcome": "penetration", "exit_price": close})
                    active = None
        previous = row

    interactions = pd.DataFrame(rows)
    if interactions.empty:
        interactions = pd.DataFrame(columns=["timestamp", "side", "start", "entry_price", "outcome", "exit_price"])
    support = interactions[interactions["side"] == "support"] if not interactions.empty else interactions
    resistance = interactions[interactions["side"] == "resistance"] if not interactions.empty else interactions

    def bounce_pct(frame: pd.DataFrame) -> float | None:
        return None if frame.empty else round(float((frame["outcome"] == "bounce").mean() * 100), 4)

    summary = {
        "bars": len(data),
        "interactions": len(interactions),
        "support_interactions": len(support),
        "support_bounce_pct": bounce_pct(support),
        "resistance_interactions": len(resistance),
        "resistance_bounce_pct": bounce_pct(resistance),
        "combined_bounce_pct": bounce_pct(interactions),
        "ema_period": ema_period,
        "atr_period": atr_period,
        "atr_multiple": atr_multiple,
        "mode": mode,
    }
    return AnalysisResult(data, interactions, summary)


def scan_ema_periods(
    bars: pd.DataFrame,
    ema_periods: Iterable[int],
    atr_period: int = 200,
    atr_multiple: float = 0.5,
    mode: Literal["close", "body"] = "close",
) -> pd.DataFrame:
    """Evaluate support, resistance, and combined bounce rates for each EMA period."""
    periods = list(ema_periods)
    if not periods or any(period < 1 for period in periods):
        raise ValueError("ema_periods must contain at least one positive period")
    rows = []
    for period in periods:
        summary = analyze_ema_interactions(bars, period, atr_period, atr_multiple, mode=mode).summary
        rows.append({
            "ema_period": period,
            "support_bounce_pct": summary["support_bounce_pct"],
            "resistance_bounce_pct": summary["resistance_bounce_pct"],
            "combined_bounce_pct": summary["combined_bounce_pct"],
            "support_interactions": summary["support_interactions"],
            "resistance_interactions": summary["resistance_interactions"],
            "interactions": summary["interactions"],
        })
    return pd.DataFrame(rows)


def _best_bounce_pct(bars: pd.DataFrame, ema_periods: Iterable[int], atr_period: int, atr_multiple: float, mode: Literal["close", "body"] = "close") -> float:
    scores = [analyze_ema_interactions(bars, p, atr_period, atr_multiple, mode=mode).summary["combined_bounce_pct"] for p in ema_periods]
    return max((float(score) for score in scores if score is not None), default=0.0)


def monte_carlo_p_value(
    bars: pd.DataFrame,
    actual_bounce_pct: float,
    ema_periods: Iterable[int],
    atr_period: int = 200,
    atr_multiple: float = 0.5,
    simulations: int = 1000,
    seed: int = 42,
    mode: Literal["close", "body"] = "close",
) -> float:
    """Shuffle log returns and estimate the probability of matching the observed optimum."""
    if simulations < 1:
        raise ValueError("simulations must be positive")
    data = _validate_bars(bars)
    rng = np.random.default_rng(seed)
    closes = data["close"].to_numpy(float)
    log_returns = np.diff(np.log(closes))
    at_least_as_high = 0
    for _ in range(simulations):
        shuffled = rng.permutation(log_returns)
        synthetic_close = np.r_[closes[0], closes[0] * np.exp(np.cumsum(shuffled))]
        synthetic = data.copy()
        synthetic["close"] = synthetic_close
        synthetic["open"] = synthetic["close"].shift(1).fillna(synthetic["close"])
        synthetic["high"] = synthetic[["open", "close"]].max(axis=1)
        synthetic["low"] = synthetic[["open", "close"]].min(axis=1)
        if _best_bounce_pct(synthetic, ema_periods, atr_period, atr_multiple, mode) >= actual_bounce_pct:
            at_least_as_high += 1
    return round((at_least_as_high + 1) / (simulations + 1), 6)
