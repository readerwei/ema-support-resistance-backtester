from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


JudgmentMode = Literal["close", "body", "full"]
TrendMode = Literal["slope", "improved"]


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


def _full_stick_relation(row: pd.Series) -> Literal["above", "below", "inside", "mixed"]:
    """Classify the complete candle, including wicks, relative to ATR bands."""
    high = float(row["high"])
    low = float(row["low"])
    upper = float(row["upper_band"])
    lower = float(row["lower_band"])
    if low > upper:
        return "above"
    if high < lower:
        return "below"
    if lower <= low and high <= upper:
        return "inside"
    return "mixed"


def _outcome_relation(row: pd.Series, mode: JudgmentMode) -> Literal["above", "below", "inside", "mixed"]:
    if mode == "body":
        return _bar_relation(row)
    return _full_stick_relation(row)


def _entry_trend(
    row: pd.Series,
    previous: pd.Series | None,
    mode: TrendMode = "slope",
    slope_threshold: float = 0.1,
) -> str:
    """Classify trend at entry using a baseline or multi-factor regime rule."""
    if mode == "slope":
        if previous is None or pd.isna(previous.get("ema")) or pd.isna(row.get("ema")):
            return "flat"
        if np.isclose(float(row["ema"]), float(previous["ema"])):
            return "flat"
        return "uptrend" if float(row["ema"]) > float(previous["ema"]) else "downtrend"
    if mode != "improved":
        raise ValueError("trend mode must be 'slope' or 'improved'")
    required = ["trend_fast_ema", "trend_slow_ema", "trend_slope_atr"]
    if any(pd.isna(row.get(column)) for column in required):
        return "range/mixed"
    fast = float(row["trend_fast_ema"])
    slow = float(row["trend_slow_ema"])
    close = float(row["close"])
    normalized_slope = float(row["trend_slope_atr"])
    if fast > slow and normalized_slope > slope_threshold and close > slow:
        return "uptrend"
    if fast < slow and normalized_slope < -slope_threshold and close < slow:
        return "downtrend"
    return "range/mixed"


def analyze_ema_interactions(
    bars: pd.DataFrame,
    ema_period: int = 70,
    atr_period: int = 200,
    atr_multiple: float = 0.5,
    mode: JudgmentMode = "close",
    trend_mode: TrendMode = "improved",
    trend_fast_ema: int = 20,
    trend_slow_ema: int = 50,
    trend_slope_lookback: int = 5,
    trend_slope_threshold: float = 0.1,
) -> AnalysisResult:
    """Classify EMA interactions using only information available at each bar close."""
    if ema_period < 1 or atr_period < 1 or atr_multiple <= 0:
        raise ValueError("Periods must be positive and atr_multiple must be greater than zero")
    if mode not in {"close", "body", "full"}:
        raise ValueError("mode must be 'close', 'body', or 'full'")
    if trend_mode not in {"slope", "improved"}:
        raise ValueError("trend_mode must be 'slope' or 'improved'")
    if trend_fast_ema < 1 or trend_slow_ema < 1 or trend_fast_ema >= trend_slow_ema:
        raise ValueError("trend_fast_ema and trend_slow_ema must be positive, with fast < slow")
    if trend_slope_lookback < 1 or trend_slope_threshold < 0:
        raise ValueError("trend_slope_lookback must be positive and trend_slope_threshold non-negative")
    data = _validate_bars(bars)
    data = data.copy()
    data["ema"] = data["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    data["atr"] = _atr(data, atr_period)
    data["upper_band"] = data["ema"] + atr_multiple * data["atr"]
    data["lower_band"] = data["ema"] - atr_multiple * data["atr"]
    data["trend_fast_ema"] = data["close"].ewm(span=trend_fast_ema, adjust=False, min_periods=trend_fast_ema).mean()
    data["trend_slow_ema"] = data["close"].ewm(span=trend_slow_ema, adjust=False, min_periods=trend_slow_ema).mean()
    data["trend_slope_atr"] = (data["trend_slow_ema"] - data["trend_slow_ema"].shift(trend_slope_lookback)) / data["atr"]

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
                prev_close = float(previous["close"])
                prev_upper = float(previous["upper_band"])
                prev_lower = float(previous["lower_band"])
                inside_band = lower <= close <= upper
                if prev_close > prev_upper and inside_band:
                    active = {"side": "support", "start": timestamp, "entry_price": close, "trend": _entry_trend(row, previous, trend_mode, trend_slope_threshold)}
                elif prev_close < prev_lower and inside_band:
                    active = {"side": "resistance", "start": timestamp, "entry_price": close, "trend": _entry_trend(row, previous, trend_mode, trend_slope_threshold)}
                elif prev_close > prev_upper and close < lower:
                    if mode in {"body", "full"}:
                        active = {"side": "support", "start": timestamp, "entry_price": prev_close, "trend": _entry_trend(row, previous, trend_mode, trend_slope_threshold)}
                        if _outcome_relation(row, mode) == "below":
                            rows.append({"timestamp": timestamp, **active, "outcome": "penetration", "exit_price": close})
                            active = None
                    else:
                        rows.append({"timestamp": timestamp, "side": "support", "outcome": "penetration", "entry_price": prev_close, "exit_price": close, "trend": _entry_trend(row, previous, trend_mode, trend_slope_threshold)})
                elif prev_close < prev_lower and close > upper:
                    if mode in {"body", "full"}:
                        active = {"side": "resistance", "start": timestamp, "entry_price": prev_close, "trend": _entry_trend(row, previous, trend_mode, trend_slope_threshold)}
                        if _outcome_relation(row, mode) == "above":
                            rows.append({"timestamp": timestamp, **active, "outcome": "penetration", "exit_price": close})
                            active = None
                    else:
                        rows.append({"timestamp": timestamp, "side": "resistance", "outcome": "penetration", "entry_price": prev_close, "exit_price": close, "trend": _entry_trend(row, previous, trend_mode, trend_slope_threshold)})
        else:
            side = active["side"]
            if mode in {"body", "full"}:
                relation = _outcome_relation(row, mode)
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
        interactions = pd.DataFrame(columns=["timestamp", "side", "start", "entry_price", "trend", "outcome", "exit_price"])
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
        "uptrend_interactions": int((interactions["trend"] == "uptrend").sum()) if not interactions.empty else 0,
        "downtrend_interactions": int((interactions["trend"] == "downtrend").sum()) if not interactions.empty else 0,
        "flat_trend_interactions": int((interactions["trend"] == "flat").sum()) if not interactions.empty else 0,
        "range_mixed_interactions": int((interactions["trend"] == "range/mixed").sum()) if not interactions.empty else 0,
        "trend_mode": trend_mode,
        "trend_fast_ema": trend_fast_ema,
        "trend_slow_ema": trend_slow_ema,
        "trend_slope_lookback": trend_slope_lookback,
        "trend_slope_threshold": trend_slope_threshold,
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
    mode: JudgmentMode = "close",
    trend_mode: TrendMode = "improved",
    trend_fast_ema: int = 20,
    trend_slow_ema: int = 50,
    trend_slope_lookback: int = 5,
    trend_slope_threshold: float = 0.1,
) -> pd.DataFrame:
    """Evaluate support, resistance, and combined bounce rates for each EMA period."""
    periods = list(ema_periods)
    if not periods or any(period < 1 for period in periods):
        raise ValueError("ema_periods must contain at least one positive period")
    rows = []
    for period in periods:
        summary = analyze_ema_interactions(bars, period, atr_period, atr_multiple, mode=mode, trend_mode=trend_mode, trend_fast_ema=trend_fast_ema, trend_slow_ema=trend_slow_ema, trend_slope_lookback=trend_slope_lookback, trend_slope_threshold=trend_slope_threshold).summary
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


def _best_bounce_pct(
    bars: pd.DataFrame,
    ema_periods: Iterable[int],
    atr_period: int,
    atr_multiple: float,
    mode: JudgmentMode = "close",
    trend_mode: TrendMode = "improved",
    trend_fast_ema: int = 20,
    trend_slow_ema: int = 50,
    trend_slope_lookback: int = 5,
    trend_slope_threshold: float = 0.1,
) -> float:
    scores = [analyze_ema_interactions(bars, p, atr_period, atr_multiple, mode=mode, trend_mode=trend_mode, trend_fast_ema=trend_fast_ema, trend_slow_ema=trend_slow_ema, trend_slope_lookback=trend_slope_lookback, trend_slope_threshold=trend_slope_threshold).summary["combined_bounce_pct"] for p in ema_periods]
    return max((float(score) for score in scores if score is not None), default=0.0)


NullDesign = Literal["bar_permutation", "legacy_return_shuffle"]


def _null_bar_permutation(data: pd.DataFrame, rng) -> pd.DataFrame:
    """Permute whole bars, each carrying its own wick geometry and its own gap.

    Preserves the return multiset AND the intrabar geometry. The legacy design
    below does not: rebuilding high/low from open/close yields candles with zero
    wicks, which changes the instrument being measured rather than only its
    ordering.
    """
    open_, high, low, close = (data[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    shape = np.c_[np.log(high / open_), np.log(low / open_), np.log(close / open_)]
    gap = np.r_[0.0, np.log(open_[1:] / close[:-1])]
    order = rng.permutation(len(data))
    log_price = np.log(close[0])
    o_, h_, l_, c_ = [], [], [], []
    for k in order:
        op = log_price + gap[k]
        o_.append(op)
        h_.append(op + shape[k, 0])
        l_.append(op + shape[k, 1])
        c_.append(op + shape[k, 2])
        log_price = c_[-1]
    out = data.copy()
    out["open"], out["high"], out["low"], out["close"] = (
        np.exp(np.asarray(x)) for x in (o_, h_, l_, c_)
    )
    return out


def _null_legacy_return_shuffle(data: pd.DataFrame, rng) -> pd.DataFrame:
    """MISCALIBRATED. Retained only to reproduce results published before the fix.

    Shuffles log returns and rebuilds bars as high=max(open,close),
    low=min(open,close). The resulting candles have zero wicks (measured mean
    wick fraction 0.00 vs 0.40 for real bars), which mis-centres the null by
    about -9.7pp in close mode and collapses `body` and `full` judgment into the
    same function. Measured false-positive rate on pure random walks:
    43% at p<0.05, against a 5% target.
    """
    closes = data["close"].to_numpy(float)
    shuffled = rng.permutation(np.diff(np.log(closes)))
    out = data.copy()
    out["close"] = np.r_[closes[0], closes[0] * np.exp(np.cumsum(shuffled))]
    out["open"] = out["close"].shift(1).fillna(out["close"])
    out["high"] = out[["open", "close"]].max(axis=1)
    out["low"] = out[["open", "close"]].min(axis=1)
    return out


_NULL_DESIGNS = {
    "bar_permutation": _null_bar_permutation,
    "legacy_return_shuffle": _null_legacy_return_shuffle,
}


def monte_carlo_p_value(
    bars: pd.DataFrame,
    actual_bounce_pct: float,
    ema_periods: Iterable[int],
    atr_period: int = 200,
    atr_multiple: float = 0.5,
    simulations: int = 1000,
    seed: int = 42,
    mode: JudgmentMode = "close",
    trend_mode: TrendMode = "improved",
    trend_fast_ema: int = 20,
    trend_slow_ema: int = 50,
    trend_slope_lookback: int = 5,
    trend_slope_threshold: float = 0.1,
    null_design: NullDesign = "bar_permutation",
    dither: bool = True,
) -> float:
    """Permutation test for the best bounce rate across `ema_periods`.

    `ema_periods` MUST be the same list used for the scan whose optimum produced
    `actual_bounce_pct`; the max-statistic correction is only valid over the set
    actually searched.

    Returns the randomized (dithered) p-value. Note `(exceeded + U)/(B + 1)`,
    not `exceeded/B`: a finite permutation test cannot justify p = 0, and the
    smallest value B permutations support is 1/(B+1). Use
    `monte_carlo_report` when you need the exceedance count and seed recorded.
    """
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if null_design not in _NULL_DESIGNS:
        raise ValueError(f"null_design must be one of {sorted(_NULL_DESIGNS)}")
    periods = list(ema_periods)
    if not periods:
        raise ValueError("ema_periods must not be empty")
    data = _validate_bars(bars)
    rng = np.random.default_rng(seed)
    build_null = _NULL_DESIGNS[null_design]

    exceeded = 0
    for _ in range(simulations):
        synthetic = build_null(data, rng)
        if _best_bounce_pct(synthetic, periods, atr_period, atr_multiple, mode, trend_mode,
                            trend_fast_ema, trend_slow_ema, trend_slope_lookback,
                            trend_slope_threshold) >= actual_bounce_pct:
            exceeded += 1

    draw = float(rng.random()) if dither else 1.0
    return round((exceeded + draw) / (simulations + 1), 6)


def monte_carlo_report(*args, **kwargs) -> dict[str, float | int | str | None]:
    """`monte_carlo_p_value` plus the provenance needed to reproduce it.

    A p-value from a finite permutation sample should never be reported without
    the exceedance count, the null design, and the seed.
    """
    simulations = kwargs.get("simulations", 1000)
    seed = kwargs.get("seed", 42)
    null_design = kwargs.get("null_design", "bar_permutation")
    dither = kwargs.get("dither", True)
    p = monte_carlo_p_value(*args, **kwargs)
    conservative = min(1.0, (round(p * (simulations + 1)) + 1) / (simulations + 1))
    return {
        "p_value": p,
        "p_value_conservative": round(conservative, 6),
        "simulations": simulations,
        "null_design": null_design,
        "seed": seed,
        "dither_rule": "p = (n_exceeded + U)/(B+1), U~Uniform(0,1)" if dither else "none",
    }
