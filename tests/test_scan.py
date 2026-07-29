import numpy as np
import pandas as pd

from ema_sr.engine import analyze_ema_interactions, scan_ema_periods


def make_bars():
    index = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    close = pd.Series(np.sin(np.arange(100) / 3) + np.arange(100) / 20 + 20, index=index)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000.0})


def test_scan_returns_support_resistance_and_combined_rates():
    result = scan_ema_periods(make_bars(), [4, 8, 12], atr_period=8, atr_multiple=0.05)
    assert list(result["ema_period"]) == [4, 8, 12]
    assert {"support_bounce_pct", "resistance_bounce_pct", "combined_bounce_pct", "interactions"} <= set(result.columns)
    assert result["combined_bounce_pct"].notna().any()


def test_scan_matches_single_period_analysis():
    bars = make_bars()
    scan = scan_ema_periods(bars, [8], atr_period=8, atr_multiple=0.05).iloc[0]
    single = analyze_ema_interactions(bars, 8, 8, 0.05).summary
    assert scan["combined_bounce_pct"] == single["combined_bounce_pct"]
    assert scan["interactions"] == single["interactions"]
