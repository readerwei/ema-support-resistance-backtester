import pandas as pd

from ema_sr.engine import _entry_trend


def test_entry_trend_uses_ema_slope():
    row = pd.Series({"ema": 101.0})
    previous = pd.Series({"ema": 100.0})
    assert _entry_trend(row, previous) == "uptrend"
    assert _entry_trend(pd.Series({"ema": 99.0}), previous) == "downtrend"
    assert _entry_trend(pd.Series({"ema": 100.0}), previous) == "flat"


def test_improved_entry_trend_requires_alignment_slope_and_price_location():
    row = pd.Series({"trend_fast_ema": 105.0, "trend_slow_ema": 100.0, "trend_slope_atr": 0.2, "close": 106.0})
    assert _entry_trend(row, None, mode="improved", slope_threshold=0.1) == "uptrend"
    price_below = row.copy()
    price_below["close"] = 99.0
    weak_slope = row.copy()
    weak_slope["trend_slope_atr"] = 0.05
    down = row.copy()
    down[["trend_fast_ema", "trend_slow_ema", "trend_slope_atr", "close"]] = [95.0, 100.0, -0.2, 94.0]
    assert _entry_trend(price_below, None, mode="improved", slope_threshold=0.1) == "range/mixed"
    assert _entry_trend(weak_slope, None, mode="improved", slope_threshold=0.1) == "range/mixed"
    assert _entry_trend(down, None, mode="improved", slope_threshold=0.1) == "downtrend"
