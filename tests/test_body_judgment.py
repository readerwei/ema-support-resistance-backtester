import pandas as pd

from ema_sr.engine import _bar_relation


def test_body_relation_requires_both_open_and_close_outside():
    bands = {"upper_band": 105.0, "lower_band": 95.0}
    assert _bar_relation(pd.Series({"open": 100.0, "close": 102.0, **bands})) == "inside"
    assert _bar_relation(pd.Series({"open": 106.0, "close": 108.0, **bands})) == "above"
    assert _bar_relation(pd.Series({"open": 92.0, "close": 94.0, **bands})) == "below"
    assert _bar_relation(pd.Series({"open": 104.0, "close": 106.0, **bands})) == "mixed"


def test_body_mode_is_exposed_in_analysis_summary():
    bars = pd.DataFrame({
        "open": [100.0, 101.0, 99.0, 102.0],
        "high": [101.0, 102.0, 100.0, 103.0],
        "low": [99.0, 100.0, 98.0, 101.0],
        "close": [100.0, 101.0, 99.0, 102.0],
        "volume": [1000.0] * 4,
    }, index=pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"))
    from ema_sr.engine import analyze_ema_interactions
    assert analyze_ema_interactions(bars, ema_period=1, atr_period=1, mode="body").summary["mode"] == "body"
