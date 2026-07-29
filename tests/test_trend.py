import pandas as pd

from ema_sr.engine import _entry_trend


def test_entry_trend_uses_ema_slope():
    row = pd.Series({"ema": 101.0})
    previous = pd.Series({"ema": 100.0})
    assert _entry_trend(row, previous) == "uptrend"
    assert _entry_trend(pd.Series({"ema": 99.0}), previous) == "downtrend"
    assert _entry_trend(pd.Series({"ema": 100.0}), previous) == "flat"
