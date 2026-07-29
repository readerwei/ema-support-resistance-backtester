import pandas as pd

from ema_sr.engine import _full_stick_relation


def test_full_stick_relation_requires_wicks_outside():
    bands = {"upper_band": 105.0, "lower_band": 95.0}
    row = lambda **values: pd.Series({**values, **bands})
    assert _full_stick_relation(row(open=106, high=108, low=105.1, close=107)) == "above"
    assert _full_stick_relation(row(open=94, high=94.9, low=92, close=93)) == "below"
    assert _full_stick_relation(row(open=106, high=108, low=104, close=107)) == "mixed"
