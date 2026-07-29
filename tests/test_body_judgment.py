import pandas as pd

from ema_sr import engine


def test_body_mode_uses_close_to_enter_but_body_to_exit(monkeypatch):
    index = pd.date_range("2024-01-01", periods=7, freq="h", tz="UTC")
    bars = pd.DataFrame({
        "open": [100.0, 100.0, 110.0, 110.0, 105.0, 105.2, 110.4],
        "high": [100.1, 100.1, 110.1, 110.1, 105.3, 107.1, 108.1],
        "low": [99.9, 99.9, 109.9, 104.9, 104.9, 105.1, 106.9],
        "close": [100.0, 100.0, 110.0, 105.5, 105.7, 108.2, 111.0],
        "volume": [1000.0] * 7,
    }, index=index)
    monkeypatch.setattr(engine, "_atr", lambda data, period: pd.Series(1.0, index=data.index))

    result = engine.analyze_ema_interactions(bars, ema_period=2, atr_period=1, atr_multiple=0.5, mode="body")

    assert result.interactions.iloc[0]["start"] == index[3]
    assert result.interactions.iloc[0]["outcome"] == "bounce"
    assert result.interactions.iloc[0]["timestamp"] == index[6]
