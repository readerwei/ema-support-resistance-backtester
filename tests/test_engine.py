import numpy as np
import pandas as pd

from ema_sr.engine import analyze_ema_interactions, monte_carlo_p_value


def bars(closes):
    index = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000.0})


def test_interactions_are_classified_without_lookahead():
    data = bars([10, 11, 12, 11, 10, 9, 10, 11, 12, 11, 10])
    result = analyze_ema_interactions(data, ema_period=3, atr_period=3, atr_multiple=0.05)
    assert set(result.interactions["outcome"].dropna().unique()) <= {"bounce", "penetration"}
    assert result.interactions["outcome"].notna().any()
    assert result.summary["interactions"] == int(result.interactions["outcome"].notna().sum())


def test_band_entry_can_complete_as_bounce():
    data = bars([10, 11, 12, 11.2, 11.8, 12.5, 13.0, 12.2, 11.0, 10.0])
    result = analyze_ema_interactions(data, ema_period=3, atr_period=3, atr_multiple=0.05)
    assert "bounce" in set(result.interactions["outcome"])


def test_monte_carlo_is_reproducible_and_bounded():
    data = bars(np.sin(np.arange(80) / 3) + np.arange(80) / 20 + 20)
    result = analyze_ema_interactions(data, ema_period=8, atr_period=8, atr_multiple=0.05)
    p1 = monte_carlo_p_value(data, result.summary["combined_bounce_pct"], ema_periods=[4, 8], atr_period=8, atr_multiple=0.05, simulations=20, seed=7)
    p2 = monte_carlo_p_value(data, result.summary["combined_bounce_pct"], ema_periods=[4, 8], atr_period=8, atr_multiple=0.05, simulations=20, seed=7)
    assert p1 == p2
    assert 0 <= p1 <= 1
