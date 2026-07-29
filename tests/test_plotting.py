import pandas as pd

from ema_sr.engine import analyze_ema_interactions
from ema_sr.plotting import write_interaction_plot


def test_interaction_plot_is_written(tmp_path):
    index = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    close = pd.Series(range(20), index=index, dtype=float) + 100
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
    result = analyze_ema_interactions(bars, ema_period=3, atr_period=3, atr_multiple=0.5)
    output = tmp_path / "chart.html"
    written = write_interaction_plot(result, output, "TEST")
    assert written == output
    assert output.exists()
    assert "Plotly" in output.read_text()
