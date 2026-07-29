from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from .engine import AnalysisResult


def build_interaction_figure(result: AnalysisResult, title: str) -> go.Figure:
    """Build a self-contained interactive price/EMA/band interaction chart."""
    bars = result.bars
    interactions = result.interactions
    figure = go.Figure()
    x_values = bars.index.strftime("%Y-%m-%d %H:%M:%S UTC")
    figure.add_trace(go.Candlestick(x=x_values, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"], name="Price"))
    figure.add_trace(go.Scatter(x=x_values, y=bars["ema"], mode="lines", name=f"EMA {result.summary['ema_period']}", line={"color": "#f1c40f", "width": 1.5}))
    figure.add_trace(go.Scatter(x=x_values, y=bars["upper_band"], mode="lines", name="Upper ATR band", line={"color": "#7f8c8d", "dash": "dot", "width": 1}))
    figure.add_trace(go.Scatter(x=x_values, y=bars["lower_band"], mode="lines", name="Lower ATR band", line={"color": "#7f8c8d", "dash": "dot", "width": 1}, fill="tonexty", fillcolor="rgba(127,140,141,0.08)"))

    if not interactions.empty:
        starts = interactions.copy()
        starts["start"] = starts["start"].astype("datetime64[ns, UTC]")
        for side, color, label in [("support", "#3498db", "Support entry"), ("resistance", "#9b59b6", "Resistance entry")]:
            subset = starts[starts["side"] == side]
            if not subset.empty:
                figure.add_trace(go.Scatter(x=subset["start"].dt.strftime("%Y-%m-%d %H:%M:%S UTC"), y=subset["entry_price"], mode="markers", name=label, marker={"color": color, "size": 7, "symbol": "circle-open"}, customdata=subset[["outcome", "trend"]], hovertemplate="%{x}<br>Entry: $%{y:.2f}<br>Outcome: %{customdata[0]}<br>Trend: %{customdata[1]}<extra></extra>"))
        for outcome, color, label, symbol in [("bounce", "#2ecc71", "Bounce exit", "triangle-up"), ("penetration", "#e74c3c", "Penetration exit", "x")]:
            subset = starts[starts["outcome"] == outcome]
            if not subset.empty:
                figure.add_trace(go.Scatter(x=subset["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC"), y=subset["exit_price"], mode="markers", name=label, marker={"color": color, "size": 9, "symbol": symbol}, customdata=subset[["side", "trend"]], hovertemplate="%{x}<br>Exit: $%{y:.2f}<br>Side: %{customdata[0]}<br>Trend at entry: %{customdata[1]}<extra></extra>"))

    figure.update_layout(title=title, template="plotly_dark", hovermode="x unified", xaxis_title="Time (UTC)", yaxis_title="Price", xaxis={"rangeslider": {"visible": False}}, legend={"orientation": "h", "y": 1.02, "yanchor": "bottom"}, height=800)
    return figure


def write_interaction_plot(result: AnalysisResult, output: str | Path, title: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_interaction_figure(result, title)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path


def build_period_scan_figure(scan, title: str):
    """Build a chart of support, resistance, and combined bounce rates by EMA period."""
    import plotly.graph_objects as go

    figure = go.Figure()
    traces = [
        ("support_bounce_pct", "Support bounce %", "#3498db", "circle"),
        ("resistance_bounce_pct", "Resistance bounce %", "#9b59b6", "square"),
        ("combined_bounce_pct", "Combined bounce %", "#2ecc71", "diamond"),
    ]
    x = scan["ema_period"].tolist()
    for column, name, color, symbol in traces:
        figure.add_trace(go.Scatter(x=x, y=scan[column].tolist(), mode="lines+markers", name=name, line={"color": color, "width": 2}, marker={"color": color, "symbol": symbol, "size": 7}, connectgaps=False))
    figure.update_layout(title=title, template="plotly_dark", xaxis_title="EMA period", yaxis_title="Bounce rate (%)", yaxis={"range": [0, 100]}, hovermode="x unified", height=650)
    return figure


def write_period_scan_plot(scan, output: str | Path, title: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_period_scan_figure(scan, title).write_html(str(path), include_plotlyjs=True, full_html=True)
    return path
