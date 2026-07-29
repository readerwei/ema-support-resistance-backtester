# EMA Support/Resistance Backtester

A backtest-only research tool inspired by the linked video. It measures whether an EMA behaves like probabilistic support or resistance for any Yahoo Finance ticker and bar timeframe.

## What it measures

For each EMA interaction, the engine creates ATR bands around the EMA:

- `upper = EMA + ATR × atr_multiple`
- `lower = EMA - ATR × atr_multiple`
- A close entering the band starts an interaction.
- In an uptrend, exiting above the upper band is a **support bounce**; exiting below the lower band is a **penetration**.
- In a downtrend, exiting below the lower band is a **resistance bounce**; exiting above the upper band is a **penetration**.

Signals use completed bars only. This is a research measurement, not a trading strategy or profitability claim.

## Install

Requires Python 3.11+ and `uv`:

```bash
cd ~/ema_sr_backtester
uv sync
```

## Examples

Daily AAPL, test EMA 70:

```bash
uv run ema-sr --symbol AAPL --start 2018-01-01 --end 2025-01-01 --timeframe 1d --ema 70 --atr-period 200 --atr-multiple 0.5
```

Scan EMA periods 20 through 200 and run a 500-simulation permutation test:

```bash
uv run ema-sr \
  --symbol SPY \
  --start 2015-01-01 \
  --end 2025-01-01 \
  --timeframe 1d \
  --ema-range 20:200:5 \
  --atr-period 50 \
  --atr-multiple 0.5 \
  --monte-carlo 500 \
  --output spy_interactions.csv \
  --plot spy_ema_interactions.html
```

The HTML chart includes candlesticks, EMA, ATR bands, interaction-entry markers, green bounce exits, and red penetration exits. It is self-contained and can be opened in a browser without a local server.

To scan multiple EMA periods, pass `--ema-range`. The range is inclusive and can be a single period or `START:STOP:STEP`. The scan reports support, resistance, and combined bounce percentages, interaction counts, the best combined period, and can write both CSV and an interactive scan chart:

```bash
uv run ema-sr \
  --symbol AAPL \
  --start 2026-07-22 \
  --end 2026-07-30 \
  --timeframe 1m \
  --ema-range 20:200:5 \
  --atr-period 50 \
  --atr-multiple 0.5 \
  --scan-output aapl_ema_scan.csv \
  --scan-plot aapl_ema_scan.html
```

The scan chart has three lines: support bounce rate, resistance bounce rate, and combined bounce rate. As in the video, higher EMA periods may produce fewer interactions, so inspect the interaction counts alongside the percentages.

Intraday examples:

```bash
uv run ema-sr --symbol TSLA --start 2025-01-01 --timeframe 15m --ema 70 --atr-period 200
uv run ema-sr --symbol QQQ --start 2025-01-01 --timeframe 4h --ema 70 --atr-period 50
```

By default, the downloader uses regular market hours. To include Yahoo's available pre-market and post-market bars:

```bash
uv run ema-sr --symbol AAPL --start 2025-01-01 --timeframe 1m --session extended --ema 20 --atr-period 50
```

`--session extended` maps to Yahoo's `includePrePost=true`; it does not guarantee a complete 24-hour overnight equity feed.

Bounce/penetration judgment defaults to the close-based method. Use `--judgment body` for the stricter candle-body method: a bar is considered above or below a band only when **both its open and close** are beyond that band. Use `--judgment full` for the strictest method: the entire candle, including wicks, must be beyond the band (`low > upper` for an above exit, or `high < lower` for a below exit). Mixed candles remain unclassified, so the interaction stays active until a later candle satisfies the selected rule.

Each classified interaction also records the trend at entry. The default `--trend-mode slope` is the simple EMA-slope baseline (`uptrend`, `downtrend`, or `flat`). For the improved multi-factor regime, use `--trend-mode improved`: fast/slow EMA alignment, ATR-normalized slow-EMA slope, and close location relative to the slow EMA must agree; otherwise the label is `range/mixed`. Defaults are fast EMA 20, slow EMA 50, slope lookback 5 bars, and normalized-slope threshold 0.1 ATR. Configure them with `--trend-fast-ema`, `--trend-slow-ema`, `--trend-slope-lookback`, and `--trend-slope-threshold`. The label is fixed at entry and appears in interaction CSV files and chart hover details.

Yahoo Finance does not provide unlimited intraday history. In particular, 1-minute data is generally limited to recent days, so use a provider with appropriate historical coverage for serious testing. Timeframes such as `4h` are resampled from Yahoo's hourly data.

## Output

The CLI prints JSON containing bar count, support/resistance interaction counts, bounce percentages, the selected parameters, optional EMA scan results, and an optional Monte Carlo p-value. `--output` writes one row per completed interaction to CSV.

## Tests

```bash
uv run --group dev pytest -q
```

## Research cautions

- EMA/ATR parameters can be overfit; use an untouched out-of-sample period.
- A bounce percentage does not establish trade profitability.
- Include spreads, slippage, commissions, market hours, corporate actions, and survivorship bias in any production-grade study.
- The permutation test shuffles log returns and compares the observed score against synthetic paths; it is evidence against one null model, not proof of causality.
- This package contains no broker credentials and no order-submission code by construction.
