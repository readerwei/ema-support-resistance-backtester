from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .data import fetch_yahoo_bars
from .engine import analyze_ema_interactions, monte_carlo_p_value, scan_ema_periods
from .plotting import write_interaction_plot, write_period_scan_plot


def _ema_periods(value: str) -> list[int]:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 1:
        return parts
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("EMA range must be PERIOD or START:STOP:STEP")
    start, stop, step = parts
    if step <= 0 or stop < start:
        raise argparse.ArgumentTypeError("EMA range must have positive step and STOP >= START")
    return list(range(start, stop + 1, step))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test EMA support/resistance effectiveness")
    parser.add_argument("--symbol", required=True, help="Ticker, e.g. AAPL or SPY")
    parser.add_argument("--start", required=True, help="UTC ISO date/time, e.g. 2023-01-01")
    parser.add_argument("--end", default=None, help="UTC ISO date/time; defaults to now")
    parser.add_argument("--timeframe", default="1d", help="1m, 15m, 1h, 4h, 1d, 1wk, etc.")
    parser.add_argument("--session", choices=["regular", "extended"], default="regular", help="Regular hours or Yahoo pre/post-market data")
    parser.add_argument("--ema", type=int, default=70, help="EMA period to analyze")
    parser.add_argument("--ema-range", type=_ema_periods, default=None, help="Scan EMA periods: PERIOD or START:STOP:STEP")
    parser.add_argument("--scan-output", help="Write full EMA period scan results to CSV")
    parser.add_argument("--scan-plot", help="Write support/resistance/combined scan chart to HTML")
    parser.add_argument("--atr-period", type=int, default=200)
    parser.add_argument("--atr-multiple", type=float, default=0.5)
    parser.add_argument("--monte-carlo", type=int, default=0, metavar="N", help="Run N shuffled-return simulations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Write classified interactions to CSV")
    parser.add_argument("--plot", help="Write an interactive Plotly HTML chart")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = fetch_yahoo_bars(args.symbol, args.start, end, args.timeframe, session=args.session)
    result = analyze_ema_interactions(bars, args.ema, args.atr_period, args.atr_multiple)
    report = dict(result.summary)
    report.update({"symbol": args.symbol.upper(), "timeframe": args.timeframe, "session": args.session, "start": str(bars.index.min()), "end": str(bars.index.max()), "data_source": bars.attrs.get("source")})
    if args.ema_range:
        scan = scan_ema_periods(bars, args.ema_range, args.atr_period, args.atr_multiple)
        report["ema_scan"] = scan.where(scan.notna(), None).to_dict(orient="records")
        best = scan.dropna(subset=["combined_bounce_pct"])
        if not best.empty:
            report["best_combined_ema_period"] = int(best.loc[best["combined_bounce_pct"].idxmax(), "ema_period"])
        if args.scan_output:
            scan.to_csv(args.scan_output, index=False)
            report["scan_csv"] = args.scan_output
        if args.scan_plot:
            write_period_scan_plot(scan, args.scan_plot, f"{args.symbol.upper()} {args.timeframe} EMA period scan")
            report["scan_plot_html"] = args.scan_plot
    if args.monte_carlo:
        actual = report["combined_bounce_pct"]
        report["monte_carlo_p_value"] = None if actual is None else monte_carlo_p_value(bars, float(actual), args.ema_range or [args.ema], args.atr_period, args.atr_multiple, args.monte_carlo, args.seed)
    if args.output:
        result.interactions.to_csv(args.output, index=False)
        report["interactions_csv"] = args.output
    if args.plot:
        write_interaction_plot(result, args.plot, f"{args.symbol.upper()} {args.timeframe} EMA/ATR interactions")
        report["plot_html"] = args.plot
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
