"""Full calibration study for the geometry-preserving permutation null.

Question: on data containing NO support/resistance, is the p-value distribution
uniform? Median and 5%-tail are not enough — we test the whole distribution
against U(0,1) with a Kolmogorov-Smirnov statistic.

Parallel over datasets. Writes JSON so the result is inspectable, not just printed.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from ema_sr.engine import _validate_bars, analyze_ema_interactions  # noqa: E402

N_DATASETS = 200
SIMS = 60
N_BARS = 900
EMA, ATR_P, ATR_M = 70, 200, 0.5


def randomwalk(n: int, seed: int, substeps: int = 8) -> pd.DataFrame:
    """Pure GBM with realistic wicks. No support, no resistance, no memory."""
    rng = np.random.default_rng(seed)
    p = 100.0
    rows = []
    for _ in range(n):
        path = [p]
        for _ in range(substeps):
            path.append(path[-1] * np.exp(0.012 / np.sqrt(substeps) * rng.standard_normal()))
        rows.append((path[0], max(path), min(path), path[-1]))
        p = path[-1]
    o, h, l, c = map(np.array, zip(*rows))
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": 1e6},
        index=pd.date_range("2016-01-01", periods=n, freq="D", tz="UTC"),
    )


def null_fixed(d: pd.DataFrame, rng) -> pd.DataFrame:
    """Permute whole bars, each keeping its own wick geometry and its own gap."""
    o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    shape = np.c_[np.log(h / o), np.log(l / o), np.log(c / o)]
    gap = np.r_[0.0, np.log(o[1:] / c[:-1])]
    idx = rng.permutation(len(d))
    logp = np.log(c[0])
    O, H, L, C = [], [], [], []
    for k in idx:
        op = logp + gap[k]
        O.append(op)
        H.append(op + shape[k, 0])
        L.append(op + shape[k, 1])
        C.append(op + shape[k, 2])
        logp = C[-1]
    out = d.copy()
    out["open"], out["high"], out["low"], out["close"] = (
        np.exp(np.asarray(x)) for x in (O, H, L, C)
    )
    return out


def one_dataset(args):
    seed, mode = args
    d = _validate_bars(randomwalk(N_BARS, seed))
    obs = analyze_ema_interactions(d, EMA, ATR_P, ATR_M, mode=mode).summary[
        "combined_bounce_pct"]
    if obs is None:
        return None
    rng = np.random.default_rng(900_000 + seed)
    ge = valid = 0
    for _ in range(SIMS):
        s = analyze_ema_interactions(null_fixed(d, rng), EMA, ATR_P, ATR_M,
                                     mode=mode).summary["combined_bounce_pct"]
        if s is None:
            continue
        valid += 1
        ge += int(s >= obs)
    if not valid:
        return None
    # Randomized permutation p-value: (#{stat>=obs} + U) / (B+1), U~Uniform(0,1).
    # Removes the discreteness that makes a KS test against continuous U(0,1)
    # reject on granularity alone. Exactly uniform under H0.
    u = float(rng.random())
    return (ge + u) / (valid + 1)


def ks_uniform(p: np.ndarray) -> tuple[float, float]:
    """KS statistic against U(0,1) plus its asymptotic p-value."""
    x = np.sort(p)
    n = len(x)
    i = np.arange(1, n + 1)
    d = max((i / n - x).max(), (x - (i - 1) / n).max())
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    j = np.arange(1, 101)
    pv = 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * j**2 * lam**2))
    return float(d), float(min(max(pv, 0.0), 1.0))


if __name__ == "__main__":
    results = {}
    for mode in ("close", "full"):
        jobs = [(2000 + k, mode) for k in range(N_DATASETS)]
        with ProcessPoolExecutor() as ex:
            pvals = [v for v in ex.map(one_dataset, jobs, chunksize=4) if v is not None]
        p = np.array(pvals)
        d_stat, ks_p = ks_uniform(p)
        results[mode] = dict(
            n_datasets=len(p),
            sims_each=SIMS,
            fpr_05=float((p < 0.05).mean()),
            fpr_10=float((p < 0.10).mean()),
            median=float(np.median(p)),
            mean=float(p.mean()),
            ks_stat=d_stat,
            ks_pvalue=ks_p,
            deciles=[float(x) for x in np.percentile(p, np.arange(10, 100, 10))],
            pvalues=[float(x) for x in p],
        )
        r = results[mode]
        print(f"\n=== {mode} mode, {r['n_datasets']} no-effect datasets x {SIMS} perms ===")
        print(f"  P(p<0.05)  : {r['fpr_05']:.3f}   (target 0.050)")
        print(f"  P(p<0.10)  : {r['fpr_10']:.3f}   (target 0.100)")
        print(f"  median p   : {r['median']:.3f}   (target 0.500)")
        print(f"  KS vs U(0,1): D={d_stat:.4f}, p={ks_p:.4f} "
              f"-> {'UNIFORM (calibrated)' if ks_p > 0.05 else 'NOT uniform'}")

    with open("calibration_study.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote calibration_study.json")
