"""Calibration tests for the permutation null.

The bug these exist to prevent: a null that changes the *instrument* rather than
only the ordering. Rebuilding bars as high=max(open,close), low=min(open,close)
produces wickless candles, which mis-centres the null and yielded a measured
43% false-positive rate at p<0.05 on pure random walks.

The cheap invariant is geometric and deterministic, so it runs in CI:
a valid null must preserve the distribution of wick size. The expensive check
(p-value uniformity over hundreds of no-effect datasets) is marked `slow`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ema_sr.engine import (
    _null_bar_permutation,
    _null_legacy_return_shuffle,
    _validate_bars,
    analyze_ema_interactions,
    monte_carlo_p_value,
)


def random_walk(n: int = 400, seed: int = 0, substeps: int = 8) -> pd.DataFrame:
    """Pure GBM with realistic wicks. Contains no support or resistance."""
    rng = np.random.default_rng(seed)
    price = 100.0
    rows = []
    for _ in range(n):
        path = [price]
        for _ in range(substeps):
            path.append(path[-1] * np.exp(0.012 / np.sqrt(substeps) * rng.standard_normal()))
        rows.append((path[0], max(path), min(path), path[-1]))
        price = path[-1]
    o, h, l, c = map(np.array, zip(*rows))
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": 1e6},
        index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
    )


def wick_fraction(bars: pd.DataFrame) -> float:
    span = bars["high"] - bars["low"]
    body = (bars["close"] - bars["open"]).abs()
    return float(((span - body) / span.replace(0, np.nan)).mean())


def test_null_preserves_wick_geometry():
    """The default null must not change the shape of a candle."""
    data = _validate_bars(random_walk(seed=1))
    rng = np.random.default_rng(0)
    real = wick_fraction(data)
    permuted = wick_fraction(_null_bar_permutation(data, rng))
    assert real > 0.2, "fixture should have meaningful wicks"
    assert permuted == pytest.approx(real, abs=0.02), (
        f"null changed bar geometry: real wick fraction {real:.3f} -> {permuted:.3f}. "
        "A null that alters the instrument cannot test a hypothesis about it."
    )


def test_legacy_null_is_wickless_and_is_the_known_defect():
    """Pins the defect so nobody reintroduces it believing it is equivalent."""
    data = _validate_bars(random_walk(seed=1))
    rng = np.random.default_rng(0)
    legacy = _null_legacy_return_shuffle(data, rng)
    assert wick_fraction(legacy) == pytest.approx(0.0, abs=1e-9)


def test_legacy_null_collapses_body_and_full_judgment():
    """With no wicks, `body` and `full` become the same function — a clear tell."""
    data = _validate_bars(random_walk(seed=2))
    legacy = _null_legacy_return_shuffle(data, np.random.default_rng(0))
    body = analyze_ema_interactions(legacy, 20, 50, 0.5, mode="body").summary
    full = analyze_ema_interactions(legacy, 20, 50, 0.5, mode="full").summary
    assert body["interactions"] == full["interactions"]
    assert body["combined_bounce_pct"] == full["combined_bounce_pct"]


def test_p_value_is_never_zero():
    """A finite permutation sample cannot justify p = 0; floor is 1/(B+1)."""
    data = _validate_bars(random_walk(seed=3))
    p = monte_carlo_p_value(data, 100.0, [20], 50, 0.5, simulations=10, seed=5)
    assert p > 0.0, "p must be strictly positive with finite permutations"
    assert p <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["close", "full"])
def test_pvalue_distribution_is_uniform_on_no_effect_data(mode):
    """The real calibration check. Slow — run with `-m slow`.

    Verified at scale (200 datasets x 60 permutations):
      close: KS p=0.95, P(p<0.05)=0.070, median 0.480  -> calibrated
      full : KS p=0.23, P(p<0.05)=0.045, median 0.565  -> uniform, mildly
             conservative; does not meet every preregistered criterion
    """
    pvals = []
    for k in range(25):
        data = _validate_bars(random_walk(n=500, seed=300 + k))
        observed = analyze_ema_interactions(data, 20, 50, 0.5, mode=mode).summary[
            "combined_bounce_pct"]
        if observed is None:
            continue
        pvals.append(monte_carlo_p_value(data, observed, [20], 50, 0.5,
                                         simulations=25, seed=700 + k, mode=mode))
    p = np.array(pvals)
    assert len(p) >= 15
    # loose bounds: this is a smoke test, not the full study
    assert (p < 0.05).mean() < 0.25, f"false-positive rate {(p < 0.05).mean():.2f} too high"
    assert 0.30 < np.median(p) < 0.70, f"median p {np.median(p):.3f} is shifted"
