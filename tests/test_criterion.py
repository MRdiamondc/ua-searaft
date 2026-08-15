"""Tests for the stopping rule and the latency cost model. No GPU, no torch.

These run in under a second and cover the parts that decide every number in the
paper: when the rule fires, what "iterations" means, and whether the linear cost
model really recovers the constants we quote.

    python -m pytest tests/test_criterion.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ua_stop.criterion import (  # noqa: E402
    EPS,
    MODES,
    StopConfig,
    budget,
    decide_stop,
    decide_stop_batch,
    delta_signal,
    iters_of,
    rel_change,
)
from ua_stop.latency import fit_linear, saving_curve, time_of  # noqa: E402

#: The measured T4 table, 1080x1920, median of 25 reps. Kept verbatim so the
#: constants quoted in the README stay checkable.
MEASURED = [
    (1, 1069.52, 0.0), (2, 1209.09, 0.0), (3, 1280.21, 0.0), (4, 1361.69, 0.0),
    (5, 1432.98, 0.0), (6, 1509.56, 0.0), (7, 1586.75, 0.0), (8, 1652.79, 0.0),
    (9, 1726.08, 0.0), (10, 1798.99, 0.0), (11, 1870.42, 0.0), (12, 1945.47, 0.0),
]


def test_rel_change_first_is_inf():
    r = rel_change([2.0, 1.0, 0.5])
    assert np.isinf(r[0])  # no predecessor, so index 0 can never fire
    assert r[1] == pytest.approx(0.5, rel=1e-6)
    assert r[2] == pytest.approx(0.5, rel=1e-3)


def test_rel_change_handles_zero_without_dividing_by_zero():
    r = rel_change([0.0, 0.0])
    assert np.isfinite(r[1])
    assert r[1] == pytest.approx(0.0)
    assert EPS > 0


def test_delta_signal_forces_first_to_inf():
    d = delta_signal([5.0, 4.0, 1.0])
    assert np.isinf(d[0])
    assert d[1] == pytest.approx(4.0)
    rel = delta_signal([5.0, 4.0, 1.0], use_abs_delta=False)
    assert np.isinf(rel[0])
    assert rel[1] == pytest.approx(1.0, rel=1e-3)  # normalised by index 1


def test_decide_stop_fires_when_both_conditions_hold():
    u = [10.0, 5.0, 4.9, 4.89, 4.889]
    d = [np.inf, 3.0, 0.01, 0.001, 0.0001]
    cfg = StopConfig(tau_rel=0.05, tau_delta=0.05, patience=1, max_iters=4)
    assert decide_stop(u, d, cfg) == 2


def test_patience_requires_a_streak():
    u = [10.0, 5.0, 4.9, 3.0, 2.99]
    d = [np.inf, 3.0, 0.01, 2.0, 0.01]
    cfg = StopConfig(tau_rel=0.05, tau_delta=0.05, patience=2, max_iters=4)
    # index 2 satisfies the rule but index 3 breaks it, so only the streak
    # ending at index 4 is long enough
    assert decide_stop(u, d, cfg) == 4


def test_mode_fixed_returns_full_budget():
    u = [1.0, 1.0, 1.0]
    d = [np.inf, 0.0, 0.0]
    assert decide_stop(u, d, StopConfig(mode="fixed", max_iters=2)) == 2


def test_d_only_ignores_uncertainty():
    u = [1.0, 1.0, 1.0, 1.0]  # a dead signal, as measured upstream
    d = [np.inf, 5.0, 0.001, 0.001]
    stop = decide_stop(u, d, StopConfig(mode="d_only", tau_delta=0.01, max_iters=3))
    assert stop == 2


def test_u_only_ignores_flow_delta():
    u = [10.0, 5.0, 4.99, 4.98]
    d = [np.inf, 9.0, 9.0, 9.0]  # never small
    stop = decide_stop(u, d, StopConfig(mode="u_only", tau_rel=0.01, max_iters=3))
    assert stop == 2


def test_tau_abs_short_circuits_relative_modes():
    u = [10.0, 0.001, 0.001]
    d = [np.inf, 9.0, 9.0]
    cfg = StopConfig(mode="and", tau_rel=1e-9, tau_delta=1e-9, tau_abs=0.01, max_iters=2)
    assert decide_stop(u, d, cfg) == 1


def test_never_stops_before_min_iters():
    u = [1.0, 1.0, 1.0, 1.0]
    d = [np.inf, 0.0, 0.0, 0.0]
    cfg = StopConfig(mode="or", tau_rel=1.0, tau_delta=1.0, min_iters=2, max_iters=3)
    assert decide_stop(u, d, cfg) == 2


def test_falls_back_to_full_budget_when_never_firing():
    u = [10.0, 8.0, 6.0, 4.0]
    d = [np.inf, 5.0, 5.0, 5.0]
    cfg = StopConfig(tau_rel=1e-9, tau_delta=1e-9, max_iters=3)
    assert decide_stop(u, d, cfg) == 3


def test_batch_matches_the_loop():
    rng = np.random.default_rng(0)
    u = np.cumsum(rng.random((7, 6)), axis=1)[:, ::-1].copy()
    d = rng.random((7, 6))
    d[:, 0] = np.inf
    cfg = StopConfig(tau_rel=0.2, tau_delta=0.5, max_iters=5)
    batched = decide_stop_batch(u, d, cfg)
    loop = [decide_stop(u[i], d[i], cfg) for i in range(u.shape[0])]
    assert list(batched) == loop


def test_iters_of_and_budget():
    assert list(iters_of(np.array([0, 1, 5]), 0)) == [0, 1, 5]
    assert list(iters_of(np.array([0, 1, 5]), -1)) == [0, 0, 4]  # clipped at zero
    assert budget([2, 4], 0) == pytest.approx(3.0)


def test_stopconfig_validation_and_label():
    with pytest.raises(ValueError):
        StopConfig(mode="nope")
    with pytest.raises(ValueError):
        StopConfig(patience=0)
    with pytest.raises(ValueError):
        StopConfig(min_iters=5, max_iters=2)
    with pytest.raises(ValueError):
        StopConfig(mode="abs_only")  # requires tau_abs
    label = StopConfig(q=0.8, tau_rel=0.02, tau_delta=0.05, patience=1).label()
    assert label == "and|q0.80|r0.02|d0.05|p1"
    assert "d_only" in MODES


def test_replace_keeps_validation():
    cfg = StopConfig().replace(mode="or", patience=3)
    assert cfg.mode == "or" and cfg.patience == 3
    with pytest.raises(ValueError):
        cfg.replace(mode="bogus")


def test_fit_linear_recovers_the_measured_constants():
    """The README quotes a = 1042.02 ms and b = 76.15 ms. Verify, don't trust."""
    fit = fit_linear(MEASURED)
    assert fit["a_fixed_ms"] == pytest.approx(1042.02, abs=0.5)
    assert fit["b_iter_ms"] == pytest.approx(76.15, abs=0.05)
    assert fit["n_max"] == 12
    assert fit["share_at_n_max"] == pytest.approx(0.47, abs=0.01)
    # R^2 is 0.9958, not 0.9999, and that is not a bad fit: the n=1 point sits
    # ~49 ms ABOVE the line because the first refinement iteration pays a
    # one-off warm-up cost. It is a property of the GPU, and it is exactly why
    # we report medians over 25 reps instead of single timings.
    assert fit["r2"] > 0.995


def test_saving_curve_ceiling_is_the_iteration_share():
    fit = fit_linear(MEASURED)
    curve = saving_curve(fit)
    assert curve[0]["n"] == 0
    # stopping at zero iterations is the hard ceiling, ~46-47% on a T4
    assert curve[0]["saving"] == pytest.approx(fit["share_at_n_max"], abs=1e-6)
    assert 0.45 < curve[0]["saving"] < 0.48
    assert curve[-1]["saving"] == pytest.approx(0.0)
    # monotonically decreasing saving as the budget grows
    savings = [row["saving"] for row in curve]
    assert all(savings[i] >= savings[i + 1] for i in range(len(savings) - 1))


def test_time_of_matches_the_measured_table_within_noise():
    """The linear model holds for n >= 2; n = 1 carries the warm-up penalty."""
    fit = fit_linear(MEASURED)
    residuals = {n: abs(time_of(fit, n) - ms) for n, ms, _ in MEASURED}
    for n, error in residuals.items():
        if n >= 2:
            assert error < 20.0, "n=%d is %.1f ms off the linear model" % (n, error)
    # the single worst point is the first budget, by a wide margin
    assert residuals[1] == max(residuals.values())
    assert residuals[1] > 40.0
