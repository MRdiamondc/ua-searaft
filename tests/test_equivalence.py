"""The offline replay must be an EQUIVALENCE, not an approximation.

The entire method rests on one claim: replaying a threshold against a cached
trace gives exactly the flow the model would have produced had it stopped
there. `scripts/selftest.py` T6 checks that on the real network with
``torch.equal``; these tests check the bookkeeping around it -- index vs
iteration count, cost-matched interpolation, Pareto logic and the oracle -- all
without torch, so they run in CI.

    python -m pytest tests/test_equivalence.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ua_stop.criterion import StopConfig, decide_stop_batch  # noqa: E402
from ua_stop.sweep import (  # noqa: E402
    bootstrap_ci,
    build_grid,
    evaluate,
    fixed_curve,
    matched_fixed_metric,
    metric_key,
    oracle,
    paired_test,
    pareto,
    signal_keys,
)
from ua_stop.utils import ukey  # noqa: E402

K = 6


def make_trace(n: int = 8, seed: int = 0, has_gt: bool = True):
    """A synthetic trace with the same keys and shapes as a real npz."""
    rng = np.random.default_rng(seed)
    steps = np.arange(K + 1)
    # Uncertainty decays and saturates. The per-sample rate spans an order of
    # magnitude on purpose: with a narrow spread every sample stops at the same
    # index, and then these tests would pass even for a constant policy.
    rate = 0.20 + 2.0 * rng.random((n, 1))
    u = 1.0 + 3.0 * np.exp(-rate * steps[None, :])
    d = np.concatenate([np.full((n, 1), np.inf), 2.0 * np.exp(-rate * steps[None, 1:])], axis=1)
    epe = 0.5 + 2.0 * np.exp(-rate * steps[None, :]) + 0.01 * rng.random((n, K + 1))
    trace = {
        ukey("raw", 0.8): u,
        "Umean:raw": u * 0.9,
        "D": d,
        "epe": epe,
        "ref": np.abs(epe - epe[:, -1:]),
        "has_gt": has_gt,
        "list_offset": 0,
        "names": [str(i) for i in range(n)],
    }
    return trace


def test_metric_key_prefers_ground_truth():
    assert metric_key(make_trace(has_gt=True)) == "epe"
    assert metric_key(make_trace(has_gt=False)) == "ref"


def test_signal_keys_finds_only_u_series():
    keys = signal_keys(make_trace())
    assert keys == [ukey("raw", 0.8)]
    assert all(k.startswith("U:") for k in keys)


def test_evaluate_reproduces_a_manual_gather():
    """This is the replay equivalence, in index space."""
    trace = make_trace()
    # tau_delta = 1e9 disables the flow-delta condition, exactly as the real
    # sweep grid does, so this test isolates the uncertainty read-out
    cfg = StopConfig(tau_rel=0.08, tau_delta=1e9, max_iters=K)
    point, per_sample, indices = evaluate(trace, ukey("raw", 0.8), cfg)

    manual_idx = decide_stop_batch(trace[ukey("raw", 0.8)], trace["D"], cfg)
    assert list(indices) == list(manual_idx)
    manual = trace["epe"][np.arange(len(manual_idx)), manual_idx]
    assert np.allclose(per_sample, manual)
    assert point.metric == pytest.approx(float(np.mean(manual)))
    assert point.budget == pytest.approx(float(np.mean(manual_idx)))
    # the rule must actually be adaptive on this trace
    assert len(set(indices.tolist())) >= 3, "fixture is not adaptive: %s" % indices


def test_full_budget_config_is_the_full_budget():
    trace = make_trace()
    point, per_sample, indices = evaluate(
        trace, ukey("raw", 0.8), StopConfig(mode="fixed", max_iters=K)
    )
    assert list(indices) == [K] * trace["epe"].shape[0]
    assert point.metric == pytest.approx(point.full_metric)
    assert point.delta_rel == pytest.approx(0.0)
    assert point.saving_iters == pytest.approx(0.0)


def test_matched_fixed_interpolates_between_integers():
    curve = np.array([4.0, 3.0, 2.0, 1.0])
    assert matched_fixed_metric(curve, 1.0) == pytest.approx(3.0)
    assert matched_fixed_metric(curve, 1.5) == pytest.approx(2.5)
    # out of range budgets clamp instead of extrapolating
    assert matched_fixed_metric(curve, 99.0) == pytest.approx(1.0)
    assert matched_fixed_metric(curve, -5.0) == pytest.approx(4.0)


def test_fixed_curve_is_the_column_mean():
    trace = make_trace()
    curve = fixed_curve(trace)
    assert curve.shape == (K + 1,)
    assert curve[0] == pytest.approx(float(np.mean(trace["epe"][:, 0])))
    assert curve[-1] == pytest.approx(float(np.mean(trace["epe"][:, -1])))


def test_pareto_keeps_only_non_dominated_points():
    trace = make_trace()
    points = []
    for tau in (0.001, 0.01, 0.05, 0.2, 0.5):
        point, _, _ = evaluate(
            trace, ukey("raw", 0.8), StopConfig(tau_rel=tau, tau_delta=0.5, max_iters=K)
        )
        points.append(point)
    front = pareto(points)
    assert front
    budgets = [p.budget for p in front]
    metrics = [p.metric for p in front]
    assert budgets == sorted(budgets)
    # cheaper must mean worse on a non-dominated front
    assert all(metrics[i] >= metrics[i + 1] for i in range(len(metrics) - 1))


def test_oracle_is_at_least_as_good_as_any_global_rule():
    trace = make_trace()
    ora = oracle(trace, slack=0.01)
    point, _, _ = evaluate(
        trace, ukey("raw", 0.8), StopConfig(tau_rel=0.02, tau_delta=0.05, max_iters=K)
    )
    assert ora["budget"] <= point.budget + 1e-9
    assert 0.0 <= ora["budget"] <= K


def test_build_grid_dedupes_ignored_thresholds():
    cfg = {
        "criterion": {"max_iters": K},
        "sweep": {
            "modes": ["u_only"],
            "tau_rels": [0.01, 0.02],
            "tau_deltas": [0.01, 0.05, 0.1],  # ignored by u_only
            "patiences": [1],
        },
    }
    grid = build_grid(cfg)
    # 2 tau_rels x 1 patience, not 2 x 3
    assert len(grid) == 2


def test_bootstrap_ci_brackets_a_known_difference():
    rng = np.random.default_rng(3)
    a = rng.normal(1.0, 0.05, 200)
    b = a + rng.normal(0.2, 0.03, 200)
    ci = bootstrap_ci(a, b, n_boot=2000, seed=7)
    assert ci["mean"] == pytest.approx(-0.2, abs=0.02)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["hi"] < 0.0, "a real difference must exclude zero"
    assert ci["n"] == 200


def test_bootstrap_ci_collapses_when_the_difference_is_constant():
    """A constant paired difference has zero variance, so lo == hi.

    That is correct behaviour, and worth knowing before anyone reports a
    suspiciously tight interval as evidence of a strong result.
    """
    a = np.linspace(0.0, 1.0, 50)
    ci = bootstrap_ci(a, a + 0.2, n_boot=500, seed=1)
    assert ci["lo"] == pytest.approx(ci["hi"])
    assert ci["mean"] == pytest.approx(-0.2)


def test_paired_test_detects_a_real_difference():
    rng = np.random.default_rng(4)
    a = rng.normal(1.0, 0.05, 60)
    result = paired_test(a, a + 0.3)
    assert result["p"] < 0.01
    assert result["test"] in ("wilcoxon", "sign")
    assert paired_test(a, a)["test"] == "none"
