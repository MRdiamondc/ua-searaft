"""Offline threshold sweep: the actual experiment.

Everything here consumes a cached trace, so a sweep over thousands of stopping
configurations runs on a CPU in seconds. Three references are computed, and the
second one is the one a reviewer will ask for:

1. **full budget** -- the model as shipped (n = max_iters).
2. **cost-matched fixed N** -- the fixed budget that costs the *same* mean
   number of iterations as our adaptive policy, linearly interpolated between
   integer budgets. Beating "full" is trivial (that is just spending less);
   beating a cost-matched fixed budget is the only result that shows
   *adaptivity* itself helps.
3. **oracle** -- the per-sample cheapest index that stays within the accuracy
   slack. It upper-bounds any global rule and turns "we could do better" into a
   number: the oracle gap in iterations.

Uncertainty on the headline claim is reported two ways: a percentile bootstrap
CI of the paired difference, and a paired Wilcoxon signed-rank test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .criterion import StopConfig, decide_stop_batch, iters_of
from .utils import out_path, provenance, save_json, ukey


@dataclass
class Point:
    """One evaluated stopping configuration."""

    label: str
    signal: str
    mode: str
    q: float
    tau_rel: float
    tau_delta: float
    patience: int
    budget: float
    metric: float
    full_metric: float
    delta_rel: float
    saving_iters: float
    matched_fixed: float
    win_vs_matched: float
    stop_hist: List[int] = field(default_factory=list)

    def as_row(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "signal": self.signal,
            "mode": self.mode,
            "q": self.q,
            "tau_rel": self.tau_rel,
            "tau_delta": self.tau_delta,
            "patience": self.patience,
            "budget": self.budget,
            "metric": self.metric,
            "full_metric": self.full_metric,
            "delta_rel": self.delta_rel,
            "saving_iters": self.saving_iters,
            "matched_fixed": self.matched_fixed,
            "win_vs_matched": self.win_vs_matched,
        }


def signal_keys(trace: Dict[str, Any]) -> List[str]:
    """Every ``U:<mode>@<q>`` series present in the trace."""
    return sorted(k for k in trace if isinstance(k, str) and k.startswith("U:"))


def metric_key(trace: Dict[str, Any]) -> str:
    """``epe`` when ground truth exists, else ``ref`` (distance to full budget)."""
    if bool(trace.get("has_gt", False)) and "epe" in trace:
        return "epe"
    return "ref"


def _matrix(trace: Dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(trace[key], dtype=np.float64)
    return np.atleast_2d(arr)


def fixed_curve(trace: Dict[str, Any], metric: Optional[str] = None) -> np.ndarray:
    """Mean metric of every *fixed* budget: the accuracy-vs-iterations curve."""
    metrics = _matrix(trace, metric or metric_key(trace))
    return np.nanmean(metrics, axis=0)


def matched_fixed_metric(curve: np.ndarray, budget: float, list_offset: int = 0) -> float:
    """Interpolate the fixed-budget curve at a fractional mean budget."""
    index = float(budget) - float(list_offset)
    index = min(max(index, 0.0), float(len(curve) - 1))
    low = int(np.floor(index))
    high = min(low + 1, len(curve) - 1)
    weight = index - low
    return float((1.0 - weight) * curve[low] + weight * curve[high])


def _matched_per_sample(
    metrics: np.ndarray, budget: float, list_offset: int = 0
) -> np.ndarray:
    """Per-sample cost-matched fixed-N metric, for the paired test."""
    index = float(budget) - float(list_offset)
    index = min(max(index, 0.0), float(metrics.shape[1] - 1))
    low = int(np.floor(index))
    high = min(low + 1, metrics.shape[1] - 1)
    weight = index - low
    return (1.0 - weight) * metrics[:, low] + weight * metrics[:, high]


def build_grid(cfg: Dict[str, Any]) -> List[StopConfig]:
    """Cartesian grid of stopping configurations, de-duplicated per mode.

    ``u_only`` ignores ``tau_delta`` and ``d_only`` ignores ``tau_rel``, so the
    naive product would evaluate the same policy dozens of times and inflate
    the apparent size of the search. We drop those duplicates explicitly.
    """
    sweep_cfg = dict(cfg.get("sweep", {}))
    base = StopConfig.from_dict(cfg.get("criterion"))
    tau_rels = [float(x) for x in sweep_cfg.get("tau_rels", [base.tau_rel])]
    tau_deltas = [float(x) for x in sweep_cfg.get("tau_deltas", [base.tau_delta])]
    patiences = [int(x) for x in sweep_cfg.get("patiences", [base.patience])]
    modes = [str(m) for m in sweep_cfg.get("modes", [base.mode])]

    grid: List[StopConfig] = []
    seen = set()
    for mode in modes:
        for tau_rel in tau_rels:
            for tau_delta in tau_deltas:
                for patience in patiences:
                    r = tau_rel if mode in ("and", "or", "u_only") else float("inf")
                    d = tau_delta if mode in ("and", "or", "d_only") else float("inf")
                    key = (mode, r, d, patience)
                    if key in seen:
                        continue
                    seen.add(key)
                    grid.append(
                        base.replace(
                            mode=mode,
                            tau_rel=tau_rel if np.isfinite(r) else base.tau_rel,
                            tau_delta=tau_delta if np.isfinite(d) else base.tau_delta,
                            patience=patience,
                        )
                    )
    return grid


def evaluate(
    trace: Dict[str, Any],
    signal: str,
    stop_cfg: StopConfig,
    metric: Optional[str] = None,
    curve: Optional[np.ndarray] = None,
) -> Tuple[Point, np.ndarray, np.ndarray]:
    """Replay one configuration over the whole trace.

    Returns the summary ``Point`` plus the per-sample metric array and the
    per-sample stop indices, which the bootstrap and the paired test need.
    """
    metric = metric or metric_key(trace)
    metrics = _matrix(trace, metric)
    signal_matrix = _matrix(trace, signal)
    deltas = _matrix(trace, "D")
    list_offset = int(trace.get("list_offset", 0))

    indices = decide_stop_batch(signal_matrix, deltas, stop_cfg)
    rows = np.arange(metrics.shape[0])
    per_sample = metrics[rows, indices]
    budgets = iters_of(indices, list_offset).astype(np.float64)

    full_index = min(int(stop_cfg.max_iters), metrics.shape[1] - 1)
    full_metric = float(np.nanmean(metrics[:, full_index]))
    mean_metric = float(np.nanmean(per_sample))
    budget = float(np.mean(budgets))
    curve = fixed_curve(trace, metric) if curve is None else curve
    matched = matched_fixed_metric(curve, budget, list_offset)

    point = Point(
        label="%s|%s" % (signal, stop_cfg.label()),
        signal=signal,
        mode=stop_cfg.mode,
        q=float(stop_cfg.q),
        tau_rel=float(stop_cfg.tau_rel),
        tau_delta=float(stop_cfg.tau_delta),
        patience=int(stop_cfg.patience),
        budget=budget,
        metric=mean_metric,
        full_metric=full_metric,
        delta_rel=float((mean_metric - full_metric) / (abs(full_metric) + 1e-12)),
        saving_iters=float(iters_of(full_index, list_offset)) - budget,
        matched_fixed=matched,
        win_vs_matched=float(matched - mean_metric),
        stop_hist=np.bincount(indices, minlength=metrics.shape[1]).tolist(),
    )
    return point, per_sample, indices


def pareto(points: Sequence[Point]) -> List[Point]:
    """Points not dominated in (budget, metric); cheapest first."""
    keep: List[Point] = []
    for candidate in sorted(points, key=lambda p: (p.budget, p.metric)):
        dominated = any(
            other.budget <= candidate.budget and other.metric <= candidate.metric
            for other in keep
        )
        if not dominated:
            keep.append(candidate)
    return keep


def oracle(
    trace: Dict[str, Any],
    metric: Optional[str] = None,
    slack: float = 0.01,
    max_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-sample cheapest index within ``slack`` of that sample's full metric.

    This is not achievable by any global rule -- it peeks at the answer -- but
    it bounds the headroom and belongs in the paper as the third column.
    """
    metric = metric or metric_key(trace)
    metrics = _matrix(trace, metric)
    list_offset = int(trace.get("list_offset", 0))
    top = metrics.shape[1] - 1 if max_index is None else int(max_index)

    chosen = np.full(metrics.shape[0], top, dtype=np.int64)
    for row in range(metrics.shape[0]):
        target = metrics[row, top] * (1.0 + float(slack)) + 1e-12
        for index in range(0, top + 1):
            value = metrics[row, index]
            if np.isfinite(value) and value <= target:
                chosen[row] = index
                break
    per_sample = metrics[np.arange(metrics.shape[0]), chosen]
    budgets = iters_of(chosen, list_offset).astype(np.float64)
    return {
        "budget": float(np.mean(budgets)),
        "metric": float(np.nanmean(per_sample)),
        "full_metric": float(np.nanmean(metrics[:, top])),
        "slack": float(slack),
        "hist": np.bincount(chosen, minlength=metrics.shape[1]).tolist(),
    }


def bootstrap_ci(
    a: np.ndarray,
    b: Optional[np.ndarray] = None,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> Dict[str, float]:
    """Percentile bootstrap CI of ``mean(a)`` or of the paired ``mean(a - b)``."""
    x = np.asarray(a, dtype=np.float64)
    if b is not None:
        x = x - np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, x.size, size=(int(n_boot), x.size))
    means = x[draws].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.percentile(means, 100.0 * alpha / 2.0)),
        "hi": float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
        "n": int(x.size),
        "n_boot": int(n_boot),
    }


def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test, with a sign-test fallback."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    diff = x - y
    if diff.size == 0 or np.allclose(diff, 0.0):
        return {"test": "none", "p": float("nan"), "n": int(diff.size)}
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return {"test": "wilcoxon", "stat": float(stat), "p": float(p), "n": int(diff.size)}
    except Exception:
        wins = int(np.sum(diff < 0))
        total = int(np.sum(diff != 0))
        # two-sided sign test via the normal approximation
        if total == 0:
            return {"test": "sign", "p": float("nan"), "n": 0}
        z = (wins - total / 2.0) / np.sqrt(total / 4.0)
        from math import erfc

        return {
            "test": "sign",
            "stat": float(z),
            "p": float(erfc(abs(z) / np.sqrt(2.0))),
            "n": total,
            "wins": wins,
        }


def run_sweep(
    trace: Dict[str, Any],
    cfg: Dict[str, Any],
    out: Optional[str] = None,
    signals: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluate the whole grid on every signal and pick an honest headline."""
    sweep_cfg = dict(cfg.get("sweep", {}))
    slack = float(sweep_cfg.get("budget_slack", 0.01))
    n_boot = int(sweep_cfg.get("bootstrap", 10000))
    seed = int(sweep_cfg.get("seed", 1234))

    metric = metric_key(trace)
    metrics = _matrix(trace, metric)
    curve = fixed_curve(trace, metric)
    list_offset = int(trace.get("list_offset", 0))
    grid = build_grid(cfg)
    keys = list(signals) if signals else signal_keys(trace)
    if not keys:
        raise ValueError("trace contains no 'U:<mode>@<q>' signals")

    points: List[Point] = []
    per_sample_of: Dict[str, np.ndarray] = {}
    for signal in keys:
        for stop_cfg in grid:
            point, per_sample, _ = evaluate(trace, signal, stop_cfg, metric, curve)
            points.append(point)
            per_sample_of[point.label] = per_sample

    full_index = min(int(StopConfig.from_dict(cfg.get("criterion")).max_iters), metrics.shape[1] - 1)
    full_metric = float(np.nanmean(metrics[:, full_index]))
    full_budget = float(iters_of(full_index, list_offset))
    front = pareto(points)

    # headline = cheapest Pareto point still within `slack` of the full budget
    affordable = [p for p in front if p.delta_rel <= slack]
    best = min(affordable, key=lambda p: p.budget) if affordable else None

    headline: Dict[str, Any] = {"found": bool(best is not None), "slack": slack}
    if best is not None:
        matched_samples = _matched_per_sample(metrics, best.budget, list_offset)
        ours = per_sample_of[best.label]
        headline.update(
            {
                "label": best.label,
                "signal": best.signal,
                "mode": best.mode,
                "budget": best.budget,
                "full_budget": full_budget,
                "iters_saved": best.saving_iters,
                "iters_saved_frac": best.saving_iters / max(full_budget, 1e-12),
                "metric": best.metric,
                "full_metric": full_metric,
                "delta_rel": best.delta_rel,
                "matched_fixed_index": best.budget - list_offset,
                "matched_fixed_metric": best.matched_fixed,
                "ci_vs_matched": bootstrap_ci(
                    ours, matched_samples, n_boot=n_boot, seed=seed
                ),
                "wilcoxon_vs_matched": paired_test(ours, matched_samples),
                "ci_vs_full": bootstrap_ci(
                    ours, metrics[:, full_index], n_boot=n_boot, seed=seed
                ),
            }
        )

    oracle_result = oracle(trace, metric, slack=slack, max_index=full_index)
    if best is not None:
        headline["oracle_gap_iters"] = float(best.budget - oracle_result["budget"])

    result = {
        "metric": metric,
        "full_metric": full_metric,
        "full_budget": full_budget,
        "budget": full_budget,
        "n_samples": int(metrics.shape[0]),
        "n_configs": len(points),
        "n_signals": len(keys),
        "fixed_curve": [float(x) for x in curve],
        "pareto": [p.as_row() for p in front],
        "oracle": oracle_result,
        "all_rows": [p.as_row() for p in points],
        "headline": headline,
        "provenance": provenance(),
    }

    if best is not None:
        print(
            "HEADLINE %s : %.2f -> %.2f iters (-%.0f%%), %s %.4f -> %.4f (%+.2f%%), "
            "vs matched fixed %.4f, p=%.3g"
            % (
                best.label,
                full_budget,
                best.budget,
                100.0 * headline["iters_saved_frac"],
                metric,
                full_metric,
                best.metric,
                100.0 * best.delta_rel,
                best.matched_fixed,
                headline["wilcoxon_vs_matched"].get("p", float("nan")),
            )
        )
        print(
            "ORACLE   %.2f iters at %s %.4f  (gap %.2f iters)"
            % (
                oracle_result["budget"],
                metric,
                oracle_result["metric"],
                headline["oracle_gap_iters"],
            )
        )
    else:
        print(
            "HEADLINE none: no configuration stayed within %.1f%% of the full budget"
            % (100.0 * slack)
        )

    if out is not None or cfg.get("out"):
        path = save_json(result, out_path(out or cfg.get("out"), "tables", "sweep.json"))
        print("saved -> %s" % path)
    return result
