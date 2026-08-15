"""A distribution-free risk certificate for the selected threshold (RCPS).

The sweep picks whichever threshold looked best *on the data*. Reporting its
accuracy as if it had been fixed in advance is exactly the criticism early-exit
papers attract. So the choice is wrapped in Risk-Controlling Prediction Sets:
walk the candidate thresholds from least to most aggressive (fixed-sequence
testing, hence no multiplicity correction), and keep the last one whose risk
upper confidence bound is still below the tolerance.

Risk, normalised and clipped to [0, 1] as Hoeffding requires:

    R(lambda) = E[ min(1, max(0, EPE_lambda - EPE_full - epsilon) / scale) ]

So "risk" means "excess error beyond a tolerated epsilon". The guarantee is

    P( R(lambda_hat) <= alpha ) >= 1 - delta

over draws of the calibration set. With roughly 40 KITTI pairs the bound is
loose but honest; Bentkus or Waudby-Smith--Ramdas bounds are tighter drop-in
replacements for :func:`hoeffding_ucb` if a reviewer asks for them.

Always report the held-out risk too (``split_calibration``): the certificate
is a guarantee about the *procedure*, the held-out number is the evidence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import out_path, provenance, save_json


def bounded_loss(
    metric_lambda: Sequence[float],
    metric_full: Sequence[float],
    epsilon: float = 0.05,
    scale: float = 1.0,
) -> np.ndarray:
    """Per-sample loss in [0, 1]: excess error beyond ``epsilon``, normalised.

    ``epsilon`` is the accuracy we are willing to give away for free (in the
    metric's own units, e.g. px of EPE); ``scale`` converts the remaining
    excess into [0, 1]. Both belong in the paper's table caption, because the
    certificate is meaningless without them.
    """
    a = np.asarray(metric_lambda, dtype=np.float64)
    b = np.asarray(metric_full, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("shapes differ: %s vs %s" % (a.shape, b.shape))
    excess = (a - b - float(epsilon)) / max(float(scale), 1e-12)
    return np.clip(excess, 0.0, 1.0)


def hoeffding_ucb(mean: float, n: int, delta: float) -> float:
    """One-sided Hoeffding upper confidence bound for a mean in [0, 1]."""
    if n <= 0:
        return 1.0
    return float(mean) + math.sqrt(math.log(1.0 / max(float(delta), 1e-12)) / (2.0 * int(n)))


def rcps_select(
    candidates: Sequence[Dict[str, Any]],
    alpha: float = 0.05,
    delta: float = 0.1,
    epsilon: float = 0.05,
    scale: float = 1.0,
) -> Dict[str, Any]:
    """Fixed-sequence RCPS over an ordered candidate list.

    ``candidates`` must be ordered from LEAST to MOST aggressive (i.e. by
    decreasing budget), each entry a dict with

        label          str
        budget         float, mean iterations
        metric         (N,) per-sample metric of the stopped flows
        metric_full    (N,) per-sample metric at the full budget

    Testing stops at the first candidate that fails, which is what makes the
    family-wise guarantee free.
    """
    trail: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    n_calibration = 0

    for candidate in candidates:
        losses = bounded_loss(
            candidate["metric"], candidate["metric_full"], epsilon=epsilon, scale=scale
        )
        losses = losses[np.isfinite(losses)]
        n_calibration = int(losses.size)
        mean = float(losses.mean()) if losses.size else 1.0
        ucb = hoeffding_ucb(mean, losses.size, delta)
        passed = bool(ucb <= float(alpha))
        record = {
            "label": candidate.get("label"),
            "budget": float(candidate.get("budget", float("nan"))),
            "risk_mean": mean,
            "risk_ucb": ucb,
            "passed": passed,
            "n": int(losses.size),
        }
        trail.append(record)
        if not passed:
            break
        selected = {
            **{k: v for k, v in candidate.items() if k not in ("metric", "metric_full")},
            "certificate": {
                "alpha": float(alpha),
                "delta": float(delta),
                "epsilon": float(epsilon),
                "scale": float(scale),
                "risk_mean": mean,
                "risk_ucb": ucb,
                "n_calibration": int(losses.size),
                "statement": (
                    "P(R <= %.3f) >= %.3f over calibration draws, with R the mean "
                    "excess error beyond %.3f" % (float(alpha), 1.0 - float(delta), float(epsilon))
                ),
            },
        }

    return {
        "alpha": float(alpha),
        "delta": float(delta),
        "epsilon": float(epsilon),
        "scale": float(scale),
        "n_calibration": n_calibration,
        "trail": trail,
        "selected": selected,
    }


def split_calibration(
    n: int, split: float = 0.5, seed: int = 1234
) -> Tuple[np.ndarray, np.ndarray]:
    """Random calibration/test split of sample indices (default 50/50)."""
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n))
    cut = int(round(float(split) * int(n)))
    cut = min(max(cut, 1), max(int(n) - 1, 1))
    return order[:cut], order[cut:]


def certify(
    trace: Dict[str, Any],
    sweep_result: Dict[str, Any],
    cfg: Dict[str, Any],
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """Calibrate on one half of the trace, report the risk on the other half.

    The candidate list is the Pareto front from the sweep, ordered by
    decreasing budget, so "more aggressive" literally means "cheaper".
    """
    from .criterion import StopConfig
    from .sweep import _matrix, evaluate, metric_key

    conf_cfg = dict(cfg.get("conformal", {}))
    alpha = float(conf_cfg.get("alpha", 0.05))
    delta = float(conf_cfg.get("delta", 0.1))
    epsilon = float(conf_cfg.get("epsilon", 0.05))
    scale = float(conf_cfg.get("scale", 1.0))
    split = float(conf_cfg.get("split", 0.5))
    seed = int(cfg.get("sweep", {}).get("seed", 1234))

    metric = metric_key(trace)
    metrics = _matrix(trace, metric)
    n_samples = metrics.shape[0]
    cal_idx, test_idx = split_calibration(n_samples, split=split, seed=seed)

    base = StopConfig.from_dict(cfg.get("criterion"))
    full_index = min(int(base.max_iters), metrics.shape[1] - 1)
    full_per_sample = metrics[:, full_index]

    rows = sorted(
        sweep_result.get("pareto", []), key=lambda r: -float(r.get("budget", 0.0))
    )
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        stop_cfg = base.replace(
            mode=str(row["mode"]),
            tau_rel=float(row["tau_rel"]),
            tau_delta=float(row["tau_delta"]),
            patience=int(row["patience"]),
        )
        _, per_sample, _ = evaluate(trace, str(row["signal"]), stop_cfg, metric)
        candidates.append(
            {
                "label": row["label"],
                "signal": row["signal"],
                "mode": row["mode"],
                "budget": float(row["budget"]),
                "metric": per_sample[cal_idx],
                "metric_full": full_per_sample[cal_idx],
                "_per_sample": per_sample,
            }
        )

    result = rcps_select(
        [{k: v for k, v in c.items() if k != "_per_sample"} for c in candidates],
        alpha=alpha,
        delta=delta,
        epsilon=epsilon,
        scale=scale,
    )
    result["metric"] = metric
    result["n_samples"] = int(n_samples)
    result["split"] = {"calibration": int(cal_idx.size), "test": int(test_idx.size)}

    if result.get("selected"):
        label = result["selected"]["label"]
        chosen = next(c for c in candidates if c["label"] == label)
        held_out_loss = bounded_loss(
            chosen["_per_sample"][test_idx],
            full_per_sample[test_idx],
            epsilon=epsilon,
            scale=scale,
        )
        result["held_out"] = {
            "n": int(test_idx.size),
            "risk_mean": float(np.nanmean(held_out_loss)),
            "metric": float(np.nanmean(chosen["_per_sample"][test_idx])),
            "full_metric": float(np.nanmean(full_per_sample[test_idx])),
            "violated": bool(float(np.nanmean(held_out_loss)) > alpha),
        }
        print(
            "CERTIFIED %s : risk %.4f (ucb %.4f <= alpha %.3f), held-out risk %.4f"
            % (
                label,
                result["selected"]["certificate"]["risk_mean"],
                result["selected"]["certificate"]["risk_ucb"],
                alpha,
                result["held_out"]["risk_mean"],
            )
        )
    else:
        print(
            "NO CANDIDATE CERTIFIED at alpha=%.3f delta=%.3f epsilon=%.3f: "
            "loosen epsilon or collect more calibration pairs" % (alpha, delta, epsilon)
        )

    result["provenance"] = provenance()
    path = save_json(result, out_path(out or cfg.get("out"), "tables", "conformal.json"))
    print("saved -> %s" % path)
    return result
