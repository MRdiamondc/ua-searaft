"""How much time can adaptive stopping actually save? Measure, do not guess.

The saving is bounded by the *iteration share* of total latency, so this module
fits an explicit per-iteration cost model

    T(n) = a + b * n

where ``a`` is the fixed cost (feature encoders, correlation pyramid, padding,
I/O) and ``b`` the cost of one refinement iteration.

Measured on a Colab T4 at 1080x1920 (i.e. before the ``scale`` fix):

    a = 1042.02 ms   b = 76.15 ms   iteration share = 47% of T(12)
    criterion overhead = 2.838 ms per iteration = 3.7% of b

The paper reports 51% for the L model on a 3090, so 47% on a T4 is the same
regime. That share is the ceiling of the whole idea: stopping at zero
iterations saves 46.4% and nothing can save more. Hence the honest target of
20-30% wall-clock at under 1% accuracy loss -- stated up front rather than
discovered by a reviewer.

Only the three timing functions need torch; the cost model itself is numpy, so
``fit_linear``/``time_of``/``saving_curve`` are unit-tested without a GPU.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import out_path, provenance, save_json

TableRow = Tuple[int, float, float]

#: Report the criterion as "cheap" below this fraction of one iteration.
CHEAP_FRACTION = 0.15


def latency_table(
    model: Any,
    sample: Dict[str, Any],
    n_list: Optional[Sequence[int]] = None,
    warmup: int = 8,
    reps: int = 25,
) -> List[TableRow]:
    """Median and IQR wall-clock time of ``forward_once`` for each budget.

    Median, not mean: on a shared Colab GPU a single scheduling hiccup ruins a
    mean. The IQR is printed so the reader can see the measurement is stable
    (it settles below 5 ms from n = 5 upwards on a T4).
    """
    n_list = [int(n) for n in (n_list or range(1, int(getattr(model, "max_iters", 12)) + 1))]
    img1, img2 = sample["img1"], sample["img2"]

    for _ in range(int(warmup)):
        model.forward_once(img1, img2, iters=max(n_list))
    model.sync()

    table: List[TableRow] = []
    for n in n_list:
        times = []
        for _ in range(int(reps)):
            model.sync()
            start = time.perf_counter()
            model.forward_once(img1, img2, iters=n)
            model.sync()
            times.append((time.perf_counter() - start) * 1e3)
        arr = np.asarray(times, dtype=np.float64)
        median = float(np.median(arr))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        table.append((int(n), median, iqr))
        print("n=%2d  %8.2f ms   (iqr %.2f)" % (n, median, iqr))
    return table


def fit_linear(table: Sequence[TableRow]) -> Dict[str, float]:
    """Least-squares fit of ``T(n) = a + b n``; also returns the share and R^2."""
    n = np.asarray([row[0] for row in table], dtype=np.float64)
    t = np.asarray([row[1] for row in table], dtype=np.float64)
    if n.size < 2:
        raise ValueError("need at least two budgets to fit a cost model")
    design = np.stack([np.ones_like(n), n], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, t, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a + b * n
    ss_res = float(((t - pred) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    n_max = int(n.max())
    total = a + b * n_max
    return {
        "a_fixed_ms": a,
        "b_iter_ms": b,
        "n_max": n_max,
        "share_at_n_max": float(b * n_max / total) if total > 0 else float("nan"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def time_of(fit: Dict[str, float], n: float) -> float:
    """Predicted wall-clock time for a (possibly fractional, i.e. mean) budget."""
    return float(fit["a_fixed_ms"]) + float(fit["b_iter_ms"]) * float(n)


def saving_curve(fit: Dict[str, float], n_max: Optional[int] = None) -> List[Dict[str, float]]:
    """Relative time saved by every fixed budget, including n = 0 (the ceiling)."""
    top = int(n_max if n_max is not None else fit["n_max"])
    full = time_of(fit, top)
    rows = []
    for n in range(0, top + 1):
        ms = time_of(fit, n)
        rows.append(
            {
                "n": int(n),
                "ms": ms,
                "saving": float((full - ms) / full) if full > 0 else float("nan"),
            }
        )
    return rows


def criterion_overhead(
    model: Any,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    reps: int = 50,
    b_iter_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Cost of evaluating the stopping signal once, on the GPU, per iteration.

    If this is not small compared with ``b`` the whole method is pointless, so
    it is measured rather than assumed. On a T4 it is 2.838 ms, i.e. 3.7% of
    one iteration; anything above CHEAP_FRACTION of ``b`` (15%) is a red flag.
    """
    from .criterion import StopConfig, decide_stop
    from .uncertainty import global_uncertainty, pixel_uncertainty, resolve_mode

    unc_cfg = cfg.get("uncertainty", {})
    mode = resolve_mode(unc_cfg.get("primary_mode", "raw"))
    q = float(unc_cfg.get("primary_q", 0.8))
    var_min = float(cfg.get("model", {}).get("var_min", 0.0))
    var_max = float(cfg.get("model", {}).get("var_max", 10.0))
    stop_cfg = StopConfig.from_dict(cfg.get("criterion"))

    out = model.forward_trace(sample["img1"], sample["img2"])
    infos = out["infos"]
    if not infos:
        raise RuntimeError("model returned no info maps; cannot time the criterion")
    info = infos[-1]

    history = [1.0, 0.9]
    times = []
    for _ in range(int(reps)):
        model.sync()
        start = time.perf_counter()
        u = pixel_uncertainty(info, var_min=var_min, var_max=var_max, mode=mode)
        value = global_uncertainty(u, quantiles=(q,), valid=sample.get("valid"))[q]
        series = history + [value]
        decide_stop(series, [np.inf, 1.0, 0.5], stop_cfg)
        model.sync()
        times.append((time.perf_counter() - start) * 1e3)

    median = float(np.median(np.asarray(times)))
    result: Dict[str, Any] = {
        "criterion_ms": median,
        "mode": mode,
        "quantile": q,
        "reps": int(reps),
        "n_pixels": int(info.shape[-1] * info.shape[-2]),
    }
    if b_iter_ms:
        result["fraction_of_iteration"] = float(median / float(b_iter_ms))
        result["cheap"] = bool(result["fraction_of_iteration"] < CHEAP_FRACTION)
    return result


def measure(
    model: Any,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """Full latency study: table, cost model, criterion overhead, saving curve."""
    lat_cfg = cfg.get("latency", {})
    table = latency_table(
        model,
        sample,
        n_list=lat_cfg.get("n_list"),
        warmup=lat_cfg.get("warmup", 8),
        reps=lat_cfg.get("reps", 25),
    )
    fit = fit_linear(table)
    overhead = criterion_overhead(model, sample, cfg, b_iter_ms=fit["b_iter_ms"])
    curve = saving_curve(fit)

    print("")
    print("fixed cost a       = %.2f ms" % fit["a_fixed_ms"])
    print("per-iteration b    = %.2f ms" % fit["b_iter_ms"])
    print(
        "iteration share    = %.0f%% of T[%d]   (paper reports 51%% for L on a 3090)"
        % (100.0 * fit["share_at_n_max"], fit["n_max"])
    )
    print(
        "criterion overhead = %.3f ms per iteration  (%.1f%% of b)"
        % (
            overhead["criterion_ms"],
            100.0 * overhead.get("fraction_of_iteration", float("nan")),
        )
    )
    print("saving ceiling     = %.1f%% (stop at n=0)" % (100.0 * curve[0]["saving"]))

    result = {
        "table": [{"n": n, "ms": ms, "iqr": iqr} for n, ms, iqr in table],
        "fit": fit,
        "overhead": overhead,
        "saving_curve": curve,
        "model": getattr(model, "describe", lambda: "")(),
        "input_size": getattr(model, "input_size", None),
        "model_size": getattr(model, "model_size", None),
        "provenance": provenance(),
    }
    path = save_json(result, out_path(out or cfg.get("out"), "tables", "latency.json"))
    print("saved -> %s" % path)
    return result
