#!/usr/bin/env python3
"""Seven hard assertions. Run this before trusting any number in this repo.

This file exists because of a real mistake. The first Colab notebook printed
``WARN`` when the uncertainty signal looked dead but still concluded
``SELF-TEST PASSED``, and it ran on random noise, where no optical-flow signal
is meaningful anyway. Two full days of "results" were built on a constant.

So here every test is an ``assert`` on a REAL image pair, and each failure
message says what to do next:

    T1  forward_once returns a (1, 2, H, W) flow at input resolution
    T2  forward_trace returns iters+1 entries and a measurable list offset
    T3  info has at least 4 channels (2 logits + 2 log-scales)
    T4  the ``scale`` config value actually reached the resize
    T5  at least one uncertainty read-out is ALIVE over the budget
    T6  a short run is bit-identical to the corresponding cached index
    T7  the stopping rule fires, and mode='fixed' really means full budget

T5 is the one that matters. T6 is what licenses the whole offline replay.

    python scripts/selftest.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np
import torch

from _cli import build_parser, cfg_from_args, opt_overrides

from ua_stop.criterion import StopConfig, decide_stop
from ua_stop.data import first_sample
from ua_stop.model_wrapper import build_model
from ua_stop.uncertainty import (
    MODES,
    clamp_pressure,
    global_uncertainty,
    pixel_uncertainty,
    signal_health,
)
from ua_stop.utils import out_path, provenance, save_json

#: Read-outs in the order we prefer them, best first. ``raw``/``geo`` are
#: pre-clamp and therefore the only ones that can move once var_min = 0.
PREFERRED = ("raw", "geo", "clamped_log", "clamped_lin", "alpha")


def real_pair(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A real image pair. Never random noise -- that was the original sin."""
    sample = first_sample(cfg)
    print("test pair: %s  shape %s  GT %s" % (
        sample.get("name"), tuple(sample["img1"].shape), bool(sample.get("has_gt"))
    ))
    return sample


def check(name: str, ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError("%s FAILED: %s" % (name, message))
    print("%s ok" % name)


def main(argv=None) -> int:
    parser = build_parser("Seven hard assertions on a real image pair.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = cfg_from_args(
        args,
        opt_overrides(
            [
                ("model.device", args.device),
                ("model.scale", args.scale),
                ("model.iters", args.iters),
            ]
        ),
    )

    results: Dict[str, Any] = {"provenance": provenance()}
    model = build_model(cfg)
    print(model.describe())
    sample = real_pair(cfg)
    img1, img2 = sample["img1"], sample["img2"]
    q = float(cfg.get("uncertainty", {}).get("primary_q", 0.8))
    var_min = float(cfg.get("model", {}).get("var_min", 0.0))
    var_max = float(cfg.get("model", {}).get("var_max", 10.0))

    # ---- T1 -----------------------------------------------------------------
    flow = model.forward_once(img1, img2)
    check(
        "T1",
        torch.is_tensor(flow) and flow.dim() == 4 and flow.shape[1] == 2,
        "expected a (B, 2, H, W) flow tensor, got %r" % (tuple(flow.shape) if torch.is_tensor(flow) else type(flow)),
    )
    check(
        "T1b",
        tuple(flow.shape[-2:]) == tuple(img1.shape[-2:]),
        "flow %s is not at input resolution %s: _restore_flow is wrong"
        % (tuple(flow.shape[-2:]), tuple(img1.shape[-2:])),
    )
    results["T1"] = {"flow_shape": list(flow.shape)}

    # ---- T2 -----------------------------------------------------------------
    iters = 6
    short = model.forward_trace(img1, img2, iters=iters)
    check(
        "T2",
        len(short["flows"]) == iters + 1,
        "expected %d flows for iters=%d, got %d" % (iters + 1, iters, len(short["flows"])),
    )
    offset = model.detect_list_offset(img1, img2, iters=3)
    check(
        "T2b",
        offset in (0, -1),
        "list offset %r is implausible; the output list layout changed upstream" % (offset,),
    )
    results["T2"] = {"n_flows": len(short["flows"]), "list_offset": offset}

    # ---- T3 -----------------------------------------------------------------
    full = model.forward_trace(img1, img2)
    infos = full["infos"]
    check(
        "T3",
        len(infos) > 0 and infos[-1].dim() == 4 and infos[-1].shape[1] >= 4,
        "info must have >=4 channels (2 logits + 2 log-scales), got %s"
        % (tuple(infos[-1].shape) if infos else None,),
    )
    results["T3"] = {"info_shape": list(infos[-1].shape)}

    # ---- T4 -----------------------------------------------------------------
    scale = model.scale_check(tol=8)
    check(
        "T4",
        bool(scale["ok"]),
        "scale=%s was NOT applied: model ran at %s but %s was expected. "
        "This is the setdefault bug -- the checkpoint is trained at 540x960 and "
        "you are paying ~4x for nothing." % (scale["scale"], scale["model_size"], scale["expected"]),
    )
    results["T4"] = scale

    # ---- T5 -----------------------------------------------------------------
    print("")
    print("%-12s %10s %10s %9s %9s %s" % ("read-out", "first", "last", "range", "drop", "verdict"))
    health: Dict[str, Any] = {}
    for mode in MODES:
        series: List[float] = []
        for info in infos:
            u = pixel_uncertainty(info, var_min=var_min, var_max=var_max, mode=mode)
            series.append(global_uncertainty(u, quantiles=(q,), valid=sample.get("valid"))[q])
        stats = signal_health(series)
        stats["series"] = series
        health[mode] = stats
        print(
            "%-12s %10.4f %10.4f %8.2f%% %8.2f%% %s"
            % (
                mode,
                stats["first"],
                stats["last"],
                100.0 * stats["range_rel"],
                100.0 * stats["drop_rel"],
                "DEAD" if stats["degenerate"] else "alive",
            )
        )
    live = [m for m in PREFERRED if m in health and not health[m]["degenerate"]]
    clip0, clip1 = clamp_pressure(infos[-1])
    print("clamp pressure: clip0 %.1f%% of pixels, clip1 %.1f%%" % (100.0 * clip0, 100.0 * clip1))
    check(
        "T5",
        bool(live),
        "every read-out is degenerate. clip0=%.1f%% clip1=%.1f%%. If clip1 is ~100%% "
        "then var_min=0 pinned component 1 to exactly 1 px; use mode='raw'. If even "
        "'raw' is flat, fall back to the GRU hidden-state probe in ua_stop/hooks.py."
        % (100.0 * clip0, 100.0 * clip1),
    )
    print("T5 ok  : live read-outs, best first -> %s" % ", ".join(live))
    if live[0] != "raw":
        print("T5 note: 'raw' is not the best read-out here; report why in the paper")
    results["T5"] = {
        "health": {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in health.items()},
        "series": {k: v["series"] for k, v in health.items()},
        "live": live,
        "clip0": clip0,
        "clip1": clip1,
        "quantile": q,
    }

    # ---- T6 -----------------------------------------------------------------
    replay = model.forward_trace(img1, img2, iters=3)
    index = 3 - int(offset)
    same = bool(torch.equal(full["flows"][index], replay["flows"][-1]))
    check(
        "T6",
        same,
        "a 3-iteration run does not match cached index %d. Offline replay would "
        "be an approximation, not an equivalence -- do not trust the sweep." % index,
    )
    print("T6 ok  : bit-identical at list index %d -> list_offset = %d" % (index, offset))
    results["T6"] = {"index": index, "bit_identical": same}

    # ---- T7 -----------------------------------------------------------------
    best = live[0]
    u_series = health[best]["series"]
    d_series = [float("inf")]
    for i in range(1, len(full["flows"])):
        diff = full["flows"][i] - full["flows"][i - 1]
        d_series.append(float(torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-12).mean().item()))
    hi = min(int(cfg.get("criterion", {}).get("max_iters", 12)), len(u_series) - 1)
    loose = decide_stop(u_series, d_series, StopConfig(mode="u_only", tau_rel=0.5, patience=1, max_iters=hi))
    fixed = decide_stop(u_series, d_series, StopConfig(mode="fixed", max_iters=hi))
    check(
        "T7",
        loose < hi,
        "even a very loose threshold (tau_rel=0.5 on '%s') never fired; the "
        "criterion cannot save anything on this signal" % best,
    )
    check("T7b", fixed == hi, "mode='fixed' must return the full budget %d, got %d" % (hi, fixed))
    results["T7"] = {"signal": best, "loose_stop": int(loose), "fixed_stop": int(fixed), "max_index": hi}

    path = save_json(results, out_path(cfg.get("out"), "tables", "selftest.json"))
    print("")
    print("SELF-TEST PASSED  (7/7)  ->  %s" % path)
    print("recommended signal: mode='%s' at q=%.2f" % (best, q))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print("")
        print("SELF-TEST FAILED")
        print(str(exc))
        sys.exit(1)
