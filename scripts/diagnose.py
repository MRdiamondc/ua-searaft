#!/usr/bin/env python3
"""Is the uncertainty signal alive, and if not, exactly which clamp killed it?

Prints one row per refinement iteration with the raw (pre-clamp) log-scales,
the mixing weight, the clamp pressure, and the same global read-out computed
both pre- and post-clamp. Everything needed to diagnose the dead signal is in
that table -- no plotting, no guessing.

    python scripts/diagnose.py

Run this whenever the numbers look strange. It is 30 seconds and one forward
pass, and it is how the ``u = 1.000 px`` constant was traced back to
``var_min = 0`` rather than to a bug in our own code.
"""

from __future__ import annotations

import sys

import torch

from _cli import build_parser, cfg_from_args, opt_overrides

from ua_stop.data import first_sample
from ua_stop.model_wrapper import build_model
from ua_stop.uncertainty import (
    clamp_pressure,
    global_uncertainty,
    mol_params,
    pixel_uncertainty,
    signal_health,
)
from ua_stop.utils import out_path, provenance, save_json


def main(argv=None) -> int:
    parser = build_parser("Per-iteration diagnosis of the MoL uncertainty head.")
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

    var_min = float(cfg.get("model", {}).get("var_min", 0.0))
    var_max = float(cfg.get("model", {}).get("var_max", 10.0))
    q = float(cfg.get("uncertainty", {}).get("primary_q", 0.8))

    model = build_model(cfg)
    print(model.describe())
    sample = first_sample(cfg)
    print("pair   : %s  shape %s" % (sample.get("name"), tuple(sample["img1"].shape)))
    out = model.forward_trace(sample["img1"], sample["img2"])
    print("scale  : %s" % model.scale_check())
    print("var    : [%s, %s]   <- var_min=0 means component 1 is clamped to [0, 0]" % (var_min, var_max))
    print("")

    header = (
        "%3s | %21s | %21s | %6s | %6s | %6s | %9s | %9s"
        % ("i", "raw0 min/med/max", "raw1 min/med/max", "w0 med", "clip0", "clip1", "u_raw@%.1f" % q, "u_clmp@%.1f" % q)
    )
    print(header)
    print("-" * len(header))

    rows = []
    raw_series, clamped_series = [], []
    for i, info in enumerate(out["infos"]):
        params = mol_params(info, var_min=var_min, var_max=var_max, clamp=True)
        raw, weight = params["raw"], params["weight"]
        clip0, clip1 = clamp_pressure(info)
        u_raw = global_uncertainty(
            pixel_uncertainty(info, var_min, var_max, "raw"), (q,), sample.get("valid")
        )[q]
        u_clamped = global_uncertainty(
            pixel_uncertainty(info, var_min, var_max, "clamped_lin"), (q,), sample.get("valid")
        )[q]
        stats = {
            "i": i,
            "raw0": [float(raw[:, 0].min()), float(raw[:, 0].median()), float(raw[:, 0].max())],
            "raw1": [float(raw[:, 1].min()), float(raw[:, 1].median()), float(raw[:, 1].max())],
            "w0_med": float(weight[:, 0].median()),
            "clip0": clip0,
            "clip1": clip1,
            "u_raw": u_raw,
            "u_clamped": u_clamped,
        }
        rows.append(stats)
        raw_series.append(u_raw)
        clamped_series.append(u_clamped)
        print(
            "%3d | %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f | %6.3f | %5.1f%% | %5.1f%% | %9.4f | %9.4f"
            % (
                i,
                stats["raw0"][0], stats["raw0"][1], stats["raw0"][2],
                stats["raw1"][0], stats["raw1"][1], stats["raw1"][2],
                stats["w0_med"],
                100.0 * clip0,
                100.0 * clip1,
                u_raw,
                u_clamped,
            )
        )

    raw_health = signal_health(raw_series)
    clamped_health = signal_health(clamped_series)
    print("")
    print("raw     : range %.2f%%  drop %.2f%%  monotone %.0f%%  -> %s" % (
        100.0 * raw_health["range_rel"], 100.0 * raw_health["drop_rel"],
        100.0 * raw_health["monotone_frac"], "DEAD" if raw_health["degenerate"] else "alive",
    ))
    print("clamped : range %.2f%%  drop %.2f%%  monotone %.0f%%  -> %s" % (
        100.0 * clamped_health["range_rel"], 100.0 * clamped_health["drop_rel"],
        100.0 * clamped_health["monotone_frac"], "DEAD" if clamped_health["degenerate"] else "alive",
    ))

    print("")
    print("how to read this table:")
    print("  * clip1 near 100%: raw1 is above 0 almost everywhere, so with var_min=0")
    print("    component 1 is pinned to exp(0)=1 px. That is why the released")
    print("    read-out reports a constant 1.000 px. Use mode='raw' (pre-clamp).")
    print("  * u_raw must DECREASE with i. If it does, the innovation has a signal;")
    print("    if it is flat while the flow delta still shrinks, the uncertainty head")
    print("    adds nothing and 'd_only' is the honest headline.")
    print("  * raw0 median far below 0 means the network is confident and clip0 is")
    print("    throwing that information away: exactly the sub-pixel range we need.")

    path = save_json(
        {
            "rows": rows,
            "raw_health": raw_health,
            "clamped_health": clamped_health,
            "quantile": q,
            "var_min": var_min,
            "var_max": var_max,
            "scale_check": model.scale_check(),
            "provenance": provenance(),
        },
        out_path(cfg.get("out"), "tables", "diagnose.json"),
    )
    print("")
    print("saved -> %s" % path)
    if raw_health["degenerate"]:
        print("VERDICT: even the pre-clamp read-out is flat on this pair.")
        print("         Try a harder pair (KITTI), then ua_stop/hooks.py HiddenStateProbe.")
        return 2
    print("VERDICT: pre-clamp read-out is alive; proceed to run_trace.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
