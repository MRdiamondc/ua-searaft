#!/usr/bin/env python3
"""Measure the cost model T(n) = a + b n, and the criterion's own overhead.

This script decides whether the whole idea is worth pursuing on your hardware:
if iterations are only a small share of total latency, no stopping rule can
save much. Run it FIRST and report the share honestly.

    python scripts/run_latency.py --set latency.reps=25
"""

from __future__ import annotations

import sys

from _cli import build_parser, cfg_from_args, opt_overrides

from ua_stop.data import first_sample
from ua_stop.latency import measure
from ua_stop.model_wrapper import build_model


def main(argv=None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = cfg_from_args(
        args,
        opt_overrides(
            [
                ("model.device", args.device),
                ("model.iters", args.iters),
                ("model.scale", args.scale),
                ("latency.reps", args.reps),
                ("latency.warmup", args.warmup),
            ]
        ),
    )

    model = build_model(cfg)
    print(model.describe())
    check = model.scale_check()
    print("scale check: %s" % check)

    sample = first_sample(cfg)
    print("timing on: %s" % sample.get("name"))
    result = measure(model, sample, cfg, out=cfg.get("out"))

    fit = result["fit"]
    ceiling = 100.0 * result["saving_curve"][0]["saving"]
    print("")
    print("interpretation:")
    print(
        "  * iterations are %.0f%% of T(%d), so %.1f%% is the ABSOLUTE ceiling "
        "for any stopping rule here" % (100.0 * fit["share_at_n_max"], fit["n_max"], ceiling)
    )
    print(
        "  * a realistic target is 20-30%% wall-clock at under 1%% accuracy loss; "
        "anything above %.1f%% would be a bug in the measurement" % ceiling
    )
    if not result["overhead"].get("cheap", True):
        print("  * WARNING: the criterion itself costs a large share of one iteration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
