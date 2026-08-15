#!/usr/bin/env python3
"""Turn the selected threshold into a distribution-free risk certificate (RCPS).

The sweep chose a threshold by looking at the data. This script makes that
selection defensible: calibrate on half the samples, certify with a Hoeffding
upper confidence bound, and report the risk on the held-out half.

    python scripts/run_calibrate.py --alpha 0.05 --delta 0.1 --epsilon 0.05
"""

from __future__ import annotations

import os
import sys

from _cli import build_parser, cfg_from_args, opt_overrides

from ua_stop.conformal import certify
from ua_stop.trace import load_trace
from ua_stop.utils import abs_path, load_json, out_path


def main(argv=None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--trace", default=None, help="path to a trace .npz")
    parser.add_argument("--sweep", default=None, help="path to sweep.json")
    parser.add_argument("--alpha", type=float, default=None, help="risk tolerance")
    parser.add_argument("--delta", type=float, default=None, help="confidence level 1-delta")
    parser.add_argument("--epsilon", type=float, default=None, help="free accuracy budget, in px")
    parser.add_argument("--scale", type=float, default=None, help="loss normaliser, in px")
    parser.add_argument("--split", type=float, default=None, help="calibration fraction")
    args = parser.parse_args(argv)

    cfg = cfg_from_args(
        args,
        opt_overrides(
            [
                ("conformal.alpha", args.alpha),
                ("conformal.delta", args.delta),
                ("conformal.epsilon", args.epsilon),
                ("conformal.scale", args.scale),
                ("conformal.split", args.split),
            ]
        ),
    )

    kind = cfg.get("source", {}).get("kind", "synthetic")
    trace_path = args.trace or out_path(cfg.get("out"), "traces", "trace_%s.npz" % kind)
    sweep_path = args.sweep or out_path(cfg.get("out"), "tables", "sweep.json")
    for path, hint in ((trace_path, "run_trace.py"), (sweep_path, "run_sweep.py")):
        if not os.path.isfile(abs_path(path)):
            raise SystemExit("missing %s\nrun scripts/%s first" % (path, hint))

    trace = load_trace(trace_path)
    sweep = load_json(sweep_path)
    if not sweep.get("pareto"):
        raise SystemExit("sweep.json has an empty Pareto front; nothing to certify")

    result = certify(trace, sweep, cfg, out=cfg.get("out"))

    print("")
    print("how to report this:")
    print("  * quote alpha, delta and epsilon in the caption; the number is")
    print("    meaningless without them")
    print("  * with ~40 calibration pairs the Hoeffding bound is loose; say so,")
    print("    and mention Bentkus / Waudby-Smith--Ramdas as tighter alternatives")
    if result.get("held_out", {}).get("violated"):
        print("  * WARNING: held-out risk exceeded alpha -- do not claim the guarantee")
    return 0


if __name__ == "__main__":
    sys.exit(main())
