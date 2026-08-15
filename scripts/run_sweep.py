#!/usr/bin/env python3
"""The experiment: replay every threshold offline against a cached trace.

CPU only, seconds to minutes. Writes tables/sweep.json and tables/sweep_all.csv
and prints the HEADLINE line, which is the sentence that goes in the abstract:
mean iterations, accuracy change, and the comparison against a COST-MATCHED
fixed budget (not just against the full budget).

    python scripts/run_sweep.py
    python scripts/run_sweep.py --trace outputs/traces/trace_kitti15.npz
"""

from __future__ import annotations

import csv
import os
import sys

from _cli import build_parser, cfg_from_args, opt_overrides

from ua_stop.criterion import StopConfig
from ua_stop.sweep import evaluate, run_sweep
from ua_stop.trace import load_trace
from ua_stop.utils import abs_path, out_path, save_json


def main(argv=None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--trace", default=None, help="path to a trace .npz")
    parser.add_argument("--slack", type=float, default=None, help="accuracy slack, e.g. 0.01")
    parser.add_argument("--signal", default=None, help="restrict to one U:<mode>@<q> key")
    args = parser.parse_args(argv)

    cfg = cfg_from_args(args, opt_overrides([("sweep.budget_slack", args.slack)]))

    trace_path = args.trace or out_path(
        cfg.get("out"), "traces", "trace_%s.npz" % cfg.get("source", {}).get("kind", "synthetic")
    )
    if not os.path.isfile(abs_path(trace_path)):
        raise SystemExit(
            "trace not found: %s\nrun scripts/run_trace.py first" % trace_path
        )
    trace = load_trace(trace_path)
    print(
        "trace  : %s  (%d samples, %d indices, metric source: %s)"
        % (
            trace_path,
            len(trace.get("names", [])),
            int(trace["D"].shape[1]),
            "ground truth" if trace.get("has_gt") else "distance to full budget",
        )
    )

    result = run_sweep(
        trace, cfg, out=cfg.get("out"), signals=[args.signal] if args.signal else None
    )

    # Attach the stop histogram of the selected configuration so fig2 can be
    # drawn from sweep.json alone, without touching the trace again.
    head = result.get("headline", {})
    if head.get("found"):
        base = StopConfig.from_dict(cfg.get("criterion"))
        row = next(r for r in result["all_rows"] if r["label"] == head["label"])
        point, _, _ = evaluate(
            trace,
            head["signal"],
            base.replace(
                mode=row["mode"],
                tau_rel=row["tau_rel"],
                tau_delta=row["tau_delta"],
                patience=int(row["patience"]),
            ),
        )
        result["headline"]["stop_hist"] = point.stop_hist
        result["list_offset"] = int(trace.get("list_offset", 0))
        save_json(result, out_path(cfg.get("out"), "tables", "sweep.json"))

    rows = result.get("all_rows", [])
    if rows:
        csv_path = abs_path(out_path(cfg.get("out"), "tables", "sweep_all.csv"))
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("saved -> %s (%d configurations)" % (csv_path, len(rows)))

    print("")
    print("read this before believing the headline:")
    print("  * 'vs matched fixed' is the real baseline; beating the full budget is trivial")
    print("  * check the d_only rows in sweep_all.csv: if they match 'and', the")
    print("    uncertainty head adds nothing and the contribution is the flow delta")
    print("  * the oracle gap is the headroom a per-tile policy could still recover")
    return 0


if __name__ == "__main__":
    sys.exit(main())
