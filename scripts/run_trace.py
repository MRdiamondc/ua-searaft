#!/usr/bin/env python3
"""Trace a dataset: ONE full-budget forward pass per pair, everything cached.

This is the only GPU-expensive step. Because a global stop at index i is
bit-identical to running the model with ``iters=i`` (verified by selftest T6),
the cached per-iteration statistics are enough to replay every threshold
afterwards on a CPU -- exactly, not approximately.

    python scripts/run_trace.py --source synthetic --n 120
    python scripts/run_trace.py --config configs/kitti15.json --limit 40
"""

from __future__ import annotations

import sys

from _cli import build_parser, cfg_from_args, opt_overrides

from ua_stop.data import build_source, first_sample
from ua_stop.model_wrapper import build_model
from ua_stop.trace import trace_dataset


def main(argv=None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--source", default=None, help="synthetic | folder | kitti15")
    parser.add_argument("--n", type=int, default=None, help="number of synthetic pairs")
    parser.add_argument("--limit", type=int, default=None, help="stop after N samples")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="ignore any .partial.pkl")
    args = parser.parse_args(argv)

    cfg = cfg_from_args(
        args,
        opt_overrides(
            [
                ("source.kind", args.source),
                ("source.n", args.n),
                ("model.device", args.device),
                ("model.iters", args.iters),
                ("model.scale", args.scale),
            ]
        ),
    )

    model = build_model(cfg)
    print(model.describe())
    source = build_source(cfg)
    print(source.describe())

    # Measure the list offset on a real pair before spending an hour tracing.
    sample = first_sample(cfg)
    offset = model.detect_list_offset(sample["img1"], sample["img2"])
    print("list_offset = %s (index 0 == %s GRU iterations)" % (offset, model.list_offset))

    path = trace_dataset(
        model,
        source,
        cfg,
        out=cfg.get("out"),
        resume=not args.no_resume,
        limit=args.limit,
    )
    print("")
    print("next: python scripts/run_sweep.py --trace %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
