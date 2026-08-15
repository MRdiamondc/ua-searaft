#!/usr/bin/env python3
"""Regenerate every figure from the saved artefacts. No GPU, no re-runs.

    python scripts/make_figures.py

Writes outputs/figs/fig1_pareto.pdf .. fig4_latency.pdf. Artefacts that are
missing are skipped with a message rather than crashing, so this is safe to run
at any point in the pipeline.
"""

from __future__ import annotations

import sys

from _cli import build_parser, cfg_from_args

from ua_stop.plots import make_all


def main(argv=None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--trace", default=None, help="path to a trace .npz for fig3")
    args = parser.parse_args(argv)

    cfg = cfg_from_args(args)
    made = make_all(cfg, out=cfg.get("out"), trace_path=args.trace)

    print("")
    print("%d figure(s) written" % len(made))
    if made:
        print("all labels are in English and fonts are embedded as TrueType,")
        print("so the PDFs drop straight into a LaTeX or Word manuscript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
