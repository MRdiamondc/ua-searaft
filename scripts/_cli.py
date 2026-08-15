"""Shared CLI plumbing for the scripts/ entry points.

Every script takes the same three arguments -- ``--config``, ``--set`` and
``--out`` -- so any config value can be overridden from the command line
without editing JSON:

    python scripts/run_trace.py --set model.scale=-1 --set source.n=120

That matters for reproducibility: the exact command line is enough to
reconstruct a run, and every script prints the resolved config before doing
any work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ua_stop.utils import deep_update, describe, load_config  # noqa: E402


def coerce(text: str) -> Any:
    """Parse a CLI value as JSON, falling back to the raw string."""
    try:
        return json.loads(text)
    except Exception:
        return text


def nest(dotted: str, value: Any) -> Dict[str, Any]:
    """``"model.scale", -1`` -> ``{"model": {"scale": -1}}``."""
    out: Dict[str, Any] = {}
    cursor = out
    parts = dotted.split(".")
    for key in parts[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value
    return out


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to a JSON config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (e.g. model.iters=12)",
    )
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--quiet", action="store_true", help="do not print the config")
    return parser


def cfg_from_args(args: argparse.Namespace, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve config file + ``--set`` overrides + script-specific extras."""
    overrides: Dict[str, Any] = {}
    for item in args.overrides or []:
        if "=" not in item:
            raise SystemExit("--set expects KEY=VALUE, got %r" % item)
        key, _, value = item.partition("=")
        overrides = deep_update(overrides, nest(key.strip(), coerce(value.strip())))
    if extra:
        overrides = deep_update(overrides, extra)
    if getattr(args, "out", None):
        overrides["out"] = args.out
    cfg = load_config(args.config, overrides)
    if not getattr(args, "quiet", False):
        print(describe(cfg))
        print("-" * 62)
    return cfg


def opt_overrides(pairs: List[Any]) -> Dict[str, Any]:
    """Build overrides from (dotted_key, value) pairs, skipping ``None``."""
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if value is not None:
            out = deep_update(out, nest(key, value))
    return out
