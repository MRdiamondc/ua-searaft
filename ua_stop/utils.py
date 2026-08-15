"""Shared helpers: config loading, output paths, seeding, JSON I/O, timing.

Nothing in this module imports torch at import time, so the analysis half of
the pipeline (and the whole unit-test suite) runs on a CPU-only machine.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from typing import Any, Dict, Optional

import numpy as np

# Keep in sync with ua_stop/__init__.py (kept local to avoid a circular import).
VERSION = "0.1.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs")
KINDS = ("traces", "tables", "figs", "logs")


def repo_path(*parts: str) -> str:
    """Absolute path inside the repository, independent of the current CWD."""
    return os.path.join(REPO_ROOT, *parts)


def abs_path(path: str) -> str:
    """Resolve a possibly relative config path against the repository root."""
    if not path:
        return path
    p = os.path.expanduser(str(path))
    return p if os.path.isabs(p) else repo_path(p)


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def out_path(out: Optional[str], kind: str, name: str) -> str:
    """Build ``<out>/<kind>/<name>`` and create the directory.

    ``kind`` is restricted so the output tree stays predictable:
    ``traces`` (npz), ``tables`` (json/csv), ``figs`` (pdf), ``logs`` (txt).
    """
    if kind not in KINDS:
        raise ValueError("kind must be one of %s, got %r" % (KINDS, kind))
    root = abs_path(out or DEFAULT_OUT)
    ensure_dir(os.path.join(root, kind))
    return os.path.join(root, kind, name)


def set_seed(seed: int = 1234) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def pick_device(preferred: str = "cuda") -> str:
    try:
        import torch

        if str(preferred).startswith("cuda") and torch.cuda.is_available():
            return str(preferred)
    except Exception:
        pass
    return "cpu"


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)


def save_json(obj: Any, path: str) -> str:
    """Write JSON atomically, tolerating numpy scalars and dataclasses."""
    path = abs_path(path)
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def load_json(path: str) -> Dict[str, Any]:
    with open(abs_path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def deep_update(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursive dict merge; ``extra`` wins. Never mutates the inputs."""
    out = dict(base or {})
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a JSON config, following an optional ``"extends"`` chain.

    ``configs/kitti15.json`` only carries its differences from
    ``configs/default.json``; the chain is resolved here so every script sees a
    single fully populated dictionary.
    """
    path = abs_path(path or repo_path("configs", "default.json"))
    seen = []

    def _load(p: str) -> Dict[str, Any]:
        p = abs_path(p)
        if p in seen:
            raise ValueError("circular 'extends' chain at %s" % p)
        seen.append(p)
        raw = dict(load_json(p))
        parent = raw.pop("extends", None)
        return deep_update(_load(parent), raw) if parent else raw

    cfg = _load(path)
    cfg = deep_update(cfg, overrides or {})
    cfg.setdefault("out", DEFAULT_OUT)
    cfg["_config_path"] = path
    return cfg


def fmt_ms(x: float) -> str:
    return "%8.2f ms" % float(x)


class Timer:
    """Wall-clock context manager: ``with Timer() as t: ...`` then ``t.ms``."""

    ms = float("nan")

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.ms = (time.perf_counter() - self._t0) * 1e3
        return False


def git_hash(root: Optional[str] = None) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root or REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip() or None
    except Exception:
        pass
    return None


def provenance(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Environment stamp embedded in every saved JSON, for the paper's appendix."""
    info: Dict[str, Any] = {
        "ua_stop": VERSION,
        "git": git_hash(),
        "upstream_git": git_hash(repo_path("third_party", "SEA-RAFT")),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        info["torch"] = None
    return deep_update(info, extra or {})


def describe(cfg: Dict[str, Any]) -> str:
    """One-screen summary of a config; every script prints this first."""
    model = cfg.get("model", {})
    source = cfg.get("source", {})
    unc = cfg.get("uncertainty", {})
    return "\n".join(
        [
            "config : %s" % cfg.get("_config_path"),
            "model  : cfg=%s" % model.get("cfg"),
            "         url=%s path=%s" % (model.get("url"), model.get("path")),
            "         scale=%s iters=%s var=[%s, %s] pad=%s amp=%s"
            % (
                model.get("scale"),
                model.get("iters"),
                model.get("var_min"),
                model.get("var_max"),
                model.get("pad_mode"),
                model.get("amp"),
            ),
            "source : kind=%s n=%s size=%s"
            % (source.get("kind"), source.get("n"), source.get("size")),
            "signal : modes=%s primary=%s@%s"
            % (unc.get("modes"), unc.get("primary_mode"), unc.get("primary_q")),
            "out    : %s" % abs_path(cfg.get("out") or DEFAULT_OUT),
        ]
    )


def ukey(mode: str, q: float) -> str:
    """Canonical trace key for one (read-out, quantile) pair.

    Defined here, in the torch-free module, so that both the GPU tracer and the
    CPU-only sweep build byte-identical key names.
    """
    return "U:%s@%.2f" % (mode, float(q))


def umean_key(mode: str) -> str:
    """Canonical trace key for the mean of a per-pixel read-out."""
    return "Umean:%s" % mode
