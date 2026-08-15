"""The stopping rule: training-free, two-condition, replayable offline.

For one sample we are given two per-iteration series, both indexed by the
*list index* of the model's output list (index 0 is SEA-RAFT's directly
regressed initial flow, i.e. zero GRU iterations):

    U[i]  global uncertainty (a high quantile of the per-pixel map)
    D[i]  mean flow update magnitude ||mu_i - mu_{i-1}|| in px, with D[0] = inf

and we define

    r[i] = |U[i-1] - U[i]| / (|U[i-1]| + EPS)      "uncertainty has saturated"
    d[i] = D[i]                                    "the flow stopped moving"

The default rule stops at the first ``i >= min_iters`` where

    (r[i] < tau_rel) AND (d[i] < tau_delta)

has held for ``patience`` consecutive indices, and never later than
``max_iters``.

Two design notes that matter for the paper:

1. The rule is *global* (one decision per image), and a global stop at index i
   is bit-identical to calling the model with ``iters=i``. That is verified
   empirically in ``scripts/selftest.py`` (T6, ``torch.equal``). So replaying
   thousands of thresholds offline from a cached trace is exact, not an
   approximation.
2. ``mode="d_only"`` is a first-class citizen, not an afterthought. It is the
   make-or-break ablation: if the flow-delta alone matches the full criterion,
   the uncertainty head contributes nothing and the paper's claim collapses.
   Always report it.

Pure numpy: thousands of configurations per second on a CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

EPS = 1e-6

#: ``and``/``or``      both / either condition
#: ``u_only``/``d_only`` single-signal ablations
#: ``abs_only``       absolute-threshold ablation (needs ``tau_abs``)
#: ``fixed``          baseline: always run the full budget
MODES = ("and", "or", "u_only", "d_only", "abs_only", "fixed")


@dataclass
class StopConfig:
    """One point in the stopping-policy space."""

    q: float = 0.8
    tau_rel: float = 0.02
    tau_delta: float = 0.05
    tau_abs: Optional[float] = None
    patience: int = 1
    min_iters: int = 1
    max_iters: int = 12
    mode: str = "and"
    use_abs_delta: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (MODES, self.mode))
        if int(self.patience) < 1:
            raise ValueError("patience must be >= 1, got %r" % (self.patience,))
        if int(self.min_iters) < 0:
            raise ValueError("min_iters must be >= 0, got %r" % (self.min_iters,))
        if int(self.max_iters) < int(self.min_iters):
            raise ValueError(
                "max_iters (%r) must be >= min_iters (%r)"
                % (self.max_iters, self.min_iters)
            )
        if self.mode == "abs_only" and self.tau_abs is None:
            raise ValueError("mode='abs_only' requires tau_abs")

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "StopConfig":
        """Build from a config dict, ignoring unrelated keys."""
        fields = cls().as_dict().keys()
        return cls(**{k: v for k, v in (d or {}).items() if k in fields})

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **kw: Any) -> "StopConfig":
        return replace(self, **kw)

    def label(self) -> str:
        """Compact, sortable identifier used in tables and figure legends."""
        parts = [self.mode, "q%.2f" % float(self.q)]
        if self.mode in ("and", "or", "u_only"):
            parts.append("r%.4g" % float(self.tau_rel))
        if self.mode in ("and", "or", "d_only"):
            parts.append("d%.4g" % float(self.tau_delta))
        if self.tau_abs is not None:
            parts.append("a%.4g" % float(self.tau_abs))
        parts.append("p%d" % int(self.patience))
        return "|".join(parts)


def _as_cfg(cfg: Optional[Any], overrides: Optional[Dict[str, Any]] = None) -> StopConfig:
    if cfg is None:
        base = StopConfig()
    elif isinstance(cfg, StopConfig):
        base = cfg
    elif isinstance(cfg, dict):
        base = StopConfig.from_dict(cfg)
    else:
        raise TypeError("cfg must be StopConfig, dict or None; got %r" % type(cfg))
    return base.replace(**overrides) if overrides else base


def rel_change(U: Sequence[float]) -> np.ndarray:
    """Relative change of ``U`` per index; ``r[0] = inf`` (no predecessor)."""
    u = np.asarray(U, dtype=np.float64).ravel()
    r = np.full(u.shape, np.inf, dtype=np.float64)
    if u.size > 1:
        r[1:] = np.abs(u[:-1] - u[1:]) / (np.abs(u[:-1]) + EPS)
    return r


def delta_signal(D: Sequence[float], use_abs_delta: bool = True) -> np.ndarray:
    """Flow-update series used by the second condition.

    ``use_abs_delta=True``  compare ||d mu|| directly, in pixels (default;
                            interpretable and dataset independent).
    ``use_abs_delta=False`` normalise by the first update, which makes the
                            threshold scale-free but couples it to index 1.
    """
    d = np.asarray(D, dtype=np.float64).ravel().astype(np.float64, copy=True)
    if d.size:
        d[0] = np.inf
    if use_abs_delta:
        return d
    ref = d[1] if d.size > 1 and np.isfinite(d[1]) and d[1] > 0 else 1.0
    out = d / (float(ref) + EPS)
    if out.size:
        out[0] = np.inf
    return out


def decide_stop(
    U: Sequence[float],
    D: Sequence[float],
    cfg: Optional[Any] = None,
    **overrides: Any,
) -> int:
    """Return the *list index* at which to stop for one sample.

    Falls back to ``min(max_iters, len(U) - 1)`` when the criterion never
    fires, which is exactly the full-budget baseline.
    """
    conf = _as_cfg(cfg, overrides)
    u = np.asarray(U, dtype=np.float64).ravel()
    if u.size == 0:
        raise ValueError("U is empty")
    d = delta_signal(D, conf.use_abs_delta)

    hi = min(int(conf.max_iters), u.size - 1)
    lo = min(max(int(conf.min_iters), 0), hi)
    if conf.mode == "fixed":
        return hi

    r = rel_change(u)
    hits = 0
    for i in range(lo, hi + 1):
        ok_u = bool(r[i] < float(conf.tau_rel))
        ok_d = bool(i < d.size and d[i] < float(conf.tau_delta))
        if conf.mode == "and":
            ok = ok_u and ok_d
        elif conf.mode == "or":
            ok = ok_u or ok_d
        elif conf.mode == "u_only":
            ok = ok_u
        elif conf.mode == "d_only":
            ok = ok_d
        else:  # abs_only
            ok = bool(u[i] < float(conf.tau_abs))
        # An absolute floor short-circuits every relative mode: once the map is
        # this confident, more iterations cannot help.
        if conf.tau_abs is not None and conf.mode != "abs_only":
            ok = ok or bool(u[i] < float(conf.tau_abs))
        hits = hits + 1 if ok else 0
        if hits >= int(conf.patience):
            return i
    return hi


def decide_stop_batch(
    U: np.ndarray,
    D: np.ndarray,
    cfg: Optional[Any] = None,
    **overrides: Any,
) -> np.ndarray:
    """Vectorised over samples: ``U``/``D`` are (N, K+1); returns (N,) indices."""
    conf = _as_cfg(cfg, overrides)
    u = np.atleast_2d(np.asarray(U, dtype=np.float64))
    d = np.atleast_2d(np.asarray(D, dtype=np.float64))
    if u.shape != d.shape:
        raise ValueError("U %s and D %s must have the same shape" % (u.shape, d.shape))
    return np.asarray(
        [decide_stop(u[i], d[i], conf) for i in range(u.shape[0])], dtype=np.int64
    )


def iters_of(index: Any, list_offset: int = 0) -> np.ndarray:
    """Convert list indices to GRU iteration counts.

    ``iterations = index + list_offset``. The offset is *measured*, not
    assumed: ``scripts/selftest.py`` (T6) replays a shorter forward pass and
    finds the bit-identical entry. On the checkpoint used here the offset is 0,
    so index 0 means "zero iterations", the free directly-regressed flow.
    """
    idx = np.asarray(index, dtype=np.int64) + int(list_offset)
    return np.clip(idx, 0, None)


def budget(indices: Iterable[int], list_offset: int = 0) -> float:
    """Mean iteration count of a stopping decision, i.e. the cost we report."""
    it = iters_of(np.asarray(list(indices), dtype=np.int64), list_offset)
    return float(np.mean(it)) if it.size else float("nan")
