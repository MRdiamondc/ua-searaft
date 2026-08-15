"""Read-outs of SEA-RAFT's Mixture-of-Laplace (MoL) uncertainty head.

Background
----------
SEA-RAFT predicts, at every refinement iteration, a four-channel ``info`` map:

    info[:, 0:2] -> mixing logits of the two Laplace components
    info[:, 2:4] -> the raw (unbounded) log-scale of each component

The released evaluation code turns that into a scalar heat-map like this
(verbatim from upstream ``custom.py``)::

    raw_b  = info[:, 2:]
    log_b  = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0,            max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)

Why the released read-out cannot be used as a stopping signal
------------------------------------------------------------
This is the central negative finding of the project, and it is reproducible
with one line of arithmetic rather than an experiment.

Component 0 is clamped to ``[0, var_max]`` and component 1 to
``[var_min, 0]``. The shipped evaluation configs use ``var_min = 0``.
Therefore component 1 is clamped to ``[0, 0]``: it is identically zero, so
``b_1 = exp(0) = 1`` px for every pixel, at every iteration, forever.
Component 0 is floored at 0 from below, so wherever the network is *confident*
(raw log-scale below 0, i.e. a sub-pixel scale) it is pinned to exactly 0 too.

The consequence: on a confident pair both channels are pinned and the scalar
read-out collapses to the constant 1.0 px. Our first run measured exactly
that::

    u median = 1.000 px  (min 1, max 1)
    U@0.8 per iteration = [1.0006 1. 1. 1. 1. 1. 1.]

A dead signal -- and, worse, one that a naive self-test still reports as
"decreasing", because the tiny 1.0006 at index 0 makes the series formally
non-increasing. Any early-exit rule built on it would stop at the first
iteration for entirely the wrong reason.

The clamp is not an upstream bug: it is a read-out-time stabiliser that keeps
the visualisation in a sane range. But Figure 6 of the paper ("more iterations
produce lower variance in the Mixture of Laplace") can only have been plotted
from PRE-clamp values, because post-clamp values cannot decrease once pinned.

So this module treats the raw, pre-clamp parameters as first-class citizens and
offers five read-outs, all computed from one and the same forward pass:

    raw          sum_k w_k * exp(raw_k)          default; linear px, unclamped
    geo          exp(sum_k w_k * raw_k)          log-space (geometric) mean
    clamped_lin  sum_k w_k * exp(clamp(raw_k))   upstream semantics, linear
    clamped_log  exp(sum_k w_k * clamp(raw_k))   upstream semantics, log-space
    alpha        w[:, 0:1]                       mixing weight only, unitless

``clamp_pressure()`` quantifies the damage as the fraction of pixels sitting
exactly on a clamp boundary. Report it in the paper next to the ablation:
it converts "the signal was dead" into a number.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

#: Quantiles cached for every sample and iteration, so the sweep can pick one.
QUANTILES: Tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95)

#: All supported read-outs, cheapest name first.
MODES: Tuple[str, ...] = ("clamped_lin", "clamped_log", "raw", "geo", "alpha")

#: Backwards-compatible short names used by the first Colab notebook.
_ALIASES = {"lin": "clamped_lin", "log": "clamped_log"}

#: exp() guard, in log space. exp(20) ~ 4.9e8 px is already meaningless.
EXP_GUARD = 20.0

#: A read-out whose relative range over the iteration budget is below this is
#: reported as degenerate (dead) by :func:`signal_health`.
DEGENERATE = 5e-3


def resolve_mode(mode: str) -> str:
    """Canonicalise a read-out name, accepting the legacy ``lin``/``log``."""
    name = _ALIASES.get(str(mode), str(mode))
    if name not in MODES:
        raise ValueError("mode must be one of %s, got %r" % (MODES, mode))
    return name


def mol_params(
    info: torch.Tensor,
    var_min: float = 0.0,
    var_max: float = 10.0,
    clamp: bool = True,
) -> Dict[str, torch.Tensor]:
    """Split ``info`` into MoL parameters.

    Returns a dict with
        ``weight`` (B, 2, H, W) softmax mixing weights,
        ``raw``    (B, 2, H, W) unbounded log-scales as predicted,
        ``log_b``  (B, 2, H, W) log-scales after the upstream clamp (or a copy
                   of ``raw`` when ``clamp=False``).
    """
    if info.dim() != 4 or info.shape[1] < 4:
        raise ValueError(
            "info must be (B, >=4, H, W); got %s" % (tuple(info.shape),)
        )
    logits = info[:, :2]
    raw = info[:, 2:4]
    weight = logits.softmax(dim=1)
    if clamp:
        log_b = torch.zeros_like(raw)
        log_b[:, 0] = torch.clamp(raw[:, 0], min=0.0, max=float(var_max))
        log_b[:, 1] = torch.clamp(raw[:, 1], min=float(var_min), max=0.0)
    else:
        log_b = raw.clone()
    return {"weight": weight, "raw": raw, "log_b": log_b}


def clamp_pressure(info: torch.Tensor) -> Tuple[float, float]:
    """Fraction of pixels pinned by each side of the upstream clamp.

    ``clip0`` -- component 0 at (or below) its lower bound 0, i.e. the network
    predicted a sub-pixel scale and the clamp threw that information away.
    ``clip1`` -- component 1 at (or above) its upper bound 0, which with
    ``var_min = 0`` means the component is pinned to exactly 1 px.

    Both numbers close to 1.0 is the signature of the dead signal.
    """
    raw = info[:, 2:4]
    clip0 = float((raw[:, 0] <= 0).float().mean().item())
    clip1 = float((raw[:, 1] >= 0).float().mean().item())
    return clip0, clip1


def pixel_uncertainty(
    info: torch.Tensor,
    var_min: float = 0.0,
    var_max: float = 10.0,
    mode: str = "raw",
) -> torch.Tensor:
    """Per-pixel scalar uncertainty map, (B, 1, H, W).

    Units are pixels for every mode except ``alpha`` (unitless mixing weight).
    """
    mode = resolve_mode(mode)
    params = mol_params(
        info, var_min=var_min, var_max=var_max, clamp=mode.startswith("clamped")
    )
    weight, raw, log_b = params["weight"], params["raw"], params["log_b"]

    if mode == "alpha":
        return weight[:, 0:1].clone()
    if mode == "clamped_lin":
        # sum_k w_k * b_k with b_0 >= 1 and (for var_min = 0) b_1 == 1:
        # this is the read-out that floors at 1 px.
        return (weight * log_b.exp()).sum(dim=1, keepdim=True)
    if mode == "clamped_log":
        return (weight * log_b).sum(dim=1, keepdim=True).exp()

    guarded = raw.clamp(min=-EXP_GUARD, max=EXP_GUARD)
    if mode == "raw":
        return (weight * guarded.exp()).sum(dim=1, keepdim=True)
    # geo
    return (weight * guarded).sum(dim=1, keepdim=True).exp()


def norm_valid(valid: Optional[torch.Tensor], like: torch.Tensor) -> Optional[torch.Tensor]:
    """Broadcast a validity mask to ``like``'s (B, 1, H, W) grid."""
    if valid is None:
        return None
    v = valid
    if v.dim() == 2:
        v = v[None, None]
    elif v.dim() == 3:
        v = v[:, None] if v.shape[0] == like.shape[0] else v[None]
    v = v.to(dtype=like.dtype, device=like.device)
    if v.shape[-2:] != like.shape[-2:]:
        v = torch.nn.functional.interpolate(v, size=like.shape[-2:], mode="nearest")
    return v


def global_uncertainty(
    u: torch.Tensor,
    quantiles: Sequence[float] = QUANTILES,
    valid: Optional[torch.Tensor] = None,
    max_elems: int = 4_000_000,
) -> Dict[float, float]:
    """Reduce a per-pixel map to global quantiles.

    A high quantile (0.8-0.9) is the useful summary: the mean is dominated by
    the large confident background, while the tail is what still moves between
    iterations.

    ``torch.quantile`` refuses inputs beyond ~2**24 elements, so the flattened
    map is strided down deterministically when needed (no RNG, so the trace
    stays reproducible).
    """
    x = u.detach().reshape(-1).float()
    if valid is not None:
        mask = norm_valid(valid, u)
        if mask is not None:
            keep = mask.reshape(-1) > 0.5
            if bool(keep.any()):
                x = x[keep]
    if x.numel() == 0:
        return {float(q): float("nan") for q in quantiles}
    if x.numel() > int(max_elems):
        step = int(np.ceil(x.numel() / float(max_elems)))
        x = x[::step]
    qs = torch.tensor([float(q) for q in quantiles], device=x.device, dtype=x.dtype)
    vals = torch.quantile(x, qs)
    return {float(q): float(v) for q, v in zip(quantiles, vals.tolist())}


def signal_health(series: Iterable[float]) -> Dict[str, float]:
    """Is this read-out alive over the iteration budget?

    ``range_rel``     (max - min) / |first|, the relative dynamic range
    ``monotone_frac`` fraction of steps that do not increase
    ``drop_rel``      (first - last) / |first|
    ``degenerate``    True when ``range_rel < DEGENERATE``

    The first Colab self-test only printed a warning here; the shipped
    ``scripts/selftest.py`` asserts on ``degenerate`` instead, which is the
    difference between finding the clamp bug in minutes and in days.
    """
    s = np.asarray(list(series), dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return {
            "first": float("nan"),
            "last": float("nan"),
            "range_rel": 0.0,
            "drop_rel": 0.0,
            "monotone_frac": 0.0,
            "degenerate": True,
            "n": float(s.size),
        }
    first, last = float(s[0]), float(s[-1])
    scale = max(abs(first), 1e-12)
    span = float(s.max() - s.min())
    diffs = np.diff(s)
    return {
        "first": first,
        "last": last,
        "range_rel": span / scale,
        "drop_rel": (first - last) / scale,
        "monotone_frac": float(np.mean(diffs <= 0.0)),
        "degenerate": bool(span / scale < DEGENERATE),
        "n": float(s.size),
    }
