"""EXPERIMENTAL (level 2): per-tile stopping. Accuracy is exact, cost is MODELLED.

Motivation: a global stop is limited by the hardest region in the frame -- one
fast-moving object keeps the whole image iterating. Splitting the frame into
tiles and stopping each tile independently should recover most of the oracle
gap measured in the sweep.

Honesty warning, and it must survive into the paper:

* The **accuracy** side is exact. ``composite_flow`` really assembles the flow
  from per-tile stop indices, so the reported EPE is what this policy delivers.
* The **cost** side is a model, not a measurement. Real savings need a kernel
  that skips inactive tiles inside the GRU; on dense CUDA the full-resolution
  update runs anyway. ``tile_cost`` reports ``equivalent_iters`` = the mean
  active-tile fraction summed over iterations, i.e. the cost an ideal sparse
  implementation would pay.

So this module is a headroom study. Reporting modelled numbers as measured
speed-ups is the single easiest way to get a paper rejected.

The stopping rule itself is unchanged -- the very same ``decide_stop`` from
``criterion.py`` is applied per tile -- so a comparison against the global
policy isolates the effect of granularity and nothing else.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .criterion import StopConfig, _as_cfg, decide_stop_batch
from .uncertainty import pixel_uncertainty, resolve_mode

DEFAULT_TILE = 64


def tile_reduce(x: torch.Tensor, tile: int = DEFAULT_TILE, how: str = "mean") -> torch.Tensor:
    """Pool a (B, 1, H, W) map down to one value per tile.

    ``how="mean"`` for uncertainty (a tile is as uncertain as its average) and
    ``how="max"`` for the flow delta (a tile is still moving if ANY pixel in it
    is still moving -- the conservative choice).
    """
    if x.dim() != 4:
        raise ValueError("expected (B, C, H, W); got %s" % (tuple(x.shape),))
    tile = int(tile)
    if how == "mean":
        return F.avg_pool2d(x, kernel_size=tile, stride=tile, ceil_mode=True)
    if how == "max":
        return F.max_pool2d(x, kernel_size=tile, stride=tile, ceil_mode=True)
    raise ValueError("how must be 'mean' or 'max', got %r" % (how,))


def tile_decide(
    u_tiles: List[torch.Tensor],
    d_tiles: List[torch.Tensor],
    cfg: Optional[Any] = None,
    halo: int = 1,
    **overrides: Any,
) -> torch.Tensor:
    """Per-tile stop index, (B, 1, Ht, Wt) int64, using the global rule per tile.

    ``halo`` dilates the "keep going" decision by that many tiles (max-pool),
    so a tile bordering an active region keeps refining. Without it, tile seams
    appear exactly where the flow is hardest.
    """
    conf = _as_cfg(cfg, overrides)
    stacked_u = torch.stack(u_tiles, dim=0)  # (K+1, B, 1, Ht, Wt)
    stacked_d = torch.stack(d_tiles, dim=0)
    length, batch = stacked_u.shape[0], stacked_u.shape[1]
    shape = stacked_u.shape[1:]

    flat_u = stacked_u.reshape(length, -1).permute(1, 0).detach().float().cpu().numpy()
    flat_d = stacked_d.reshape(length, -1).permute(1, 0).detach().float().cpu().numpy()
    flat_d[:, 0] = np.inf
    indices = decide_stop_batch(flat_u, flat_d, conf)
    stop = torch.from_numpy(indices.astype(np.int64)).reshape(shape)

    if int(halo) > 0:
        size = 2 * int(halo) + 1
        spread = F.max_pool2d(
            stop.float().unsqueeze(0) if stop.dim() == 3 else stop.float(),
            kernel_size=size,
            stride=1,
            padding=int(halo),
        )
        stop = spread.reshape(shape).round().to(torch.int64)
    return stop.to(u_tiles[0].device)


def composite_flow(
    flows: List[torch.Tensor], stop: torch.Tensor, tile: int = DEFAULT_TILE
) -> torch.Tensor:
    """Assemble the flow each tile would have produced at its own stop index.

    Exact, not approximate: pixels simply take their value from ``flows[i]``
    with ``i`` the stop index of their tile.
    """
    target = flows[-1]
    height, width = target.shape[-2:]
    index_map = F.interpolate(stop.float(), size=(height, width), mode="nearest").long()
    out = torch.zeros_like(target)
    for i, flow in enumerate(flows):
        mask = (index_map == i).to(dtype=target.dtype)
        if float(mask.sum().item()) == 0.0:
            continue
        if flow.shape[-2:] != target.shape[-2:]:
            flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
        out = out + mask * flow
    return out


def tile_cost(stop: torch.Tensor, max_index: Optional[int] = None) -> Dict[str, float]:
    """MODELLED cost of a per-tile schedule (see the module docstring)."""
    idx = stop.detach().float()
    top = int(max_index if max_index is not None else int(idx.max().item()))
    fractions = []
    for i in range(1, top + 1):
        fractions.append(float((idx >= i).float().mean().item()))
    equivalent = float(sum(fractions))
    return {
        "equivalent_iters": equivalent,
        "mean_active_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "max_index": top,
        "mean_stop_index": float(idx.mean().item()),
        "active_per_iter": fractions,
        "note": "modelled cost: assumes an ideal sparse kernel, NOT measured wall-clock",
    }


def run_tile_policy(
    model: Any,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    tile: int = DEFAULT_TILE,
    halo: int = 1,
) -> Dict[str, Any]:
    """One pair, one tile policy: exact accuracy plus modelled cost."""
    from .trace import flow_metrics

    unc_cfg = cfg.get("uncertainty", {})
    mode = resolve_mode(unc_cfg.get("primary_mode", "raw"))
    var_min = float(cfg.get("model", {}).get("var_min", 0.0))
    var_max = float(cfg.get("model", {}).get("var_max", 10.0))
    stop_cfg = StopConfig.from_dict(cfg.get("criterion"))

    out = model.forward_trace(sample["img1"], sample["img2"])
    flows, infos = out["flows"], out["infos"]

    u_tiles: List[torch.Tensor] = []
    d_tiles: List[torch.Tensor] = []
    for i, flow in enumerate(flows):
        info = infos[i] if i < len(infos) else infos[-1]
        u = pixel_uncertainty(info, var_min=var_min, var_max=var_max, mode=mode)
        u_tiles.append(tile_reduce(u, tile, "mean"))
        if i == 0:
            d_tiles.append(torch.full_like(u_tiles[-1], float("inf")))
        else:
            diff = flows[i] - flows[i - 1]
            mag = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-12)
            if mag.shape[-2:] != u.shape[-2:]:
                mag = F.interpolate(mag, size=u.shape[-2:], mode="bilinear", align_corners=False)
            d_tiles.append(tile_reduce(mag, tile, "max"))

    stop = tile_decide(u_tiles, d_tiles, stop_cfg, halo=halo)
    composed = composite_flow(flows, stop, tile)
    cost = tile_cost(stop, max_index=min(int(stop_cfg.max_iters), len(flows) - 1))

    gt = sample.get("flow_gt")
    valid = sample.get("valid")
    result: Dict[str, Any] = {
        "tile": int(tile),
        "halo": int(halo),
        "mode": mode,
        "cost": cost,
        "tile_grid": list(stop.shape[-2:]),
        "tile_metrics": flow_metrics(composed, gt, valid),
        "full_metrics": flow_metrics(flows[-1], gt, valid),
        "ref_vs_full": float(
            torch.sqrt(((composed - flows[-1]) ** 2).sum(dim=1, keepdim=True) + 1e-12)
            .mean()
            .item()
        ),
    }
    print(
        "tile policy: %.2f equivalent iters (%.0f%% tiles active on average), "
        "ref vs full %.4f px  [modelled cost]"
        % (
            cost["equivalent_iters"],
            100.0 * cost["mean_active_fraction"],
            result["ref_vs_full"],
        )
    )
    return result
