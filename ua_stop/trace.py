"""One full-budget GPU pass per sample; everything else is replayed offline.

Layout of the saved ``.npz`` -- every metric array has shape ``(N, K+1)``,
indexed by sample and by list index:

    U:<mode>@<q>   global uncertainty quantile, one key per (read-out, quantile)
    Umean:<mode>   mean of the per-pixel read-out
    D              mean ||mu_i - mu_{i-1}|| in px, with D[:, 0] = inf
    clip0, clip1   clamp pressure: fraction of pixels on each clamp bound
    epe, px1, fl   accuracy against ground truth (NaN when unavailable)
    ref            EPE against the full-budget flow: the no-GT surrogate

plus ``names``, ``list_offset``, ``modes``, ``quantiles``, ``source``, ``has_gt``.

Recording all five read-outs at all five quantiles in the same pass costs
almost nothing -- they are reductions of a tensor already in memory -- and buys
a complete offline study. No second GPU pass is ever needed to answer "what if
we had used the geometric read-out at q = 0.9?".
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .uncertainty import (
    MODES,
    QUANTILES,
    clamp_pressure,
    global_uncertainty,
    norm_valid,
    pixel_uncertainty,
    resolve_mode,
)
from .utils import abs_path, out_path, provenance, save_json, ukey, umean_key

PARTIAL_EVERY = 25


def flow_delta(flows: List[torch.Tensor]) -> np.ndarray:
    """Mean per-pixel flow update magnitude per list index; entry 0 is ``inf``."""
    out = np.full(len(flows), np.inf, dtype=np.float64)
    for index in range(1, len(flows)):
        diff = flows[index] - flows[index - 1]
        mag = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-12)
        out[index] = float(mag.mean().item())
    return out


def flow_metrics(
    pred: torch.Tensor,
    gt: Optional[torch.Tensor],
    valid: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """EPE, 1px error rate and KITTI Fl-all, all in percent except EPE.

    Fl follows the KITTI definition: an outlier is a pixel whose EPE exceeds
    3 px **and** 5% of the ground-truth magnitude.
    """
    if gt is None:
        return {"epe": float("nan"), "px1": float("nan"), "fl": float("nan")}
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    error = pred - gt
    epe_map = torch.sqrt((error ** 2).sum(dim=1, keepdim=True) + 1e-12)
    magnitude = torch.sqrt((gt ** 2).sum(dim=1, keepdim=True) + 1e-12)
    mask = norm_valid(valid, epe_map)
    if mask is None:
        mask = torch.ones_like(epe_map)
    denom = mask.sum().clamp(min=1.0)
    outlier = ((epe_map > 3.0) & (epe_map > 0.05 * magnitude)).float()
    return {
        "epe": float(((epe_map * mask).sum() / denom).item()),
        "px1": float((((epe_map > 1.0).float() * mask).sum() / denom).item()) * 100.0,
        "fl": float(((outlier * mask).sum() / denom).item()) * 100.0,
    }


def ref_series(flows: List[torch.Tensor]) -> np.ndarray:
    """EPE of every intermediate flow against the full-budget flow.

    Without ground truth this is the metric the sweep optimises: it measures
    "how much do I differ from what I would have got by paying full price",
    which is exactly the quantity an early-exit rule is allowed to trade away.
    """
    final = flows[-1]
    out = np.zeros(len(flows), dtype=np.float64)
    for index, flow in enumerate(flows):
        diff = flow - final
        mag = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-12)
        out[index] = float(mag.mean().item())
    return out


def trace_one(model: Any, sample: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """All per-iteration series for a single pair, from ONE forward pass."""
    unc_cfg = cfg.get("uncertainty", {})
    modes = [resolve_mode(m) for m in unc_cfg.get("modes", MODES)]
    quantiles = [float(q) for q in unc_cfg.get("quantiles", QUANTILES)]
    var_min = float(cfg.get("model", {}).get("var_min", 0.0))
    var_max = float(cfg.get("model", {}).get("var_max", 10.0))

    out = model.forward_trace(
        sample["img1"], sample["img2"], iters=cfg.get("model", {}).get("iters")
    )
    flows, infos = out["flows"], out["infos"]
    length = len(flows)

    row: Dict[str, np.ndarray] = {}
    for mode in modes:
        for q in quantiles:
            row[ukey(mode, q)] = np.full(length, np.nan)
        row[umean_key(mode)] = np.full(length, np.nan)
    row["clip0"] = np.full(length, np.nan)
    row["clip1"] = np.full(length, np.nan)

    valid = sample.get("valid")
    for index in range(length):
        info = infos[index] if index < len(infos) else None
        if info is None:
            continue
        row["clip0"][index], row["clip1"][index] = clamp_pressure(info)
        for mode in modes:
            u = pixel_uncertainty(info, var_min=var_min, var_max=var_max, mode=mode)
            # infos stay at model resolution, so the mask is resampled to match
            quant = global_uncertainty(u, quantiles=quantiles, valid=valid)
            for q in quantiles:
                row[ukey(mode, q)][index] = quant[float(q)]
            row[umean_key(mode)][index] = float(u.mean().item())

    row["D"] = flow_delta(flows)
    row["ref"] = ref_series(flows)

    gt = sample.get("flow_gt")
    for key in ("epe", "px1", "fl"):
        row[key] = np.full(length, np.nan)
    if gt is not None:
        for index, flow in enumerate(flows):
            metrics = flow_metrics(flow, gt, valid)
            for key in ("epe", "px1", "fl"):
                row[key][index] = metrics[key]
    return row


def trace_dataset(
    model: Any,
    source: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
    out: Optional[str] = None,
    resume: bool = True,
    limit: Optional[int] = None,
) -> str:
    """Trace every sample and save one npz. Resumable: Colab disconnects."""
    kind = getattr(source, "kind", "source")
    path = out_path(out or cfg.get("out"), "traces", "trace_%s.npz" % kind)
    partial = path + ".partial.pkl"

    state: Dict[str, Any] = {"rows": [], "names": [], "done": 0}
    if resume and os.path.isfile(partial):
        try:
            with open(partial, "rb") as handle:
                state = pickle.load(handle)
            print("resuming: %d samples already traced" % state["done"])
        except Exception:
            state = {"rows": [], "names": [], "done": 0}

    total = None
    try:
        total = len(source)  # type: ignore[arg-type]
    except Exception:
        total = None
    if limit:
        total = min(total, int(limit)) if total else int(limit)

    iterator: Iterable[Dict[str, Any]] = source
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(source, total=total, ncols=80)
    except Exception:
        pass

    list_offset = int(getattr(model, "list_offset", 0))
    for index, sample in enumerate(iterator):
        if limit and index >= int(limit):
            break
        if index < state["done"]:
            continue
        state["rows"].append(trace_one(model, sample, cfg))
        state["names"].append(str(sample.get("name", index)))
        state["done"] = index + 1
        if state["done"] % PARTIAL_EVERY == 0:
            with open(partial, "wb") as handle:
                pickle.dump(state, handle)

    rows: List[Dict[str, np.ndarray]] = state["rows"]
    if not rows:
        raise RuntimeError("nothing traced: the source yielded no samples")

    keys = sorted({key for row in rows for key in row})
    width = max(len(row[next(iter(row))]) for row in rows)
    arrays: Dict[str, Any] = {}
    for key in keys:
        stacked = np.full((len(rows), width), np.nan)
        for i, row in enumerate(rows):
            series = np.asarray(row.get(key, []), dtype=np.float64)
            stacked[i, : series.size] = series[:width]
        arrays[key] = stacked

    unc_cfg = cfg.get("uncertainty", {})
    has_gt = bool(np.isfinite(arrays.get("epe", np.full((1, 1), np.nan))).any())
    np.savez_compressed(
        path,
        names=np.asarray(state["names"], dtype=object).astype("U"),
        list_offset=np.asarray(list_offset),
        modes=np.asarray([resolve_mode(m) for m in unc_cfg.get("modes", MODES)], dtype="U"),
        quantiles=np.asarray([float(q) for q in unc_cfg.get("quantiles", QUANTILES)]),
        source=np.asarray(str(kind)),
        has_gt=np.asarray(has_gt),
        **arrays,
    )
    save_json(
        {
            "trace": path,
            "n_samples": len(rows),
            "n_indices": int(width),
            "source": str(kind),
            "has_gt": has_gt,
            "list_offset": list_offset,
            "keys": keys,
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "provenance": provenance(),
        },
        out_path(out or cfg.get("out"), "tables", "trace_%s_meta.json" % kind),
    )
    if os.path.isfile(partial):
        try:
            os.remove(partial)
        except Exception:
            pass
    print(
        "traced %d samples x %d indices -> %s (%.0f KB)"
        % (len(rows), width, path, os.path.getsize(path) / 1024.0)
    )
    return path


def load_trace(path: str) -> Dict[str, Any]:
    """Load a trace npz into a plain dict (CPU only, no torch needed)."""
    with np.load(abs_path(path), allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    if "names" in data:
        data["names"] = [str(x) for x in np.atleast_1d(data["names"])]
    if "modes" in data:
        data["modes"] = [str(x) for x in np.atleast_1d(data["modes"])]
    if "quantiles" in data:
        data["quantiles"] = [float(x) for x in np.atleast_1d(data["quantiles"])]
    for key in ("source", "has_gt", "list_offset"):
        value = data.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 0:
            data[key] = value.item()
    return data
