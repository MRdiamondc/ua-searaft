"""Thin wrapper around the upstream SEA-RAFT model (consumed as a submodule).

Responsibilities, in the order they bite you:

1. Put ``third_party/SEA-RAFT`` and its ``core/`` on ``sys.path``. Upstream's
   own ``custom.py`` does ``sys.path.append('core')``, which only works when
   the process happens to run from the upstream root; we do it by absolute
   path instead.

2. Build model arguments from JSON *without* upstream's argparse. Colab injects
   ``-f /root/.local/share/jupyter/runtime/kernel-xxx.json`` into ``sys.argv``
   and upstream's ``parse_args`` crashes on it.

3. Apply ``scale`` correctly. The shipped eval configs carry ``scale: -1``
   because the checkpoint is trained at 540x960. Our first run used
   ``cfg.setdefault("scale", 0)`` -- which cannot override an existing ``-1`` --
   and then never passed the value to the resize, so a 1080x1920 demo pair ran
   at full resolution: 1945 ms for 12 iterations instead of roughly 500 ms.
   ``SeaRaftWrapper.scale_check()`` turns that class of bug into an assertion
   (``scripts/selftest.py``, T4), so it cannot silently regress again.

4. Expose two entry points with *identical* numerics:
   ``forward_once(iters=n)`` for latency measurement, and ``forward_trace()``
   for the single full-budget pass that feeds the trace. Because a global stop
   at list index n is bit-identical to ``forward_once(iters=n)``
   (``detect_list_offset`` proves it with ``torch.equal``), the offline replay
   of any threshold is exact.

Image convention: float32 tensors, shape (B, 3, H, W), values in [0, 255].
SEA-RAFT normalises internally.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import abs_path, load_json, pick_device, repo_path

UPSTREAM_DEFAULT = repo_path("third_party", "SEA-RAFT")

_SETUP_HINT = (
    "Upstream SEA-RAFT not found at %s\n"
    "  fix:  bash scripts/setup_upstream.sh\n"
    "  or :  git submodule update --init --recursive"
)


def add_upstream_to_path(root: Optional[str] = None) -> str:
    """Make ``raft``, ``utils.utils`` and friends importable. Returns the root."""
    root = abs_path(root or UPSTREAM_DEFAULT)
    core = os.path.join(root, "core")
    if not os.path.isdir(core):
        raise FileNotFoundError(_SETUP_HINT % root)
    for path in (core, root):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def build_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Merge the upstream eval JSON with our explicit overrides.

    Overrides use ``setattr``, never ``setdefault``: the whole point is to be
    able to *change* values the upstream config already defines (``scale``!).
    """
    model_cfg = dict(cfg.get("model", {}))
    raw: Dict[str, Any] = {}
    if model_cfg.get("cfg"):
        raw = dict(load_json(abs_path(model_cfg["cfg"])))
    # some upstream configs nest the network hyper-parameters one level deep
    for key in ("model", "args"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            raw.pop(key)
            merged = dict(nested)
            merged.update(raw)
            raw = merged

    args = SimpleNamespace(**raw)
    for key in ("scale", "iters", "var_min", "var_max", "url", "path"):
        if model_cfg.get(key) is not None:
            setattr(args, key, model_cfg[key])
    for key, default in (
        ("scale", 0),
        ("iters", 12),
        ("var_min", 0.0),
        ("var_max", 10.0),
        ("url", None),
        ("path", None),
    ):
        if not hasattr(args, key):
            setattr(args, key, default)
    args.pad_mode = model_cfg.get("pad_mode", getattr(args, "pad_mode", "sintel"))
    args.amp = bool(model_cfg.get("amp", False))
    args.device = pick_device(model_cfg.get("device", "cuda"))
    return args


class SeaRaftWrapper:
    """Runs SEA-RAFT and exposes its full per-iteration output list."""

    def __init__(self, model: Any, args: SimpleNamespace, device: Optional[str] = None):
        self.model = model
        self.args = args
        self.device = torch.device(device or args.device)
        self.pad_mode = str(getattr(args, "pad_mode", "sintel"))
        self.scale = float(getattr(args, "scale", 0.0))
        self.max_iters = int(getattr(args, "iters", 12))
        self.var_min = float(getattr(args, "var_min", 0.0))
        self.var_max = float(getattr(args, "var_max", 10.0))
        self.amp = bool(getattr(args, "amp", False))
        self.upstream_root: Optional[str] = None
        self.input_size: Optional[Tuple[int, int]] = None
        self.model_size: Optional[Tuple[int, int]] = None
        self.list_offset = 0

    # -- construction ---------------------------------------------------

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> "SeaRaftWrapper":
        root = add_upstream_to_path((cfg.get("model") or {}).get("upstream"))
        args = build_args(cfg)
        from raft import RAFT  # upstream module, importable after path setup

        if getattr(args, "url", None):
            model = RAFT.from_pretrained(args.url, args=args)
        else:
            path = abs_path(args.path) if args.path else None
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(
                    "model.url is unset and model.path %r does not exist" % (args.path,)
                )
            model = RAFT(args)
            try:
                from utils.utils import load_ckpt

                load_ckpt(model, path)
            except Exception:
                state = torch.load(path, map_location="cpu")
                model.load_state_dict(state.get("model", state), strict=False)

        model = model.to(args.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        wrapper = cls(model, args)
        wrapper.upstream_root = root
        return wrapper

    # -- plumbing -------------------------------------------------------

    def _to_device(self, image: Any) -> torch.Tensor:
        x = image
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dim() == 3 and x.shape[-1] in (1, 3):  # HWC -> CHW
                x = x.permute(2, 0, 1)
        if not torch.is_tensor(x):
            raise TypeError("expected tensor or ndarray, got %r" % type(image))
        if x.dim() == 3:
            x = x[None]
        return x.to(device=self.device, dtype=torch.float32)

    def _padder(self, shape):
        from utils.utils import InputPadder

        return InputPadder(shape, mode=self.pad_mode)

    def _prep(self, img1: Any, img2: Any):
        x1 = self._to_device(img1)
        x2 = self._to_device(img2)
        self.input_size = (int(x1.shape[-2]), int(x1.shape[-1]))
        if self.scale != 0:
            factor = 2.0 ** self.scale
            x1 = F.interpolate(x1, scale_factor=factor, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, scale_factor=factor, mode="bilinear", align_corners=False)
        self.model_size = (int(x1.shape[-2]), int(x1.shape[-1]))
        padder = self._padder(x1.shape)
        x1, x2 = padder.pad(x1, x2)
        return x1, x2, padder

    def _call(self, x1: torch.Tensor, x2: torch.Tensor, iters: int):
        try:
            return self.model(x1, x2, iters=int(iters), test_mode=True)
        except TypeError:
            return self.model(x1, x2, iters=int(iters))

    def _run(self, x1: torch.Tensor, x2: torch.Tensor, iters: int):
        with torch.no_grad():
            if self.amp and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    return self._call(x1, x2, iters)
            return self._call(x1, x2, iters)

    @staticmethod
    def _as_lists(out: Any) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if isinstance(out, dict):
            flows, infos = out.get("flow"), out.get("info")
        elif isinstance(out, (list, tuple)) and len(out) >= 2:
            flows, infos = out[0], out[1]
        else:
            raise TypeError("unexpected model output type %r" % type(out))
        if torch.is_tensor(flows):
            flows = [flows]
        if torch.is_tensor(infos):
            infos = [infos]
        return list(flows or []), list(infos or [])

    def _restore_flow(self, flow: torch.Tensor, padder) -> torch.Tensor:
        """Undo padding and the ``scale`` downsample, magnitude included."""
        flow = padder.unpad(flow)
        if self.scale != 0 and self.input_size is not None:
            inv = 0.5 ** self.scale  # == 2.0 when scale == -1
            flow = (
                F.interpolate(
                    flow, size=self.input_size, mode="bilinear", align_corners=False
                )
                * inv
            )
        return flow

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # -- public API -----------------------------------------------------

    def forward_once(self, img1: Any, img2: Any, iters: Optional[int] = None) -> torch.Tensor:
        """Final flow only, at input resolution. Used for latency timing."""
        iters = self.max_iters if iters is None else int(iters)
        x1, x2, padder = self._prep(img1, img2)
        flows, _ = self._as_lists(self._run(x1, x2, iters))
        if not flows:
            raise RuntimeError("model returned no flow")
        return self._restore_flow(flows[-1], padder)

    def forward_trace(self, img1: Any, img2: Any, iters: Optional[int] = None) -> Dict[str, Any]:
        """Every intermediate flow and info map from ONE full-budget pass.

        ``flows`` are returned at input resolution; ``infos`` stay at model
        resolution on purpose -- the uncertainty read-out is a per-pixel
        statistic and resampling it would smear the tail we care about.
        """
        iters = self.max_iters if iters is None else int(iters)
        x1, x2, padder = self._prep(img1, img2)
        flows, infos = self._as_lists(self._run(x1, x2, iters))
        return {
            "flows": [self._restore_flow(f, padder) for f in flows],
            "infos": [padder.unpad(i) for i in infos],
            "iters": iters,
            "list_offset": self.list_offset,
            "input_size": self.input_size,
            "model_size": self.model_size,
        }

    def detect_list_offset(self, img1: Any, img2: Any, iters: int = 3) -> Optional[int]:
        """Find which list index equals ``forward_once(iters=iters)`` bit for bit.

        ``iterations = index + list_offset``. Measuring this instead of
        assuming it is what makes the offline replay trustworthy.
        """
        ref = self.forward_once(img1, img2, iters=int(iters))
        trace = self.forward_trace(img1, img2, iters=self.max_iters)
        for index, flow in enumerate(trace["flows"]):
            if flow.shape == ref.shape and torch.equal(flow, ref):
                self.list_offset = int(iters) - int(index)
                return self.list_offset
        return None

    def scale_check(self, tol: int = 8) -> Dict[str, Any]:
        """Did ``scale`` actually reach the resize? Run a forward pass first."""
        if self.input_size is None or self.model_size is None:
            raise RuntimeError("call forward_once()/forward_trace() first")
        expected = (
            int(round(self.input_size[0] * (2.0 ** self.scale))),
            int(round(self.input_size[1] * (2.0 ** self.scale))),
        )
        ok = (
            abs(self.model_size[0] - expected[0]) <= int(tol)
            and abs(self.model_size[1] - expected[1]) <= int(tol)
        )
        return {
            "scale": self.scale,
            "input_size": list(self.input_size),
            "model_size": list(self.model_size),
            "expected": list(expected),
            "tol": int(tol),
            "ok": bool(ok),
        }

    def describe(self) -> str:
        return (
            "SeaRaftWrapper(device=%s, scale=%s, iters=%s, var=[%s, %s], pad=%s, amp=%s)"
            % (
                self.device,
                self.scale,
                self.max_iters,
                self.var_min,
                self.var_max,
                self.pad_mode,
                self.amp,
            )
        )


def build_model(cfg: Dict[str, Any]) -> SeaRaftWrapper:
    """Convenience alias used by every script."""
    return SeaRaftWrapper.build(cfg)
