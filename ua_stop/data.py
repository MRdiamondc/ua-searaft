"""Data sources: exact-GT synthetic pairs, an image folder, and KITTI-2015.

Every source yields dicts with

    name      str
    img1      (1, 3, H, W) float32 in [0, 255]
    img2      (1, 3, H, W) float32 in [0, 255]
    flow_gt   (1, 2, H, W) float32 or None
    valid     (1, 1, H, W) float32 or None
    has_gt    bool

Why a synthetic source at all
-----------------------------
The first run traced exactly ONE image pair (N = 1), which is enough to debug
plumbing and useless for statistics. KITTI-2015 is 2 GB and the network is not
always available. So ``SyntheticPairs`` warps a real photograph by a known
affine transform and derives the ground-truth flow *analytically* -- not by
re-estimating it -- which gives 120 pairs with exact GT in seconds and makes
the error bars in the paper meaningful.

Its honest caveat, which belongs in the paper: no occlusions, no independent
motion, and the pairs share content, so bootstrap intervals computed on it are
optimistic. It is a *development* set; KITTI-2015 remains the reported one.
"""

from __future__ import annotations

import glob
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import abs_path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff")


# -- image I/O ----------------------------------------------------------


def read_image(path: str) -> torch.Tensor:
    """Read an image as (1, 3, H, W) float32 in [0, 255], RGB."""
    path = abs_path(path)
    array: Optional[np.ndarray] = None
    try:
        import cv2

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            array = bgr[:, :, ::-1].copy()
    except Exception:
        array = None
    if array is None:
        from PIL import Image

        with Image.open(path) as img:
            array = np.asarray(img.convert("RGB"))
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    return tensor.permute(2, 0, 1)[None]


def resize_to(image: torch.Tensor, size: Optional[Sequence[int]]) -> torch.Tensor:
    if not size:
        return image
    height, width = int(size[0]), int(size[1])
    if (image.shape[-2], image.shape[-1]) == (height, width):
        return image
    return F.interpolate(image, size=(height, width), mode="area")


def procedural_image(size: Sequence[int], seed: int = 0) -> torch.Tensor:
    """Deterministic multi-scale texture, used only when no photo is available.

    Band-limited noise is a poor stand-in for natural images (its correlation
    structure is wrong), so this is a fallback, never the default.
    """
    height, width = int(size[0]), int(size[1])
    rng = np.random.default_rng(int(seed))
    acc = np.zeros((height, width, 3), dtype=np.float32)
    amplitude = 1.0
    for level in (4, 8, 16, 32, 64):
        coarse = rng.random((level, level, 3)).astype(np.float32)
        tensor = torch.from_numpy(coarse).permute(2, 0, 1)[None]
        up = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        acc += amplitude * up[0].permute(1, 2, 0).numpy()
        amplitude *= 0.6
    acc -= acc.min()
    acc /= max(float(acc.max()), 1e-6)
    return torch.from_numpy(acc * 255.0).permute(2, 0, 1)[None].float()


def list_images(folder: str) -> List[str]:
    folder = abs_path(folder)
    files: List[str] = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, "*" + ext)))
        files.extend(glob.glob(os.path.join(folder, "*" + ext.upper())))
    return sorted(set(files))


# -- sources ------------------------------------------------------------


class BaseSource:
    kind = "base"
    has_gt = False

    def __len__(self) -> int:  # pragma: no cover - trivial
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict[str, Any]]:  # pragma: no cover - trivial
        raise NotImplementedError

    def describe(self) -> str:
        return "%s (n=%d, gt=%s)" % (self.kind, len(self), self.has_gt)


class SyntheticPairs(BaseSource):
    """Affine-warped real images with analytic ground-truth flow.

    With ``c`` the image centre, ``M = (1 + zoom) * R(theta)`` and shift ``t``:

        Warp(p) = M (p - c) + c + t
        F(p)    = Warp(p) - p                          (exact, no estimation)
        im2(q)  = im1(Warp^-1(q))                      (bilinear resample)
        valid   = 1 where Warp(p) stays inside the frame
    """

    kind = "synthetic"
    has_gt = True

    def __init__(
        self,
        n: int = 120,
        size: Sequence[int] = (540, 960),
        images: Optional[Sequence[str]] = None,
        seed: int = 1234,
        max_rot_deg: float = 3.0,
        max_zoom: float = 0.04,
        max_shift_px: float = 25.0,
    ) -> None:
        self.n = int(n)
        self.size = (int(size[0]), int(size[1]))
        self.seed = int(seed)
        self.max_rot_deg = float(max_rot_deg)
        self.max_zoom = float(max_zoom)
        self.max_shift_px = float(max_shift_px)
        self.bases = self._load_bases(images)

    def _load_bases(self, images: Optional[Sequence[str]]) -> List[torch.Tensor]:
        bases: List[torch.Tensor] = []
        for path in images or []:
            resolved = abs_path(path)
            if os.path.isfile(resolved):
                bases.append(resize_to(read_image(resolved), self.size))
        if not bases:
            bases = [procedural_image(self.size, seed=s) for s in (11, 22)]
            self.synthetic_bases = True
        else:
            self.synthetic_bases = False
        return bases

    def __len__(self) -> int:
        return self.n

    def _params(self, rng: np.random.Generator) -> Tuple[float, float, float, float]:
        theta = math.radians(float(rng.uniform(-self.max_rot_deg, self.max_rot_deg)))
        zoom = float(rng.uniform(-self.max_zoom, self.max_zoom))
        tx = float(rng.uniform(-self.max_shift_px, self.max_shift_px))
        ty = float(rng.uniform(-self.max_shift_px, self.max_shift_px))
        return theta, zoom, tx, ty

    def _make(self, image: torch.Tensor, theta: float, zoom: float, tx: float, ty: float):
        _, _, height, width = image.shape
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        px, py = xs - cx, ys - cy
        scale = 1.0 + zoom
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # forward map: where does pixel p end up in image 2
        wx = scale * (cos_t * px - sin_t * py) + cx + tx
        wy = scale * (sin_t * px + cos_t * py) + cy + ty
        flow = torch.stack([wx - xs, wy - ys], dim=0)[None]
        valid = (
            (wx >= 0) & (wx <= width - 1) & (wy >= 0) & (wy <= height - 1)
        ).float()[None, None]

        # inverse map: sample image 1 to synthesise image 2
        qx, qy = xs - cx - tx, ys - cy - ty
        inv = 1.0 / scale
        ix = inv * (cos_t * qx + sin_t * qy) + cx
        iy = inv * (-sin_t * qx + cos_t * qy) + cy
        grid = torch.stack(
            [2.0 * ix / (width - 1) - 1.0, 2.0 * iy / (height - 1) - 1.0], dim=-1
        )[None]
        warped = F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return warped, flow, valid

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        for index in range(self.n):
            base = self.bases[index % len(self.bases)]
            theta, zoom, tx, ty = self._params(rng)
            img2, flow, valid = self._make(base, theta, zoom, tx, ty)
            yield {
                "name": "syn%04d" % index,
                "img1": base,
                "img2": img2,
                "flow_gt": flow,
                "valid": valid,
                "has_gt": True,
                "params": {
                    "theta_deg": math.degrees(theta),
                    "zoom": zoom,
                    "tx": tx,
                    "ty": ty,
                },
            }


class FolderPairs(BaseSource):
    """Consecutive images from a folder (or an explicit list). No ground truth."""

    kind = "folder"
    has_gt = False

    def __init__(
        self,
        folder: Optional[str] = None,
        images: Optional[Sequence[str]] = None,
        n: Optional[int] = None,
        size: Optional[Sequence[int]] = None,
    ) -> None:
        files: List[str] = []
        if folder:
            files = list_images(folder)
        if not files:
            files = [abs_path(p) for p in (images or []) if os.path.isfile(abs_path(p))]
        if len(files) < 2:
            raise FileNotFoundError(
                "need at least two images; folder=%r images=%r" % (folder, images)
            )
        self.files = files
        self.size = tuple(size) if size else None
        pairs = [(files[i], files[i + 1]) for i in range(len(files) - 1)]
        self.pairs = pairs[: int(n)] if n else pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for first, second in self.pairs:
            img1 = resize_to(read_image(first), self.size)
            img2 = resize_to(read_image(second), self.size)
            yield {
                "name": os.path.basename(first),
                "img1": img1,
                "img2": img2,
                "flow_gt": None,
                "valid": None,
                "has_gt": False,
            }


def read_kitti_flow(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """KITTI 16-bit flow PNG -> (H, W, 2) float flow and (H, W) valid mask."""
    path = abs_path(path)
    try:
        from utils.frame_utils import readFlowKITTI  # upstream helper

        flow, valid = readFlowKITTI(path)
        return np.asarray(flow, dtype=np.float32), np.asarray(valid, dtype=np.float32)
    except Exception:
        import cv2

        raw = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise FileNotFoundError("cannot read KITTI flow %s" % path)
        raw = raw[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        flow = (raw[:, :, :2] - 2.0 ** 15) / 64.0
        valid = raw[:, :, 2]
        return flow.astype(np.float32), (valid > 0.5).astype(np.float32)


class Kitti15Pairs(BaseSource):
    """KITTI-2015 flow training split: 200 pairs with sparse ground truth.

    Expected layout (the official zip, unpacked):
        <root>/training/image_2/000000_10.png, 000000_11.png, ...
        <root>/training/flow_occ/000000_10.png
    """

    kind = "kitti15"
    has_gt = True

    def __init__(
        self,
        root: str,
        n: Optional[int] = None,
        split: str = "training",
        occluded: bool = True,
    ) -> None:
        self.root = abs_path(root)
        self.split = str(split)
        image_dir = os.path.join(self.root, self.split, "image_2")
        flow_dir = os.path.join(
            self.root, self.split, "flow_occ" if occluded else "flow_noc"
        )
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                "KITTI images not found at %s (expected <root>/training/image_2)"
                % image_dir
            )
        self.flow_dir = flow_dir if os.path.isdir(flow_dir) else None
        firsts = sorted(glob.glob(os.path.join(image_dir, "*_10.png")))
        self.samples: List[Tuple[str, str, Optional[str]]] = []
        for first in firsts:
            second = first.replace("_10.png", "_11.png")
            if not os.path.isfile(second):
                continue
            flow = None
            if self.flow_dir:
                candidate = os.path.join(self.flow_dir, os.path.basename(first))
                flow = candidate if os.path.isfile(candidate) else None
            self.samples.append((first, second, flow))
        if n:
            self.samples = self.samples[: int(n)]
        self.has_gt = any(flow for _, _, flow in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for first, second, flow_path in self.samples:
            img1 = read_image(first)
            img2 = read_image(second)
            flow_gt = None
            valid = None
            if flow_path:
                flow_np, valid_np = read_kitti_flow(flow_path)
                flow_gt = torch.from_numpy(flow_np).permute(2, 0, 1)[None].float()
                valid = torch.from_numpy(valid_np)[None, None].float()
            yield {
                "name": os.path.basename(first),
                "img1": img1,
                "img2": img2,
                "flow_gt": flow_gt,
                "valid": valid,
                "has_gt": flow_gt is not None,
            }


def build_source(cfg: Dict[str, Any]) -> BaseSource:
    """Instantiate the source described by ``cfg["source"]``."""
    spec = dict(cfg.get("source", {}))
    kind = str(spec.get("kind", "synthetic")).lower()
    if kind == "synthetic":
        return SyntheticPairs(
            n=spec.get("n", 120),
            size=spec.get("size", (540, 960)),
            images=spec.get("images"),
            seed=spec.get("seed", 1234),
            max_rot_deg=spec.get("max_rot_deg", 3.0),
            max_zoom=spec.get("max_zoom", 0.04),
            max_shift_px=spec.get("max_shift_px", 25.0),
        )
    if kind == "folder":
        return FolderPairs(
            folder=spec.get("folder"),
            images=spec.get("images"),
            n=spec.get("n"),
            size=spec.get("size") if spec.get("resize") else None,
        )
    if kind in ("kitti15", "kitti", "kitti2015"):
        root = spec.get("kitti_root")
        if not root:
            raise ValueError("source.kitti_root is required for kind='kitti15'")
        return Kitti15Pairs(root=root, n=spec.get("n"), occluded=spec.get("occluded", True))
    raise ValueError(
        "source.kind must be one of ('synthetic', 'folder', 'kitti15'), got %r" % kind
    )


def first_sample(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """One sample, for self-tests and latency timing (never random noise)."""
    for sample in build_source(cfg):
        return sample
    raise RuntimeError("source produced no samples")
