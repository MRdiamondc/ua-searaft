"""The synthetic pairs must have EXACT ground truth, or the dev set is a lie.

The generator applies a known affine warp

    Warp(p) = M (p - c) + c + t,     M = (1 + zoom) R(theta)
    F(p)    = Warp(p) - p
    im2(q)  = im1(Warp^-1(q))

so the flow it reports is analytic, not estimated. These tests verify the two
properties that can silently break: that warping image 1 by the reported flow
reproduces image 2, and that the validity mask really marks out-of-bounds
pixels.

Needs torch (the generator uses ``grid_sample``); skipped automatically when
torch is unavailable.

    python -m pytest tests/test_synthetic_gt.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from ua_stop.data import SyntheticPairs, procedural_image, resize_to  # noqa: E402

SIZE = (96, 128)


def warp_with_flow(image, flow):
    """Sample ``image`` at ``p + flow(p)``: the standard forward-warp check."""
    import torch.nn.functional as F

    height, width = image.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    x = xs[None] + flow[:, 0]
    y = ys[None] + flow[:, 1]
    grid = torch.stack(
        [2.0 * x / (width - 1) - 1.0, 2.0 * y / (height - 1) - 1.0], dim=-1
    )
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def first(source):
    for sample in source:
        return sample
    raise AssertionError("source yielded nothing")


def test_sample_shapes_and_ranges():
    sample = first(SyntheticPairs(n=2, size=SIZE, seed=0))
    assert sample["img1"].shape == (1, 3, SIZE[0], SIZE[1])
    assert sample["img2"].shape == sample["img1"].shape
    assert sample["flow_gt"].shape == (1, 2, SIZE[0], SIZE[1])
    assert sample["has_gt"] is True
    assert float(sample["img1"].min()) >= 0.0
    assert float(sample["img1"].max()) <= 255.0


def test_flow_reproduces_image2_on_valid_pixels():
    """If this fails, every EPE computed on synthetic data is meaningless."""
    sample = first(SyntheticPairs(n=1, size=SIZE, seed=1))
    warped = warp_with_flow(sample["img2"], sample["flow_gt"])
    valid = sample["valid"]
    mask = valid if valid.dim() == 4 else valid[None, None]
    # Erode the mask slightly: bilinear sampling at the very border is not
    # exact, and that is expected rather than a bug.
    inner = torch.zeros_like(mask)
    inner[..., 2:-2, 2:-2] = mask[..., 2:-2, 2:-2]
    error = (warped - sample["img1"]).abs() * inner
    denom = float(inner.sum().item()) * sample["img1"].shape[1]
    mean_error = float(error.sum().item()) / max(denom, 1.0)
    assert mean_error < 8.0, "mean intensity error %.2f/255 is too high" % mean_error


def test_zero_motion_gives_zero_flow():
    source = SyntheticPairs(
        n=1, size=SIZE, seed=2, max_rot_deg=0.0, max_zoom=0.0, max_shift_px=0.0
    )
    sample = first(source)
    assert float(sample["flow_gt"].abs().max()) < 1e-3
    assert torch.allclose(sample["img1"], sample["img2"], atol=1.0)


def test_pure_shift_gives_a_constant_flow():
    source = SyntheticPairs(
        n=1, size=SIZE, seed=3, max_rot_deg=0.0, max_zoom=0.0, max_shift_px=5.0
    )
    sample = first(source)
    flow = sample["flow_gt"]
    # a translation-only warp has a spatially constant flow field
    assert float(flow[:, 0].std()) < 1e-3
    assert float(flow[:, 1].std()) < 1e-3
    assert float(flow.abs().max()) <= 5.0 + 1e-3


def test_valid_mask_marks_out_of_bounds():
    source = SyntheticPairs(
        n=1, size=SIZE, seed=4, max_rot_deg=0.0, max_zoom=0.0, max_shift_px=20.0
    )
    sample = first(source)
    fraction = float(sample["valid"].float().mean())
    assert 0.5 < fraction < 1.0, "a 20 px shift must invalidate a border strip"


def test_determinism_from_the_seed():
    a = first(SyntheticPairs(n=1, size=SIZE, seed=5))
    b = first(SyntheticPairs(n=1, size=SIZE, seed=5))
    assert torch.equal(a["img2"], b["img2"])
    assert torch.equal(a["flow_gt"], b["flow_gt"])


def test_procedural_image_and_resize():
    image = procedural_image(SIZE, seed=0)
    assert image.shape[-2:] == SIZE
    smaller = resize_to(image, (48, 64))
    assert smaller.shape[-2:] == (48, 64)
    assert float(smaller.min()) >= 0.0


def test_len_matches_requested_count():
    source = SyntheticPairs(n=7, size=SIZE, seed=6)
    assert len(source) == 7
    assert sum(1 for _ in source) == 7
    assert np.isfinite(float(first(source)["flow_gt"].mean()))
