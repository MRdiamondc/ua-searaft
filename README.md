# UA-SEA-RAFT

**Uncertainty-Aware Adaptive Refinement for SEA-RAFT**

Training-free adaptive inference for [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) that reuses its Mixture-of-Laplace (MoL) head as an uncertainty signal instead of spending the same refinement budget everywhere.

> Research implementation. Run the self-tests before reporting benchmark numbers.

## Why this exists

SEA-RAFT predicts MoL parameters at every refinement iteration, but the uncertainty is discarded at inference. The standard pipeline still spends a fixed budget, usually up to 12 iterations, on every pixel. Smooth regions can converge early, while boundaries, occlusions, and large motion need more refinement.

UA-SEA-RAFT adds adaptive halting without retraining:

```text
U_i = Quantile_q(u_i)
stop when relative uncertainty change is small
         AND the flow update is small
```

The global policy is bit-identical to running SEA-RAFT with `iters=n`. This makes threshold sweeps reproducible from one full-budget trace instead of requiring a new GPU run for every threshold.

## Main contributions

1. **Correct MoL inference read-out.** The upstream training clamp is not reused at inference. With the released `var_min=0` setting, the naive read-out forces one component to `b=1 px` and pins confident predictions, collapsing the scalar signal. The implementation reads the pre-clamp `info` tensor directly:

   ```text
   w = softmax(logits)
   b = exp(clipped raw_log_b)
   E|e| = sum_k w_k b_k
   F(t) = sum_k w_k (1 - exp(-t / b_k))
   ```

2. **Adaptive stopping.** Supports image-level `global` halting and experimental `tile` halting. The global method is exact. Tile mode uses exact correlation-row indexing on the active crop and reports its frozen-context semantics explicitly.

3. **Calibration evaluation.** Includes AUSE, coverage reliability, coverage-ECE, Spearman rank correlation, heavy-component AUROC, and one-parameter scale recalibration.

4. **Honest evaluation.** Includes `d_only`, uncertainty-only ablations, an oracle upper bound, cost-matched fixed-budget comparison, paired bootstrap intervals, Wilcoxon signed-rank testing, and an RCPS/Hoeffding risk certificate.

5. **Two implementation fixes.**
   - Explicitly overrides the upstream `scale` value instead of using a failing `setdefault` pattern.
   - Replaces the invalid â€œmonotonicity means healthyâ€ test with dynamic-range and spatial-variance checks.

6. **Free baseline speedup.** Upsampling is performed once after the refinement loop rather than once per iteration when only the final upsampled result is used.

## Repository contents

The current reference implementation is intentionally self-contained:

```text
ua_searaft.py       model wrapper, MoL read-out, stopping, metrics, tests
README.md           this document
```

SEA-RAFT is used as an external upstream dependency and is not vendored here.

## Installation

```bash
git clone https://github.com/MRdiamondc/ua-searaft.git
cd ua-searaft

# Clone the upstream implementation outside this repository.
git clone https://github.com/princeton-vl/SEA-RAFT third_party/SEA-RAFT

pip install torch numpy einops huggingface_hub
```

For loading local image files, install Pillow:

```bash
pip install pillow
```

The released checkpoint is loaded through Hugging Face when running the model:

```text
MemorySlices/Tartan-C-T-TSKH-spring540x960-M
```

## Quickstart

Run the CPU-only self-tests first:

```bash
python ua_searaft.py selftest
```

Inspect whether the uncertainty signal is alive:

```bash
python ua_searaft.py diagnose \
  --set scale=-1 \
  --image1 path/to/frame_0001.png \
  --image2 path/to/frame_0002.png
```

Measure the latency model on the target GPU:

```bash
python ua_searaft.py latency \
  --set scale=-1 \
  --height 540 \
  --width 960
```

If image paths are omitted, `diagnose` uses a synthetic shifted pair for development only. Synthetic data is optimistic: it has exact construction but no real occlusions or motion statistics.

## Configuration overrides

Use repeatable dotted overrides:

```bash
python ua_searaft.py diagnose \
  --set scale=-1 \
  --set halt.mode=full \
  --set halt.granularity=global \
  --set halt.q=0.8 \
  --set halt.tau_rel=0.02 \
  --set halt.tau_delta=0.05 \
  --set halt.min_iters=2 \
  --set halt.max_iters=12
```

Important options:

| Option | Meaning | Default |
|---|---|---:|
| `scale` | Resolution exponent relative to the input | `-1` |
| `halt.mode` | `full`, `u_rel`, `u_abs`, `d_only`, `fixed`, `random`, or `oracle` | `full` |
| `halt.granularity` | `global` or `tile` | `global` |
| `halt.q` | Quantile used for global uncertainty | `0.80` |
| `halt.tau_rel` | Relative uncertainty saturation threshold | `0.02` |
| `halt.tau_abs` | Absolute uncertainty-drop threshold | `0.01` |
| `halt.tau_delta` | Flow-update threshold | `0.05` |
| `halt.min_iters` | Minimum refinement iterations | `2` |
| `halt.max_iters` | Maximum refinement iterations | `12` |
| `halt.tile` | Tile size at 1/8 resolution | `16` |
| `mol.b_clip` | Numerical log-scale guard, not the training clamp | `8.0` |
| `mol.recal_scale` | Multiplicative uncertainty recalibration | `1.0` |

## Evaluation protocol

Do not compare adaptive inference only against â€œalways 12 iterationsâ€. That comparison rewards any method that spends less. The primary baseline is a fixed iteration count with the **same measured mean cost**.

Report all of the following:

- EPE and 1-pixel outlier rate.
- Mean and percentile iterations used.
- Measured latency and the fitted `T(n)=a+bÂ·n` model.
- Cost-matched fixed-budget baseline.
- `d_only` ablation, to test whether uncertainty contributes beyond flow movement.
- Uncertainty-only ablations.
- Oracle iteration choice, as an upper bound.
- Paired bootstrap confidence interval and Wilcoxon signed-rank test.
- Calibration: AUSE, coverage-ECE, signed coverage bias, Spearman correlation, and heavy-component AUROC.
- RCPS-selected threshold and held-out risk.

The absolute saving ceiling of the linear cost model is the refinement fraction at the maximum budget. On the example T4 measurements (`a=1042.02 ms`, `b=76.15 ms`), this ceiling is about 46%, not a claim of achievable accuracy-preserving speedup.

## Exactness notes

- **Global halting:** exact and bit-identical to fixed-budget SEA-RAFT with the selected iteration count.
- **Tile halting:** correlation lookup on the cropped active region is exact, but frozen tiles provide frozen context to active neighbours through the measured halo. Results are therefore labelled `exact-with-frozen-context`, not incorrectly advertised as bit-identical.
- **Tile cost:** the current implementation performs real cropped inference, but any ideal sparse-kernel projection must be labelled modelled until measured with a production sparse kernel.

## Self-tests

`python ua_searaft.py selftest` checks:

- the naive MoL read-out collapses under the released clamp settings;
- the corrected read-out has a live dynamic range;
- MoL quantiles invert the CDF;
- cropped correlation equals the corresponding full-grid slice;
- global early exit equals fixed `iters=n`;
- the resolution-scale override cannot regress;
- the receptive-field halo is measured;
- tile halting stays inside budget;
- AUSE, RCPS, and cost-matched evaluation behave as expected.

## Related work

- Y. Wang, L. Lipson, and J. Deng. **SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow.** ECCV 2024 Oral. [arXiv:2405.14793](https://arxiv.org/abs/2405.14793).
- Y. Wang and J. Deng. **WAFT: Warping-Alone Field Transforms for Optical Flow.** ICLR 2026 Oral. [arXiv:2506.21526](https://arxiv.org/abs/2506.21526). WAFT replaces the cost volume with high-resolution warping and is a useful second baseline for testing whether adaptive halting transfers beyond SEA-RAFT.

## Citation

Please cite SEA-RAFT when using this repository:

```bibtex
@inproceedings{wang2024searaft,
  title     = {{SEA-RAFT}: Simple, Efficient, Accurate {RAFT} for Optical Flow},
  author    = {Wang, Yihan and Lipson, Lahav and Deng, Jia},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```

## License

This repository is MIT. SEA-RAFT is BSD-3-Clause and is not vendored here. See the upstream repository and the project license files for attribution details.
