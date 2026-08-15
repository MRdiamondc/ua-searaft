# UA-SEA-RAFT — uncertainty-aware adaptive stopping for SEA-RAFT

Training-free early exit for [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
(Wang, Lipson & Deng, ECCV 2024 Oral, [arXiv:2405.14793](https://arxiv.org/abs/2405.14793)).
SEA-RAFT already predicts a Mixture-of-Laplace uncertainty at **every** refinement
iteration and then throws it away. This repository uses it as a stopping signal:
run the refinement loop only until the uncertainty **saturates**, instead of
always paying for a fixed budget of 12 iterations.

No retraining, no fine-tuning, no change to upstream code. Upstream is used
unmodified as a git submodule.

```
U_i = Quantile_q( u_i )                     global uncertainty at iteration i
stop when  |U_{i-1} - U_i| / (U_{i-1} + eps) < tau_rel   (uncertainty saturated)
           AND  mean ||mu_i - mu_{i-1}||    < tau_delta  (flow stopped moving)
```

## The one insight that makes this cheap to study

A **global** (image-level) stop at iteration `n` is *bit-identical* to calling
the model with `iters=n`. So there is no need to re-run the network for every
threshold: run the full budget once, cache the per-iteration statistics, and
replay thousands of thresholds offline on a CPU. Latency is measured separately
via a cost model `T(n) = a + b·n`.

This equivalence is not assumed, it is asserted: `scripts/selftest.py` T6
checks `torch.equal` between a short run and the cached index, and prints the
measured list offset.

## The negative result you should read first

The uncertainty read-out shipped in upstream `custom.py` **cannot** be used as a
stopping signal, and it takes one line of arithmetic to see why:

```python
log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0,            max=args.var_max)
log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
```

The released eval configs set `var_min = 0`. Component 1 is therefore clamped to
`[0, 0]` — identically zero, so `b_1 = exp(0) = 1` px for every pixel, at every
iteration, forever. Component 0 is floored at 0, so wherever the network is
*confident* (sub-pixel scale, i.e. negative log-scale) it is pinned to 0 as well.
On a confident pair the scalar read-out collapses to the constant **1.000 px**.
Our first run measured exactly that:

```
u median = 1.000 px  (min 1, max 1)
U@0.8 per iteration = [1.0006 1. 1. 1. 1. 1. 1.]
```

Worse, a naive self-test calls that series "decreasing", because the `1.0006` at
index 0 makes it formally non-increasing. Figure 6 of the paper (variance
decreases with iterations) can only have been plotted from **pre-clamp** values.

So `ua_stop/uncertainty.py` exposes five read-outs from the same forward pass —
`raw`, `geo`, `clamped_lin`, `clamped_log`, `alpha` — and `clamp_pressure()`
reports the fraction of pixels pinned on each bound, which turns "the signal was
dead" into a number you can put in a table.

## Install

```bash
git clone https://github.com/USERNAME/ua-searaft.git
cd ua-searaft
pip install -r requirements.txt
bash scripts/setup_upstream.sh      # clones SEA-RAFT into third_party/, installs einops
```

Works on a free Colab T4 (16 GB). Only `einops` and `huggingface_hub` are needed
beyond a standard torch install; the released checkpoint
(`MemorySlices/Tartan-C-T-TSKH-spring540x960-M`, ~90 MB) is pulled automatically.

## Quickstart

```bash
python scripts/selftest.py          # 7 hard assertions. Never skip this.
python scripts/diagnose.py          # is the signal alive? which clamp killed it?
python scripts/run_latency.py       # T(n) = a + b n, and the saving ceiling
python scripts/run_trace.py         # one full-budget pass per pair -> .npz
python scripts/run_sweep.py         # the experiment, CPU only
python scripts/run_calibrate.py     # RCPS risk certificate
python scripts/make_figures.py      # fig1..fig4 as vector PDF
```

Every script accepts `--config` and repeatable dotted overrides:

```bash
python scripts/run_trace.py --config configs/kitti15.json --set model.scale=-1 --set source.n=120
```

## Layout

```
ua_stop/
  uncertainty.py    five MoL read-outs, clamp pressure, signal_health
  criterion.py      the stopping rule; pure numpy, six modes incl. ablations
  model_wrapper.py  upstream SEA-RAFT without forking it (+ the scale fix)
  data.py           synthetic exact-GT pairs, image folders, KITTI-2015
  trace.py          ONE full-budget pass per sample -> cached npz
  latency.py        T(n) = a + b n, criterion overhead, saving curve
  sweep.py          offline replay, Pareto front, cost-matched baseline, oracle
  conformal.py      RCPS risk certificate for the selected threshold
  plots.py          the four paper figures
  hooks.py          fallback signal: GRU hidden-state update norm
  tile_stop.py      EXPERIMENTAL per-tile stopping (modelled cost only)
scripts/            eight entry points, all with --help
configs/            default.json and kitti15.json (extends default)
tests/              pytest; the numpy-only ones run without a GPU
docs/               RESULTS.md (fill with your numbers), PUBLISH.md
```

## What is honest about the evaluation

These are the things a reviewer will attack, so they are built into the code:

1. **The ceiling is stated up front.** On a T4 at 1080×1920, refinement is only
   **47%** of total latency (`a = 1042.02 ms`, `b = 76.15 ms`) — the paper
   reports 51% for the L model on a 3090. Stopping at zero iterations saves
   **46.4%** and *nothing can save more*. The realistic target is 20–30% at
   under 1% accuracy loss.
2. **The baseline is a cost-matched fixed budget**, not the full budget. Beating
   "always 12 iterations" by spending less is trivial. `sweep.py` interpolates
   the fixed-N curve at the *same* mean cost and reports a paired bootstrap CI
   and a Wilcoxon test against it.
3. **`d_only` is reported always.** If the flow-delta alone matches the full
   criterion, the uncertainty head contributes nothing — that is the
   make-or-break ablation, and it is a first-class mode, not an appendix.
4. **An oracle bounds the headroom**, so "we could do better" becomes a number.
5. **Selection is certified.** The threshold is chosen on data, so `conformal.py`
   wraps it in RCPS with a Hoeffding bound and reports held-out risk too.
6. **Modelled vs measured is never blurred.** `tile_stop.py` accuracy is exact,
   its cost is a model of an ideal sparse kernel and says so in every output.
7. **The synthetic set is a development set only.** Exact ground truth, but no
   occlusions and no real motion statistics, so it is optimistic by construction.

## Known bug we shipped and fixed

`config/eval/spring-M.json` ships `scale: -1` and the checkpoint is trained at
540×960, but `cfg.setdefault("scale", 0)` can never override an existing value,
so our first run executed at 1080×1920 — roughly 4× slower for nothing. The fix
lives in `model_wrapper.build_args` (explicit `setattr`, never `setdefault`) and
is asserted by selftest T4 (`scale_check`), which fails loudly rather than
silently wasting an hour.

## Citation

If this code is useful, please cite SEA-RAFT as well:

```bibtex
@inproceedings{wang2024searaft,
  title     = {{SEA-RAFT}: Simple, Efficient, Accurate {RAFT} for Optical Flow},
  author    = {Wang, Yihan and Lipson, Lahav and Deng, Jia},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```

## License

This repository: MIT (see `LICENSE`). SEA-RAFT is BSD-3-Clause and is **not**
vendored here — `scripts/setup_upstream.sh` clones it into `third_party/`. See
`NOTICE` for attribution details.
