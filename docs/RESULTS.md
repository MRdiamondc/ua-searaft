# Results

Fill this file from the JSON artefacts in `outputs/tables/`. Every table below
maps 1:1 onto a saved file, so nothing has to be copied by hand twice, and
anything still marked `TODO` is a number we do **not** have yet — never a guess.

Hardware of the reference run: Google Colab free tier, NVIDIA T4 (16 GB),
torch 2.x, CUDA 12.x.

## 1. Latency cost model — `tables/latency.json`

Measured at 1080×1920 (i.e. **before** the `scale` fix), median of 25 reps,
8 warm-up passes, IQR ≤ 5 ms for n ≥ 5:

| n | T(n) [ms] | n | T(n) [ms] |
|---|-----------|---|-----------|
| 1 | 1069.52 | 7  | 1586.75 |
| 2 | 1209.09 | 8  | 1652.79 |
| 3 | 1280.21 | 9  | 1726.08 |
| 4 | 1361.69 | 10 | 1798.99 |
| 5 | 1432.98 | 11 | 1870.42 |
| 6 | 1509.56 | 12 | 1945.47 |

| quantity | value | note |
|---|---|---|
| fixed cost `a` | 1042.02 ms | encoders, correlation pyramid, padding, I/O |
| per-iteration `b` | 76.15 ms | |
| iteration share at n=12 | 47% | paper reports 51% for L on a 3090 |
| saving ceiling (n=0) | 46.4% vs measured T(12), 46.7% vs the fit | **the hard limit** |
| criterion overhead | 2.838 ms/iter = 3.7% of `b` | cheap (red line: 15%) |

Saving of a fixed budget: n=8 → 15.0%, n=6 → 22.4%, n=4 → 30.0%, n=2 → 37.9%.

**After the `scale` fix (540×960): TODO.** Expect ≈4× faster, `T(12) ≈ 500 ms`,
`b ≈ 19 ms`. Re-run `scripts/run_latency.py` and replace this whole section —
every latency number in the paper must come from the fixed configuration.

## 2. Signal liveness — `tables/selftest.json`, `tables/diagnose.json`

| read-out | first | last | range | verdict |
|---|---|---|---|---|
| `clamped_lin` (upstream) | 1.0006 | 1.0000 | 0.06% | **DEAD** (measured) |
| `clamped_log` | TODO | TODO | TODO | TODO |
| `raw` (pre-clamp) | TODO | TODO | TODO | TODO |
| `geo` | TODO | TODO | TODO | TODO |
| `alpha` | TODO | TODO | TODO | TODO |

Clamp pressure: `clip0 = TODO%`, `clip1 = TODO%`. If `clip1 ≈ 100%`, that is the
`var_min = 0` collapse described in the README.

If **all five** read-outs are degenerate, switch to the GRU hidden-state update
norm (`ua_stop/hooks.py`, `HiddenStateProbe`) and say so plainly: the
contribution then becomes "a cheap internal-dynamics signal", not "the model's
own uncertainty".

## 3. Main result — `tables/sweep.json`, `tables/sweep_all.csv`

| policy | mean iters | EPE (px) | Δ vs full | time saved |
|---|---|---|---|---|
| full budget (N=12) | 12.00 | TODO | — | 0% |
| cost-matched fixed N | TODO | TODO | TODO | TODO |
| **adaptive (ours)** | TODO | TODO | TODO | TODO |
| oracle | TODO | TODO | TODO | TODO |

Paired bootstrap CI (10 000 resamples) of ours − cost-matched: `TODO`.
Wilcoxon signed-rank p: `TODO`. Oracle gap: `TODO` iterations.

**The claim is only as strong as row 2.** If the CI crosses zero, the honest
sentence is "adaptive stopping matches a cost-matched fixed budget while
removing the need to tune N per dataset" — which is still a contribution, just a
smaller one.

## 4. Ablations — `tables/sweep_all.csv`

| mode | mean iters | EPE | reading |
|---|---|---|---|
| `and` (default) | TODO | TODO | both conditions |
| `or` | TODO | TODO | more aggressive |
| `u_only` | TODO | TODO | uncertainty alone |
| `d_only` | TODO | TODO | **make-or-break**: flow delta alone |
| `abs_only` | TODO | TODO | absolute threshold |

Also sweep `q ∈ {0.5, 0.7, 0.8, 0.9, 0.95}` and `patience ∈ {1, 2}`; the trace
already contains every quantile, so this costs nothing.

## 5. Risk certificate — `tables/conformal.json`

| α | δ | ε | scale | n cal | risk | UCB | held-out risk |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.1 | 0.05 px | 1.0 px | TODO | TODO | TODO | TODO |

Selected threshold: `TODO`. State α, δ and ε in the caption — the number is
meaningless without them — and note that Hoeffding is loose at n ≈ 40.

## 6. Per-tile headroom (experimental) — `tile_stop.py`

| policy | equivalent iters | mean active tiles | EPE | note |
|---|---|---|---|---|
| global | TODO | 100% | TODO | measured cost |
| per-tile 64px | TODO | TODO | TODO | **modelled cost only** |

Never present the tile numbers as wall-clock speed-ups; a dense CUDA kernel does
not get faster because some tiles are inactive.

## 7. Figures — `figs/`

| file | content |
|---|---|
| `fig1_pareto.pdf` | accuracy vs mean iterations, with the fixed-N curve and the oracle |
| `fig2_hist.pdf` | distribution of stop iterations (a single spike = not adaptive) |
| `fig3_signals.pdf` | all read-outs normalised to index 0 — the clamp story |
| `fig4_latency.pdf` | T(n) with the linear fit and the saving curve |

## 8. Threats to validity

- Synthetic pairs have exact GT but no occlusions → optimistic; KITTI-2015 train
  is the real check (`configs/kitti15.json`).
- The trace of record was N = 1 sample. Anything below ~40 pairs is a smoke test,
  not a result.
- Colab is a shared GPU; always report medians and IQR, never single timings.
- One checkpoint (`spring540x960-M`) and one resolution so far → do not
  generalise to S/L or to other datasets without re-running.
