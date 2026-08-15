"""Paper figures. Vector PDF, embedded TrueType, English labels only.

Four figures, each answering one question a reviewer will ask:

    fig1_pareto.pdf     Does adaptivity beat a cost-matched fixed budget?
    fig2_hist.pdf       Where does the rule actually stop? (is it adaptive at all?)
    fig3_signals.pdf    Which read-out is alive? (the clamp story, in one plot)
    fig4_latency.pdf    Is latency really linear in iterations, and what is the ceiling?

Deliberately torch-free and matplotlib-only: figures are regenerated from the
saved JSON/NPZ artefacts on a laptop, without a GPU and without re-running
anything.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .utils import abs_path, load_json, out_path

FIGSIZE = (4.2, 3.1)


def _setup():
    """Configure matplotlib for headless, publication-quality PDF output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,  # TrueType, so the PDF is editable and embeds cleanly
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def _save(fig, path: str) -> str:
    path = abs_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.clf()
    print("figure -> %s" % path)
    return path


def fig_pareto(sweep: Dict[str, Any], path: str) -> str:
    """Accuracy vs mean iterations: all configs, the front, and both baselines.

    The fixed-budget curve is the honest baseline. Any point BELOW it at the
    same x wins by being adaptive rather than merely cheaper.
    """
    plt = _setup()
    metric = sweep.get("metric", "epe")
    rows = sweep.get("all_rows", [])
    front = sweep.get("pareto", [])
    curve = sweep.get("fixed_curve", [])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if rows:
        ax.scatter(
            [r["budget"] for r in rows],
            [r["metric"] for r in rows],
            s=4,
            c="0.75",
            linewidths=0,
            label="all configurations",
        )
    if curve:
        ax.plot(
            range(len(curve)),
            curve,
            color="tab:gray",
            ls="--",
            lw=1.0,
            marker="s",
            ms=2.5,
            label="fixed budget N",
        )
    if front:
        ax.plot(
            [r["budget"] for r in front],
            [r["metric"] for r in front],
            color="tab:blue",
            lw=1.3,
            marker="o",
            ms=3,
            label="adaptive (Pareto)",
        )
    oracle = sweep.get("oracle") or {}
    if oracle:
        ax.scatter(
            [oracle["budget"]],
            [oracle["metric"]],
            marker="*",
            s=55,
            c="tab:green",
            label="oracle",
            zorder=5,
        )
    head = sweep.get("headline") or {}
    if head.get("found"):
        ax.scatter(
            [head["budget"]],
            [head["metric"]],
            marker="D",
            s=28,
            facecolors="none",
            edgecolors="tab:red",
            lw=1.2,
            label="selected",
            zorder=6,
        )
    if "full_budget" in sweep and "full_metric" in sweep:
        ax.axhline(sweep["full_metric"], color="k", lw=0.7, ls=":")
        ax.annotate(
            "full budget",
            xy=(sweep["full_budget"], sweep["full_metric"]),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=6.5,
        )
    ax.set_xlabel("mean refinement iterations")
    ax.set_ylabel("EPE (px)" if metric == "epe" else "distance to full-budget flow (px)")
    ax.set_title("Accuracy vs compute")
    ax.legend(loc="upper right", frameon=False)
    return _save(fig, path)


def fig_budget_hist(sweep: Dict[str, Any], path: str, list_offset: int = 0) -> str:
    """Distribution of stop indices. A single spike means the rule is not adaptive."""
    plt = _setup()
    head = sweep.get("headline") or {}
    hist: Sequence[float] = head.get("stop_hist") or sweep.get("stop_hist") or []
    if not hist:
        for row in sweep.get("all_rows", []):
            if row.get("label") == head.get("label") and row.get("stop_hist"):
                hist = row["stop_hist"]
                break

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if hist:
        counts = np.asarray(hist, dtype=np.float64)
        total = counts.sum() or 1.0
        iters = np.arange(counts.size) + int(list_offset)
        ax.bar(iters, 100.0 * counts / total, width=0.72, color="tab:blue")
        mean = float((iters * counts).sum() / total)
        ax.axvline(mean, color="tab:red", lw=1.0, ls="--")
        ax.annotate(
            "mean %.2f" % mean,
            xy=(mean, ax.get_ylim()[1] * 0.9),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=6.5,
            color="tab:red",
        )
    else:
        ax.text(0.5, 0.5, "no stop histogram in sweep.json", ha="center", fontsize=7)
    ax.set_xlabel("stop iteration")
    ax.set_ylabel("share of samples (%)")
    ax.set_title("Where the rule stops")
    return _save(fig, path)


def fig_signals(trace_path: str, path: str, quantile: float = 0.8) -> str:
    """Every read-out, normalised to its own first value: the liveness figure.

    A flat line at 1.0 is a dead signal. This is where the ``clamped_*`` curves
    sit when ``var_min = 0``, and it is the single most informative plot in the
    report: it shows the negative result and the fix in the same axes.
    """
    plt = _setup()
    with np.load(abs_path(trace_path), allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    keys = sorted(k for k in data if k.startswith("U:") and k.endswith("@%.2f" % quantile))
    plotted = 0
    for key in keys:
        series = np.nanmean(np.atleast_2d(data[key]), axis=0)
        if not np.isfinite(series).any() or series[0] == 0:
            continue
        ax.plot(
            range(series.size),
            series / series[0],
            marker="o",
            ms=2.5,
            lw=1.1,
            label=key.split(":", 1)[1].split("@")[0],
        )
        plotted += 1
    if "D" in data:
        delta = np.nanmean(np.atleast_2d(data["D"]), axis=0)
        finite = np.where(np.isfinite(delta))[0]
        if finite.size:
            ref = delta[finite[0]]
            ax.plot(
                finite,
                delta[finite] / (ref if ref else 1.0),
                color="k",
                ls="--",
                lw=1.0,
                marker="^",
                ms=2.5,
                label="flow delta",
            )
    ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
    if not plotted:
        ax.text(0.5, 0.5, "no U:* series at q=%.2f" % quantile, ha="center", fontsize=7)
    ax.set_xlabel("refinement iteration (list index)")
    ax.set_ylabel("signal, normalised to index 0")
    ax.set_title("Signal liveness at q = %.2f" % quantile)
    ax.legend(loc="best", frameon=False, ncol=2)
    return _save(fig, path)


def fig_latency(latency: Dict[str, Any], path: str) -> str:
    """Measured T(n) with the linear fit, plus the saving curve on a twin axis."""
    plt = _setup()
    table = latency.get("table", [])
    fit = latency.get("fit", {})

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if table:
        n = np.asarray([row["n"] for row in table], dtype=np.float64)
        ms = np.asarray([row["ms"] for row in table], dtype=np.float64)
        err = np.asarray([row.get("iqr", 0.0) for row in table], dtype=np.float64)
        ax.errorbar(n, ms, yerr=err / 2.0, fmt="o", ms=3, lw=0.8, capsize=1.5, label="measured")
        if fit:
            grid = np.linspace(0.0, float(n.max()), 50)
            ax.plot(
                grid,
                fit["a_fixed_ms"] + fit["b_iter_ms"] * grid,
                color="tab:red",
                lw=1.0,
                label="a + b n  (a=%.0f ms, b=%.1f ms)" % (fit["a_fixed_ms"], fit["b_iter_ms"]),
            )
            ax.axhline(fit["a_fixed_ms"], color="0.5", lw=0.7, ls=":")
            ax.annotate(
                "fixed cost: iterations are only %.0f%% of T(%d)"
                % (100.0 * fit["share_at_n_max"], int(fit["n_max"])),
                xy=(0.04, 0.06),
                xycoords="axes fraction",
                fontsize=6.5,
            )
    curve = latency.get("saving_curve", [])
    if curve:
        twin = ax.twinx()
        twin.plot(
            [row["n"] for row in curve],
            [100.0 * row["saving"] for row in curve],
            color="tab:green",
            lw=1.0,
            ls="--",
        )
        twin.set_ylabel("time saved (%)", color="tab:green")
        twin.tick_params(axis="y", labelcolor="tab:green")
        twin.grid(False)
    ax.set_xlabel("refinement iterations n")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency is linear in iterations")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, path)


def make_all(cfg: Dict[str, Any], out: Optional[str] = None, trace_path: Optional[str] = None) -> List[str]:
    """Regenerate every figure that has its artefact on disk. Missing = skipped."""
    out = out or cfg.get("out")
    made: List[str] = []

    sweep_json = out_path(out, "tables", "sweep.json")
    if os.path.isfile(abs_path(sweep_json)):
        sweep = load_json(sweep_json)
        made.append(fig_pareto(sweep, out_path(out, "figs", "fig1_pareto.pdf")))
        made.append(
            fig_budget_hist(
                sweep,
                out_path(out, "figs", "fig2_hist.pdf"),
                list_offset=int(sweep.get("list_offset", 0)),
            )
        )
    else:
        print("skip fig1/fig2: %s not found" % sweep_json)

    if trace_path is None:
        kind = str(cfg.get("source", {}).get("kind", "synthetic"))
        trace_path = out_path(out, "traces", "trace_%s.npz" % kind)
    if os.path.isfile(abs_path(trace_path)):
        q = float(cfg.get("uncertainty", {}).get("primary_q", 0.8))
        made.append(fig_signals(trace_path, out_path(out, "figs", "fig3_signals.pdf"), q))
    else:
        print("skip fig3: %s not found" % trace_path)

    latency_json = out_path(out, "tables", "latency.json")
    if os.path.isfile(abs_path(latency_json)):
        made.append(fig_latency(load_json(latency_json), out_path(out, "figs", "fig4_latency.pdf")))
    else:
        print("skip fig4: %s not found" % latency_json)

    return made
