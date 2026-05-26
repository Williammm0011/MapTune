"""Random-sample pruning experiment.

Usage: python experiment.py

Reads settings from [experiment] in config.toml.
For each trial: randomly pick sample_size gates, run ABC, observe used/unused,
remove unused, run ABC again. Produces a gate-usage heatmap saved to logs/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from src.abc_mapper import TechMapper
from src.experiment import run_trials


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_library(cfg, lib_name):
    for lib in cfg["library"]:
        if lib["name"] == lib_name:
            return lib["genlib"]
    raise SystemExit(f"Unknown library '{lib_name}'")


def resolve_design(cfg, bench_name):
    all_paths = (
        cfg["benchmarks"].get("bench", [])
        + cfg["benchmarks"].get("bench_large", [])
        + cfg["benchmarks"].get("bench_seq", [])
        + cfg["benchmarks"].get("blif", [])
    )
    for path in all_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem == bench_name:
            return path
    raise SystemExit(f"Unknown benchmark '{bench_name}'")


def plot_trials(trials, n_gates, lib_name, bench_name, out_path):
    n_trials = len(trials)
    n_cols = n_trials * 2  # left = random sample run, right = pruned run

    # Build color grid: 0=white, 1=light green (in lib unused), 2=dark green (in lib used)
    grid = np.zeros((n_gates, n_cols), dtype=int)
    for t, trial in enumerate(trials):
        left, right = t * 2, t * 2 + 1
        for i in trial["unused_idx"]:
            grid[i, left] = 1
        for i in trial["used_idx"]:
            grid[i, left] = 2
        for i in trial["unused_idx_b"]:
            grid[i, right] = 1
        for i in trial["used_idx_b"]:
            grid[i, right] = 2

    color_map = np.array([
        [1.00, 1.00, 1.00],   # 0 → white
        [0.60, 0.88, 0.60],   # 1 → light green
        [0.11, 0.37, 0.13],   # 2 → dark green
    ])
    rgb = color_map[grid]

    cell_w, cell_h = 0.28, 0.07
    fig_w = max(8, n_cols * cell_w + 2.0)
    fig_h = max(8, n_gates * cell_h + 3.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.08)
    ax_cost = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # --- Cost bar chart (top) ---
    bar_w = 0.4
    for t, trial in enumerate(trials):
        lx, rx = t * 2, t * 2 + 1
        ax_cost.bar(lx, trial["cost_before"], width=bar_w, color="#999999", zorder=2)
        ax_cost.bar(rx, trial["cost_after"],  width=bar_w, color="#1B5E20", zorder=2)
    ax_cost.axhline(1.0, color="red", linewidth=0.8, linestyle="--", label="baseline")
    ax_cost.set_xlim(-0.5, n_cols - 0.5)
    ax_cost.set_xticks([])
    ax_cost.set_ylabel("cost", fontsize=7)
    ax_cost.tick_params(axis="y", labelsize=6)
    ax_cost.set_title(f"Gate usage — {lib_name} / {bench_name}  "
                      f"(gray=before, green=after, red=baseline)", fontsize=8)
    for t in range(1, n_trials):
        ax_cost.axvline(x=t * 2 - 0.5, color="black", linewidth=0.8)

    # --- Heatmap (bottom) ---
    ax_heat.imshow(rgb, aspect="auto", interpolation="nearest",
                   extent=[-0.5, n_cols - 0.5, n_gates - 0.5, -0.5])

    for t in range(1, n_trials):
        ax_heat.axvline(x=t * 2 - 0.5, color="black", linewidth=0.8)
    for t in range(n_trials):
        ax_heat.axvline(x=t * 2 + 0.5, color="gray", linewidth=0.3, linestyle="--")

    ax_heat.set_xticks([t * 2 + 0.5 for t in range(n_trials)])
    ax_heat.set_xticklabels([f"T{t+1}" for t in range(n_trials)], fontsize=7)
    ax_heat.xaxis.set_label_position("bottom")
    ax_heat.set_xlabel("L = random sample  |  R = after removing unused", fontsize=7)
    ax_heat.set_ylabel("Gate index", fontsize=8)

    legend = [
        mpatches.Patch(facecolor=[1, 1, 1], edgecolor="gray", label="not in lib"),
        mpatches.Patch(color=color_map[1], label="in lib, unused"),
        mpatches.Patch(color=color_map[2], label="in lib, used"),
    ]
    ax_heat.legend(handles=legend, loc="lower right",
                   bbox_to_anchor=(1.0, -0.06), fontsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def main():
    cfg = load_config()
    ecfg = cfg["experiment"]
    lib_name    = ecfg["lib"]
    bench_name  = ecfg["bench"]
    n_trials    = ecfg["n_trials"]
    sample_size = ecfg["sample_size"]

    genlib = resolve_library(cfg, lib_name)
    design = resolve_design(cfg, bench_name)

    os.makedirs(cfg["paths"]["temp_blifs_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["gen_newlibs_dir"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    temp_blif = f"{cfg['paths']['temp_blifs_dir']}/{bench_name}_exp_temp.blif"

    print(f"Library     : {genlib}")
    print(f"Design      : {design}")
    print(f"Trials      : {n_trials}  |  sample_size: {sample_size}")
    print()

    mapper = TechMapper(genlib, design, cfg["paths"]["gen_newlibs_dir"] + "/", temp_blif)
    print(f"Total mutable gates: {mapper.num_arms}")
    print(f"Baseline cost      : 1.0000  "
          f"(delay={mapper.baseline_delay:.1f}ps, area={mapper.baseline_area:.1f})\n")

    trials = run_trials(mapper, n_trials, sample_size)

    out_path = f"logs/experiment_{lib_name}_{bench_name}.png"
    plot_trials(trials, mapper.num_arms, lib_name, bench_name, out_path)


if __name__ == "__main__":
    main()
