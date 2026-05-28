"""
SA agent with "remove 5 unused + add 2" action for gate library subset selection.

Each iteration:
  1. Uses cached gate-usage info from the last accepted mapping to identify
     which gates in the current subset ABC did NOT instantiate (unused).
  2. Removes up to 5 randomly chosen unused gates.
  3. Adds 2 randomly chosen gates from outside the current subset.
  4. Maps the neighbour and applies the standard SA acceptance criterion.

After 100 iterations, saves a per-trial heatmap (gate usage × cost bars).
"""

import argparse
import csv
import datetime
import json
import logging
import math
import os
import random
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import tomli as tomllib

from abc_mapper import TechMapper

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── LOGGING MANAGER ─────────────────────────────────────────────────────────


class RunLogger:
    def __init__(self, base_dir: str, run_name: str, args_dict: dict):
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        log_file = os.path.join(self.run_dir, "run.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )

        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(args_dict, f, indent=2)

        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.fieldnames = [
            "step", "temperature", "n_selected", "n_unused", "n_removed", "n_added",
            "area", "delay", "cost", "best_cost", "accepted", "elapsed_s"
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def info(self, msg: str):
        logging.info(msg)

    def log_step(self, row_dict: dict):
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
        self.info(f"\n[Logs saved to {self.run_dir}/]")


# ── CONFIG & PARSING ────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.path.join(_ROOT, "config.toml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _resolve_library(cfg: dict, lib_name: str) -> str:
    for lib in cfg.get("library", []):
        if lib["name"] == lib_name:
            return os.path.join(_ROOT, lib["genlib"])
    raise ValueError(f"Library '{lib_name}' not found in config.toml [[library]] entries.")


def parse_args():
    cfg = _load_config()

    default_lib = cfg.get("library", [{"name": "7nm"}])[0]["name"]
    default_bench = cfg.get("benchmarks", {}).get("bench", ["benchmarks/c880.bench"])[0]
    paths = cfg.get("paths", {})
    sp = cfg.get("sa_prune", {})

    p = argparse.ArgumentParser(description="SA with remove-unused + add action for MapTune")
    p.add_argument("--lib", default=default_lib, help="Library name from config.toml")
    p.add_argument("--bench", default=default_bench, help="Path to .bench / .blif file")
    p.add_argument("--out-dir", default=paths.get("gen_newlibs_dir", "gen_newlibs/"),
                   help="Output dir for mapped libraries")
    p.add_argument("--temp-blif", default=os.path.join(
                   paths.get("temp_blifs_dir", "temp_blifs"), "sa_prune_temp.blif"),
                   help="Temporary blif file")
    p.add_argument("--log-dir", default=paths.get("log_dir", "logs"),
                   help="Base directory for run logs")
    p.add_argument("--n-select", type=int, default=sp.get("n_select", 50),
                   help="Initial number of selected gates")
    p.add_argument("--iterations", type=int, default=sp.get("iterations", 100),
                   help="Total SA iterations")
    p.add_argument("--t0", type=float, default=sp.get("t0", 0.5),
                   help="Initial temperature")
    p.add_argument("--t-min", type=float, default=sp.get("t_min", 0.001),
                   help="Final minimum temperature")
    p.add_argument("--seed", type=int, default=sp.get("seed", 42),
                   help="Random seed")
    p.add_argument("--n-remove", type=int, default=sp.get("n_remove", 5),
                   help="Max unused gates to remove per action")
    p.add_argument("--n-add", type=int, default=sp.get("n_add", 2),
                   help="Gates to add per action")
    p.add_argument("--random-remove", action="store_true",
                   default=sp.get("random_remove", False),
                   help="Pick removed gates randomly from current lib (not just unused)")
    return p.parse_args(), cfg


def calculate_cost(delay, area, base_delay, base_area):
    if math.isnan(delay) or math.isnan(area) or delay <= 0 or area <= 0:
        return float("inf")
    return (delay / base_delay) * (area / base_area)


# ── PLOTTING ─────────────────────────────────────────────────────────────────

def _encode_col(state: set, used: set, n_gates: int) -> list:
    """Return length-n_gates int list: 0=not in lib, 1=in lib unused, 2=in lib used."""
    col = [0] * n_gates
    for i in state:
        col[i] = 2 if i in used else 1
    return col


_RGB = {
    0: [1.0, 1.0, 1.0],          # white — not in library
    1: [0.60, 0.88, 0.60],       # light green — in library, unused
    2: [0.11, 0.37, 0.13],       # dark green — in library, used
}


def plot_trial_heatmap(trial_data: list, out_path: str, n_gates: int):
    n_trials = len(trial_data)

    # Build int matrix (n_gates, n_trials) — one column per iteration
    matrix = np.zeros((n_gates, n_trials), dtype=np.int8)
    costs = np.zeros(n_trials)
    best_costs = np.zeros(n_trials)

    for t, td in enumerate(trial_data):
        matrix[:, t] = td["col"]
        costs[t] = td["cost"]
        best_costs[t] = td["best_cost"]

    # RGB image array
    rgb = np.ones((n_gates, n_trials, 3), dtype=np.float32)
    for val, color in _RGB.items():
        mask = matrix == val
        rgb[mask] = color

    cell_w, cell_h = 0.15, 0.07
    fig_w = max(8, n_trials * cell_w + 2.0)
    fig_h = max(8, n_gates * cell_h + 3.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.08)

    # ── Top subplot: cost bars ───────────────────────────────────────────────
    ax_cost = fig.add_subplot(gs[0])

    xs = np.arange(n_trials)
    ax_cost.bar(xs, costs, width=0.6, color="#1B5E20")
    ax_cost.plot(xs, best_costs, color="orange", linewidth=1.0, zorder=3)
    ax_cost.axhline(1.0, color="red", linestyle="--", linewidth=0.8)

    ax_cost.set_xlim(-0.5, n_trials - 0.5)
    ax_cost.set_xticks([])
    ax_cost.set_ylabel("cost", fontsize=7)
    ax_cost.tick_params(axis="y", labelsize=6)

    # ── Bottom subplot: gate heatmap ─────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1])

    ax_heat.imshow(
        rgb,
        aspect="auto",
        interpolation="nearest",
        extent=[-0.5, n_trials - 0.5, n_gates - 0.5, -0.5],
    )

    for t in range(n_trials):
        ax_heat.axvline(t - 0.5, color="black", linewidth=0.4)

    # Red outlines on gates that changed in accepted moves
    for t, td in enumerate(trial_data):
        for g in td.get("removed", []) + td.get("added", []):
            ax_heat.add_patch(plt.Rectangle(
                (t - 0.5, g - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=0.8, zorder=3,
            ))

    ax_heat.set_xlim(-0.5, n_trials - 0.5)
    ax_heat.set_ylim(n_gates - 0.5, -0.5)

    ax_heat.set_xticks(np.arange(n_trials))
    ax_heat.set_xticklabels([f"T{t+1}" for t in range(n_trials)], fontsize=5, rotation=90)
    ax_heat.set_xlabel("Iteration", fontsize=8)
    ax_heat.set_ylabel("Gate index", fontsize=8)

    # Legend
    patches = [
        mpatches.Patch(color=_RGB[0], label="not in lib"),
        mpatches.Patch(color=_RGB[1], label="in lib unused"),
        mpatches.Patch(color=_RGB[2], label="in lib used"),
        mpatches.Patch(facecolor="none", edgecolor="red", label="removed / added"),
    ]
    ax_heat.legend(handles=patches, loc="lower right", fontsize=6,
                   framealpha=0.8, edgecolor="gray")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── MAIN ENGINE ─────────────────────────────────────────────────────────────

def main():
    args, cfg = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    genlib_path = _resolve_library(cfg, args.lib)
    bench_path = os.path.join(_ROOT, args.bench)
    bench_stem = os.path.splitext(os.path.basename(bench_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_sa_prune_{args.lib}_{bench_stem}"

    out_dir = os.path.join(_ROOT, args.out_dir, run_name)
    temp_blif = args.temp_blif if os.path.isabs(args.temp_blif) \
        else os.path.join(_ROOT, args.temp_blif)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(temp_blif), exist_ok=True)

    run_cfg = {**vars(args), "out_dir": out_dir, "genlib_path": genlib_path}
    logger = RunLogger(os.path.join(_ROOT, args.log_dir), run_name, run_cfg)

    logger.info(f"Initializing SA-prune for benchmark: {bench_path}")
    logger.info(f"Library: {args.lib} ({genlib_path})")
    logger.info(f"Action: remove up to {args.n_remove} unused gates, add {args.n_add} gates")

    mapper = TechMapper(genlib_path, bench_path, out_dir, temp_blif, area_mode=True)
    base_delay, base_area = mapper.baseline_delay, mapper.baseline_area
    logger.info(f"Baseline — Area: {base_area:.3f}, Delay: {base_delay:.3f} ps")
    logger.info(f"Total mutable gates: {mapper.num_arms}\n")

    if mapper.num_arms < args.n_select:
        raise ValueError(
            f"n_select ({args.n_select}) > available mutable gates ({mapper.num_arms})")

    # Initialise: sample a starting subset and map it to get cached usage
    current_state = set(random.sample(range(mapper.num_arms), args.n_select))
    curr_delay, curr_area, cached_used = mapper.map_subset_with_usage(
        sorted(current_state), tag="init")
    current_cost = calculate_cost(curr_delay, curr_area, base_delay, base_area)

    best_state, best_cost = set(current_state), current_cost
    best_area, best_delay = curr_area, curr_delay

    T = args.t0
    cooling_factor = (args.t_min / args.t0) ** (1.0 / args.iterations)

    logger.info(f"Starting SA-prune for {args.iterations} iterations ...")
    logger.info(
        f"{'Iter':>5} | {'Temp':>6} | {'Gates':>5} | {'Unused':>6} | "
        f"{'Rmv':>3} | {'Add':>3} | {'Area':>8} | {'Delay':>8} | "
        f"{'Cost':>8} | {'Best':>8} | Acc")
    logger.info("-" * 88)

    trial_data = []
    start_time = time.time()

    for step in range(1, args.iterations + 1):
        unused = current_state - cached_used

        # Build neighbor: either remove OR add
        available = set(range(mapper.num_arms)) - current_state
        remove_pool = current_state if args.random_remove else unused
        if remove_pool and (not available or random.random() < 0.5):
            to_remove = set(random.sample(list(remove_pool), min(args.n_remove, len(remove_pool))))
            to_add = set()
            neighbor_state = current_state - to_remove
        else:
            to_remove = set()
            to_add = set(random.sample(list(available), min(args.n_add, len(available))))
            neighbor_state = current_state | to_add

        if len(neighbor_state) == 0:
            n_delay, n_area = float("nan"), float("nan")
            n_used: set = set()
        else:
            n_delay, n_area, n_used = mapper.map_subset_with_usage(
                sorted(neighbor_state), tag=f"step{step}")

        neighbor_cost = calculate_cost(n_delay, n_area, base_delay, base_area)

        # SA acceptance; revert if rejected (current_state/cached_used unchanged)
        delta = neighbor_cost - current_cost
        if delta < 0:
            accepted = True
        else:
            prob = math.exp(-delta / T) if T > 0 else 0.0
            accepted = random.random() < prob

        if accepted:
            current_state, current_cost = neighbor_state, neighbor_cost
            curr_area, curr_delay = n_area, n_delay
            cached_used = n_used
            if current_cost < best_cost:
                best_state, best_cost = set(current_state), current_cost
                best_area, best_delay = curr_area, curr_delay

        T *= cooling_factor
        elapsed = time.time() - start_time

        # Record accepted state after SA decision
        trial_data.append({
            "cost":      current_cost,
            "best_cost": best_cost,
            "col":       _encode_col(current_state, cached_used, mapper.num_arms),
            "removed":   sorted(to_remove) if accepted else [],
            "added":     sorted(to_add)    if accepted else [],
        })

        logger.log_step({
            "step": step,
            "temperature": round(T, 6),
            "n_selected": len(current_state),
            "n_unused": len(unused),
            "n_removed": len(to_remove),
            "n_added": len(to_add),
            "area": round(curr_area, 4),
            "delay": round(curr_delay, 4),
            "cost": round(current_cost, 6),
            "best_cost": round(best_cost, 6),
            "accepted": 1 if accepted else 0,
            "elapsed_s": round(elapsed, 2),
        })

        if step % 10 == 0 or step == 1:
            acc_str = "Acc" if accepted else "Rej"
            logger.info(
                f"{step:5d} | {T:6.4f} | {len(current_state):5d} | {len(unused):6d} | "
                f"{len(to_remove):3d} | {len(to_add):3d} | {curr_area:8.2f} | "
                f"{curr_delay:8.2f} | {current_cost:8.4f} | {best_cost:8.4f} | {acc_str}")

    logger.info("\n" + "=" * 72)
    logger.info("Optimization Complete!")
    logger.info(f"Elapsed Time : {time.time() - start_time:.1f} seconds")
    logger.info(f"Baseline ADP : 1.0000 (Area: {base_area:.2f}, Delay: {base_delay:.2f})")
    logger.info(f"Best Cost    : {best_cost:.4f} (Area: {best_area:.2f}, Delay: {best_delay:.2f})")
    logger.info(f"ADP Improvement: {(1.0 - best_cost) * 100:+.2f}%")
    logger.info(f"\nBest Selected Gates ({len(best_state)}):")
    for idx in sorted(best_state):
        parts = mapper.mutable_gates[idx].split()
        logger.info(f"  [{idx:3d}] {parts[1]:<30s} area={parts[2]}")
    logger.info("=" * 72)

    # Save trial data for re-plotting
    trials_path = os.path.join(logger.run_dir, "trials.json")
    with open(trials_path, "w") as f:
        json.dump(trial_data, f)
    logger.info(f"Trial data saved to {trials_path}")

    # Generate heatmap
    heatmap_path = os.path.join(logger.run_dir, "heatmap.png")
    logger.info(f"Generating heatmap → {heatmap_path}")
    plot_trial_heatmap(trial_data, heatmap_path, mapper.num_arms)

    logger.close()


if __name__ == "__main__":
    main()
