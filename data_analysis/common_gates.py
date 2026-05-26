"""
plot_gate_usage.py

Auto mode (default): scans log_path for subdirs matching `sa_multi_7nm_*`,
groups them by design name, and saves one PNG per design into log_path.

Manual mode (--out specified): collects ALL sa_multi_7nm_* summary.json
under log_path into a single plot saved to --out.

Usage:
  # Auto — one plot per design, saved next to the run dirs
  python plot_gate_usage.py /path/to/logs/

  # Manual — single aggregated plot
  python plot_gate_usage.py /path/to/logs/ --out combined.png

  # Different library size
  python plot_gate_usage.py /path/to/logs/ --n-gates 161
"""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Only process dirs that match this pattern (ignore random_* and sky130_*)
DIR_PATTERN = re.compile(r"^\d{8}_\d{6}_sa_multi_7nm_(.+)$")


def load_best_state(summary_path: Path) -> "list[int]":
    """Return the best_state of the best_agent from a summary.json."""
    with open(summary_path) as fp:
        data = json.load(fp)
    best_id = data.get("best_agent")
    for agent in data.get("agents", []):
        if agent.get("agent_id") == best_id:
            return agent.get("best_state", [])
    return []


def collect_counts(json_files: list[Path], n_gates: int) -> np.ndarray:
    counts = np.zeros(n_gates, dtype=int)
    for f in json_files:
        for idx in load_best_state(f):
            if 0 <= idx < n_gates:
                counts[idx] += 1
    return counts


def plot(counts: np.ndarray, title: str, out_path: Path, n_runs: int):
    n = len(counts)
    x = np.arange(n)
    max_count = counts.max() if counts.max() > 0 else 1

    norm = mcolors.Normalize(vmin=0, vmax=max_count)
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x, counts, width=0.8, color=cmap(norm(counts)), linewidth=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=30)
    cbar.set_label("Times selected", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    tick_positions = np.arange(0, n, 10)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=8)

    ax.set_xlabel("Gate index", fontsize=12)
    ax.set_ylabel("Times selected", fontsize=12)
    ax.set_title(f"Gate Usage Rate — {title}  (n_runs={n_runs})", fontsize=14)
    ax.set_xlim(-1, n)
    ax.set_ylim(0, max_count + 1)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    n_used = int((counts > 0).sum())
    ax.text(
        0.99, 0.97,
        f"Gates with ≥1 selection: {n_used} / {n}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved → {out_path}")
    plt.close(fig)


def find_sa_7nm_dirs(log_path: Path) -> dict[str, list[Path]]:
    """Return {design_name: [summary.json, ...]} for all sa_multi_7nm_* dirs."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(log_path.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_PATTERN.match(d.name)
        if not m:
            continue
        design = m.group(1)  # e.g. "c880", "b10"
        summary = d / "summary.json"
        if summary.exists():
            groups[design].append(summary)
        else:
            print(f"  [warn] no summary.json in {d.name}, skipping")
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Plot gate usage rate from SA logs")
    parser.add_argument("log_path", type=Path,
                        help="Root directory containing sa_multi_7nm_* run dirs")
    parser.add_argument("--n-gates", type=int, default=161,
                        help="Total gates in library (default: 161)")
    parser.add_argument("--out", type=Path, default=None,
                        help="If set, aggregate all designs into one plot saved here")
    args = parser.parse_args()

    groups = find_sa_7nm_dirs(args.log_path)
    if not groups:
        raise FileNotFoundError(
            f"No sa_multi_7nm_* dirs with summary.json found under {args.log_path}"
        )

    if args.out:
        # Manual mode: single aggregated plot
        all_files = [f for files in groups.values() for f in files]
        print(
            f"Aggregating {len(all_files)} run(s) across designs: {sorted(groups)}")
        counts = collect_counts(all_files, args.n_gates)
        plot(counts, "all designs", args.out, n_runs=len(all_files))
        _print_top10(counts)
    else:
        # Auto mode: one plot per design
        print(f"Found {len(groups)} design(s): {sorted(groups.keys())}")
        for design, files in sorted(groups.items()):
            print(f"\n[{design}]  {len(files)} run(s)")
            counts = collect_counts(files, args.n_gates)
            out_path = args.log_path / f"gate_usage_7nm_{design}.png"
            plot(counts, f"7nm / {design}", out_path, n_runs=len(files))
            _print_top10(counts)


def _print_top10(counts: np.ndarray):
    top10 = np.argsort(counts)[::-1][:10]
    print("  Top-10:", "  ".join(f"{idx}({counts[idx]})" for idx in top10))


if __name__ == "__main__":
    main()
