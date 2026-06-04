"""Batch runner for GradMap benchmarks.

Runs GradMapper over the configured benchmark list and writes structured logs
under logs/{timestamp}_gradmap_batch/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 backport

_SRC = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.gradmap_mapper import GradMapper


SUMMARY_FIELDS = [
    "bench",
    "status",
    "baseline_delay",
    "baseline_area",
    "best_delay",
    "best_area",
    "best_adp",
    "reward",
    "used_cells",
    "elapsed_sec",
]


def load_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GradMap over benchmark batch")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmark names from config.toml. Defaults to the first 10 gradmap benchmarks.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optimizer steps. Defaults to [gradmap].default_steps.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Base log directory relative to the repository root.",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Existing batch log directory to resume. Completed ok result.json files are reused.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed benchmark.",
    )
    parser.add_argument(
        "--heatmap-top-gates",
        type=int,
        default=60,
        help="Maximum number of gate rows to show in used_gate_heatmap.png. Use 0 for all gates.",
    )
    return parser.parse_args()


def fmt_metric(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def csv_value(value: float | int | str | None) -> float | int | str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def run_one(bench: str, cfg: dict[str, Any], steps: int, bench_dir: str) -> dict[str, Any]:
    os.makedirs(bench_dir, exist_ok=True)
    mapper: GradMapper | None = None
    start = time.time()

    try:
        mapper = GradMapper(bench, cfg, steps=steps)
        delay, area = mapper.run(verbose=False, progress=True)
        reward = mapper.calculate_reward(delay, area)
        elapsed = round(time.time() - start, 2)
        result = {
            "bench": mapper.bench_stem,
            "status": "ok",
            "steps": steps,
            "baseline_delay": mapper.baseline_delay,
            "baseline_area": mapper.baseline_area,
            "best_delay": delay,
            "best_area": area,
            "best_adp": mapper._best_adp,
            "reward": reward,
            "used_cells": sum(mapper.used_cells.values()),
            "used_cell_counts": dict(mapper.used_cells.most_common()),
            "elapsed_sec": elapsed,
        }
        print(
            "        done  "
            f"reward={fmt_metric(reward)}  "
            f"best delay={fmt_metric(delay, 2)}  "
            f"area={fmt_metric(area, 4)}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        result = {
            "bench": bench,
            "status": "error",
            "steps": steps,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": elapsed,
        }
        print(f"        error after {elapsed:.1f}s: {exc}", flush=True)

    output = mapper.last_output if mapper is not None else ""
    write_text(os.path.join(bench_dir, "run.log"), output)
    write_json(os.path.join(bench_dir, "result.json"), result)
    return result


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    return {field: csv_value(result.get(field)) for field in SUMMARY_FIELDS}


def plot_used_gate_heatmap(
    results: list[dict[str, Any]],
    benchmarks: list[str],
    out_path: str,
    top_gates: int,
) -> str | None:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return None

    bench_to_counts = {
        result["bench"]: result.get("used_cell_counts", {})
        for result in successful
    }
    plotted_benches = [bench for bench in benchmarks if bench in bench_to_counts]
    gate_totals: dict[str, int] = {}
    for counts in bench_to_counts.values():
        for gate, count in counts.items():
            gate_totals[gate] = gate_totals.get(gate, 0) + int(count)

    gates = sorted(gate_totals, key=lambda gate: (-gate_totals[gate], gate))
    if top_gates > 0:
        gates = gates[:top_gates]

    if not gates or not plotted_benches:
        return None

    matrix = np.array(
        [
            [bench_to_counts[bench].get(gate, 0) for bench in plotted_benches]
            for gate in gates
        ],
        dtype=float,
    )
    max_count = max(1.0, float(matrix.max()))

    fig_w = max(10.0, 3.8 + 1.05 * len(plotted_benches))
    fig_h = max(6.0, 2.4 + 0.34 * len(gates))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#fbfdfb")
    ax.set_facecolor("#ffffff")

    cmap = LinearSegmentedColormap.from_list(
        "gate_usage_greens",
        ["#ffffff", "#d8f3dc", "#74c69d", "#2d6a4f", "#081c15"],
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max_count, aspect="auto")

    ax.set_title(
        "GradMap Used Gate Counts",
        fontsize=18,
        fontweight="bold",
        pad=18,
        color="#102a18",
    )
    ax.set_xlabel("Testbench", fontsize=12, labelpad=12, color="#274734")
    ax.set_ylabel("Gate", fontsize=12, labelpad=12, color="#274734")
    ax.set_xticks(np.arange(len(plotted_benches)))
    ax.set_xticklabels(plotted_benches, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(gates)))
    ax.set_yticklabels(gates, fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(plotted_benches), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(gates), 1), minor=True)
    ax.grid(which="minor", color="#e6efe8", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0, colors="#274734")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx, gate in enumerate(gates):
        for col_idx, bench in enumerate(plotted_benches):
            value = int(bench_to_counts[bench].get(gate, 0))
            if value == 0:
                continue
            text_color = "#ffffff" if value >= max_count * 0.55 else "#0b2e13"
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    colorbar.set_label("Used count", rotation=270, labelpad=16, color="#274734")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors="#274734")

    subtitle = (
        f"Showing top {len(gates)} gates by total usage"
        if top_gates > 0
        else f"Showing all {len(gates)} used gates"
    )
    fig.text(0.01, 0.01, subtitle, fontsize=9, color="#5b6f60")
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    cfg_path = os.path.join(_REPO_ROOT, "config.toml")
    cfg = load_config(cfg_path)

    available = cfg["benchmarks"]["gradmap"]
    benchmarks = args.benchmarks if args.benchmarks is not None else available[:10]
    unknown = [bench for bench in benchmarks if bench not in available]
    if unknown:
        raise SystemExit(
            "Unknown gradmap benchmark(s): "
            + ", ".join(unknown)
            + "\nAvailable: "
            + ", ".join(available)
        )

    steps = args.steps if args.steps is not None else cfg["gradmap"]["default_steps"]
    if args.resume_dir is not None:
        batch_dir = args.resume_dir
        if not os.path.isabs(batch_dir):
            batch_dir = os.path.join(_REPO_ROOT, batch_dir)
        batch_name = os.path.basename(os.path.normpath(batch_dir))
        suffix = "_gradmap_batch"
        timestamp = batch_name[:-len(suffix)] if batch_name.endswith(suffix) else batch_name
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"{timestamp}_gradmap_batch"
        batch_dir = os.path.join(_REPO_ROOT, args.log_dir, batch_name)
    os.makedirs(batch_dir, exist_ok=True)

    metadata = {
        "run_name": batch_name,
        "timestamp": timestamp,
        "steps": steps,
        "benchmarks": benchmarks,
        "heatmap_top_gates": args.heatmap_top_gates,
        "used_gate_heatmap": "used_gate_heatmap.png",
        "gradmap": cfg["gradmap"],
    }
    write_json(os.path.join(batch_dir, "config.json"), metadata)

    rel_batch_dir = os.path.relpath(batch_dir, _REPO_ROOT)
    print(f"Gradmap batch: {len(benchmarks)} benchmarks, {steps} steps")
    print(f"Logs: {rel_batch_dir}/")

    summary_path = os.path.join(batch_dir, "summary.csv")
    failures = 0
    results = []
    with open(summary_path, "w", newline="") as summary_f:
        writer = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for idx, bench in enumerate(benchmarks, 1):
            print(f"[{idx:>2}/{len(benchmarks)}] {bench:<12}", flush=True)
            bench_dir = os.path.join(batch_dir, bench)
            result_path = os.path.join(bench_dir, "result.json")
            if args.resume_dir is not None and os.path.isfile(result_path):
                existing_result = read_json(result_path)
                if existing_result.get("status") == "ok":
                    result = existing_result
                    print("        reused existing result", flush=True)
                else:
                    result = run_one(bench, cfg, steps, bench_dir)
            else:
                result = run_one(bench, cfg, steps, bench_dir)
            results.append(result)
            writer.writerow(summary_row(result))
            summary_f.flush()
            if result["status"] != "ok":
                failures += 1
                if args.fail_fast:
                    break

    heatmap_path = os.path.join(batch_dir, "used_gate_heatmap.png")
    plotted_path = plot_used_gate_heatmap(
        results,
        benchmarks,
        heatmap_path,
        args.heatmap_top_gates,
    )

    rel_summary = os.path.relpath(summary_path, _REPO_ROOT)
    print(f"Batch done. summary -> {rel_summary}")
    if plotted_path:
        rel_heatmap = os.path.relpath(plotted_path, _REPO_ROOT)
        print(f"Used-gate heatmap -> {rel_heatmap}")
    else:
        print("Used-gate heatmap skipped: no successful benchmark results")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
