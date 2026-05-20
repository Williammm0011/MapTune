"""
Random baseline agent for gate library subset selection.

At each iteration picks a uniformly random number of gates (1..num_arms) and
evaluates the mapping. Logs follow the same format as sa.py for comparison.
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
            "step", "n_selected", "area", "delay", "cost", "best_cost", "elapsed_s"
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
    rand = cfg.get("random", {})

    p = argparse.ArgumentParser(description="Random baseline agent for MapTune")
    p.add_argument("--lib", default=default_lib, help="Library name from config.toml")
    p.add_argument("--bench", default=default_bench, help="Path to .bench / .blif file")
    p.add_argument("--out-dir", default=paths.get("gen_newlibs_dir", "gen_newlibs/"),
                   help="Output dir for mapped libraries")
    p.add_argument("--temp-blif", default=os.path.join(
                   paths.get("temp_blifs_dir", "temp_blifs"), "random_temp.blif"),
                   help="Temporary blif file")
    p.add_argument("--log-dir", default=paths.get("log_dir", "logs"),
                   help="Base directory for run logs")
    p.add_argument("--iterations", type=int, default=rand.get("iterations", 1500),
                   help="Number of random trials")
    p.add_argument("--seed", type=int, default=rand.get("seed", 42),
                   help="Random seed")
    return p.parse_args(), cfg


def calculate_cost(delay, area, base_delay, base_area):
    if math.isnan(delay) or math.isnan(area) or delay <= 0 or area <= 0:
        return float('inf')
    return (delay / base_delay) * (area / base_area)


# ── MAIN ENGINE ─────────────────────────────────────────────────────────────

def main():
    args, cfg = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    genlib_path = _resolve_library(cfg, args.lib)
    bench_path = os.path.join(_ROOT, args.bench)
    bench_stem = os.path.splitext(os.path.basename(bench_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_random_{args.lib}_{bench_stem}"

    out_dir = os.path.join(_ROOT, args.out_dir, run_name)
    temp_blif = args.temp_blif if os.path.isabs(args.temp_blif) else os.path.join(_ROOT, args.temp_blif)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(temp_blif), exist_ok=True)

    run_cfg = {**vars(args), "out_dir": out_dir, "genlib_path": genlib_path}
    logger = RunLogger(os.path.join(_ROOT, args.log_dir), run_name, run_cfg)

    logger.info(f"Initializing Random agent for benchmark: {bench_path}")
    logger.info(f"Library: {args.lib} ({genlib_path})")

    mapper = TechMapper(genlib_path, bench_path, out_dir, temp_blif, area_mode=True)

    base_delay, base_area = mapper.baseline_delay, mapper.baseline_area
    logger.info(f"Baseline - Area: {base_area:.3f}, Delay: {base_delay:.3f} ps")
    logger.info(f"Total mutable gates available: {mapper.num_arms}\n")

    best_cost = float('inf')
    best_area = best_delay = float('nan')
    best_state = []

    logger.info(f"Starting random search for {args.iterations} iterations...")
    logger.info(f"{'Iter':>5} | {'Gates':>5} | {'Area':>8} | {'Delay':>8} | {'Cost':>8} | {'Best':>8}")
    logger.info("-" * 62)

    start_time = time.time()

    for step in range(1, args.iterations + 1):
        n = random.randint(1, mapper.num_arms)
        state = sorted(random.sample(range(mapper.num_arms), n))

        r_area, r_delay = mapper.map_subset(state, tag=f"random_{step}")
        cost = calculate_cost(r_delay, r_area, base_delay, base_area)

        if cost < best_cost:
            best_cost = cost
            best_area, best_delay = r_area, r_delay
            best_state = state

        elapsed = time.time() - start_time

        logger.log_step({
            "step": step,
            "n_selected": n,
            "area": round(r_area, 4),
            "delay": round(r_delay, 4),
            "cost": round(cost, 6),
            "best_cost": round(best_cost, 6),
            "elapsed_s": round(elapsed, 2),
        })

        if step % 100 == 0 or step == 1:
            logger.info(
                f"{step:5d} | {n:5d} | {r_area:8.2f} | {r_delay:8.2f} | {cost:8.4f} | {best_cost:8.4f}")

    logger.info("\n" + "=" * 62)
    logger.info("Random Search Complete!")
    logger.info(f"Elapsed Time : {time.time() - start_time:.1f} seconds")
    logger.info(f"Baseline ADP : 1.0000 (Area: {base_area:.2f}, Delay: {base_delay:.2f})")
    logger.info(f"Best Cost    : {best_cost:.4f} (Area: {best_area:.2f}, Delay: {best_delay:.2f})")

    improvement = (1.0 - best_cost) * 100
    logger.info(f"ADP Improvement: {improvement:+.2f}%")

    logger.info(f"\nBest Selected Gates ({len(best_state)}):")
    for idx in best_state:
        parts = mapper.mutable_gates[idx].split()
        name, area_val = parts[1], parts[2]
        logger.info(f"  [{idx:3d}] {name:<30s} area={area_val}")
    logger.info("=" * 62)

    logger.close()


if __name__ == "__main__":
    main()
