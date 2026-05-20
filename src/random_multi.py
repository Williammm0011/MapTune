"""
Parallel random baseline for gate library subset selection.

Launches N independent random-search workers (multiprocessing), each with a
different seed. Each worker samples uniformly random gate subsets every
iteration and tracks the best seen — no temperature or acceptance logic.
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
from multiprocessing import Pool

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
    raise ValueError(
        f"Library '{lib_name}' not found in config.toml [[library]] entries.")


def calculate_cost(delay, area, base_delay, base_area):
    if math.isnan(delay) or math.isnan(area) or delay <= 0 or area <= 0:
        return float('inf')
    return (delay / base_delay) * (area / base_area)


def parse_args():
    cfg = _load_config()

    default_lib = cfg.get("library", [{"name": "7nm"}])[0]["name"]
    default_bench = cfg.get("benchmarks", {}).get("bench", ["benchmarks/c880.bench"])[0]
    paths = cfg.get("paths", {})
    rm = cfg.get("random_multi", {})

    p = argparse.ArgumentParser(description="Parallel random baseline for MapTune")
    p.add_argument("--lib", default=default_lib,
                   help="Library name from config.toml")
    p.add_argument("--bench", default=default_bench,
                   help="Path to .bench / .blif file")
    p.add_argument("--out-dir", default=paths.get("gen_newlibs_dir", "gen_newlibs/"),
                   help="Output dir for mapped libraries")
    p.add_argument("--log-dir", default=paths.get("log_dir", "logs"),
                   help="Base directory for run logs")

    p.add_argument("--n-agents", type=int, default=rm.get("n_agents", 4),
                   help="Number of parallel random workers")
    p.add_argument("--iterations", type=int, default=rm.get("iterations", 3000),
                   help="Random trials per agent")
    p.add_argument("--seed", type=int, default=rm.get("seed", 42),
                   help="Base random seed (agent i uses seed+i)")

    return p.parse_args(), cfg


# ── WORKER ───────────────────────────────────────────────────────────────────

def _worker(agent_id: int, args, cfg: dict, run_name_base: str) -> dict:
    seed = args.seed + agent_id
    random.seed(seed)
    np.random.seed(seed)

    genlib_path = _resolve_library(cfg, args.lib)
    bench_path = os.path.join(_ROOT, args.bench)
    bench_stem = os.path.splitext(os.path.basename(bench_path))[0]

    out_dir = os.path.join(_ROOT, args.out_dir,
                           run_name_base, f"agent_{agent_id}")
    temp_blif = os.path.join(
        _ROOT, "temp_blifs", f"random_multi_agent_{agent_id}.blif")
    log_subdir = os.path.join(_ROOT, args.log_dir, run_name_base)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(temp_blif), exist_ok=True)

    run_cfg = {
        **vars(args),
        "agent_id": agent_id,
        "seed": seed,
        "out_dir": out_dir,
        "genlib_path": genlib_path,
    }
    logger = RunLogger(log_subdir, f"agent_{agent_id}", run_cfg)

    logger.info(
        f"[Agent {agent_id}] seed={seed}, bench={bench_stem}, lib={args.lib}")

    mapper = TechMapper(genlib_path, bench_path, out_dir,
                        temp_blif, area_mode=True)
    base_delay, base_area = mapper.baseline_delay, mapper.baseline_area

    logger.info(
        f"[Agent {agent_id}] Baseline Area={base_area:.3f}, Delay={base_delay:.3f} ps")
    logger.info(
        f"[Agent {agent_id}] Available gates: {mapper.num_arms}\n")

    best_cost = float('inf')
    best_area = best_delay = float('nan')
    best_state = []

    logger.info(
        f"{'Iter':>5} | {'Gates':>5} | {'Area':>8} | {'Delay':>8} | {'Cost':>8} | {'Best':>8}")
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
                f"{step:5d} | {n:5d} | {r_area:8.2f} | {r_delay:8.2f} | "
                f"{cost:8.4f} | {best_cost:8.4f}")

    improvement = (1.0 - best_cost) * 100
    logger.info(
        f"\n[Agent {agent_id}] Best cost: {best_cost:.4f} ({improvement:+.2f}% vs baseline)")
    logger.close()

    return {
        "agent_id": agent_id,
        "seed": seed,
        "best_cost": best_cost,
        "best_area": best_area,
        "best_delay": best_delay,
        "best_state": best_state,
        "improvement_pct": round(improvement, 4),
        "elapsed_s": round(time.time() - start_time, 2),
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args, cfg = parse_args()

    bench_path = os.path.join(_ROOT, args.bench)
    bench_stem = os.path.splitext(os.path.basename(bench_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_base = f"{timestamp}_random_multi_{args.lib}_{bench_stem}"

    print(
        f"\nStarting {args.n_agents} parallel random agents — run: {run_name_base}")
    print(
        f"Iterations: {args.iterations} | Seed base: {args.seed}\n")

    worker_args = [(i, args, cfg, run_name_base) for i in range(args.n_agents)]
    with Pool(processes=args.n_agents) as pool:
        results = pool.starmap(_worker, worker_args)

    best = min(results, key=lambda r: r["best_cost"])

    summary_dir = os.path.join(_ROOT, args.log_dir, run_name_base)
    os.makedirs(summary_dir, exist_ok=True)
    summary = {
        "run_name": run_name_base,
        "n_agents": args.n_agents,
        "best_agent": best["agent_id"],
        "best_cost": best["best_cost"],
        "best_improvement_pct": best["improvement_pct"],
        "agents": results,
    }
    with open(os.path.join(summary_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"{'Agent':>5} | {'Seed':>6} | {'BestCost':>9} | {'Improvement':>11} | {'Time(s)':>7}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["agent_id"]):
        marker = " *" if r["agent_id"] == best["agent_id"] else ""
        print(f"{r['agent_id']:5d} | {r['seed']:6d} | "
              f"{r['best_cost']:9.4f} | {r['improvement_pct']:+10.2f}% | "
              f"{r['elapsed_s']:7.1f}{marker}")
    print("=" * 64)
    print(
        f"\nGlobal best: Agent {best['agent_id']} — cost {best['best_cost']:.4f} ({best['improvement_pct']:+.2f}%)")
    print(f"Summary saved to: {summary_dir}/summary.json\n")


if __name__ == "__main__":
    main()
