"""
Parallel adaptive Simulated Annealing for gate library subset selection.

Launches N independent SA workers (multiprocessing), each with a different
seed and randomly chosen initial gate count. Each worker adapts its own
cooling factor during the run based on a sliding-window acceptance rate.
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
from collections import deque
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
            "step", "temperature", "n_selected", "area", "delay",
            "cost", "best_cost", "action", "accepted", "cooling_factor", "elapsed_s"
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
    default_bench = cfg.get("benchmarks", {}).get(
        "bench", ["benchmarks/c880.bench"])[0]
    paths = cfg.get("paths", {})

    p = argparse.ArgumentParser(description="Parallel adaptive SA for MapTune")
    p.add_argument("--lib", default=default_lib,
                   help="Library name from config.toml")
    p.add_argument("--bench", default=default_bench,
                   help="Path to .bench / .blif file")
    p.add_argument("--out-dir", default=paths.get("gen_newlibs_dir", "gen_newlibs/"),
                   help="Output dir for mapped libraries")
    p.add_argument("--log-dir", default="logs",
                   help="Base directory for run logs")

    p.add_argument("--n-agents", type=int, default=4,
                   help="Number of parallel SA workers")
    p.add_argument("--n-select-min", type=int, default=20,
                   help="Min initial gate count")
    p.add_argument("--n-select-max", type=int, default=80,
                   help="Max initial gate count")

    p.add_argument("--iterations", type=int, default=1500,
                   help="SA iterations per agent")
    p.add_argument("--t0", type=float, default=0.5, help="Initial temperature")
    p.add_argument("--t-min", type=float, default=0.001,
                   help="Final minimum temperature")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (agent i uses seed+i)")

    p.add_argument("--adapt-interval", type=int, default=100,
                   help="Steps between acceptance-rate checks for cooling adaptation")
    p.add_argument("--acc-low", type=float, default=0.10,
                   help="Acceptance rate below this → slow cooling")
    p.add_argument("--acc-high", type=float, default=0.40,
                   help="Acceptance rate above this → fast cooling")
    p.add_argument("--adapt-rate", type=float, default=0.05,
                   help="Fractional change to cooling factor per adaptation step")

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
        _ROOT, "temp_blifs", f"sa_multi_agent_{agent_id}.blif")
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

    n_init = random.randint(
        min(args.n_select_min, mapper.num_arms),
        min(args.n_select_max, mapper.num_arms)
    )
    logger.info(
        f"[Agent {agent_id}] Baseline Area={base_area:.3f}, Delay={base_delay:.3f} ps")
    logger.info(
        f"[Agent {agent_id}] Initial gates: {n_init} / {mapper.num_arms} available\n")

    current_state = set(random.sample(range(mapper.num_arms), n_init))
    curr_area, curr_delay = mapper.map_subset(
        sorted(current_state), tag="sa_init")
    current_cost = calculate_cost(curr_delay, curr_area, base_delay, base_area)

    best_state = set(current_state)
    best_cost, best_area, best_delay = current_cost, curr_area, curr_delay

    base_cf = (args.t_min / args.t0) ** (1.0 / args.iterations)
    cooling_factor = base_cf
    cf_min = base_cf ** 2
    cf_max = min(base_cf ** 0.5, 0.9999)

    T = args.t0
    recent = deque(maxlen=args.adapt_interval)
    start_time = time.time()

    logger.info(
        f"{'Iter':>5} | {'Temp':>7} | {'Gates':>5} | {'Area':>8} | {'Delay':>8} | {'Cost':>8} | {'CF':>8} | Action")
    logger.info("-" * 80)

    for step in range(1, args.iterations + 1):
        gate_to_toggle = random.randint(0, mapper.num_arms - 1)
        neighbor_state = set(current_state)

        if gate_to_toggle in neighbor_state:
            neighbor_state.remove(gate_to_toggle)
            action_str = f"Rem {gate_to_toggle}"
        else:
            neighbor_state.add(gate_to_toggle)
            action_str = f"Add {gate_to_toggle}"

        if len(neighbor_state) == 0:
            neighbor_cost = float('inf')
            n_area = n_delay = float('nan')
        else:
            n_area, n_delay = mapper.map_subset(
                sorted(neighbor_state), tag=f"sa_{step}")
            neighbor_cost = calculate_cost(
                n_delay, n_area, base_delay, base_area)

        delta_cost = neighbor_cost - current_cost
        if delta_cost < 0:
            accepted = True
        else:
            prob = math.exp(-delta_cost / T) if T > 0 else 0
            accepted = random.random() < prob

        if accepted:
            current_state, current_cost = neighbor_state, neighbor_cost
            curr_area, curr_delay = n_area, n_delay
            if current_cost < best_cost:
                best_state = set(current_state)
                best_cost, best_area, best_delay = current_cost, curr_area, curr_delay

        recent.append(1 if accepted else 0)

        # Adapt cooling factor every adapt_interval steps
        if step % args.adapt_interval == 0 and len(recent) == args.adapt_interval:
            acc_rate = sum(recent) / args.adapt_interval
            if acc_rate > args.acc_high:
                cooling_factor = min(
                    cooling_factor * (1 + args.adapt_rate), cf_max)
            elif acc_rate < args.acc_low:
                cooling_factor = max(
                    cooling_factor * (1 - args.adapt_rate), cf_min)

        T *= cooling_factor
        elapsed = time.time() - start_time

        logger.log_step({
            "step": step,
            "temperature": round(T, 6),
            "n_selected": len(current_state),
            "area": round(curr_area, 4),
            "delay": round(curr_delay, 4),
            "cost": round(current_cost, 6),
            "best_cost": round(best_cost, 6),
            "action": action_str,
            "accepted": 1 if accepted else 0,
            "cooling_factor": round(cooling_factor, 8),
            "elapsed_s": round(elapsed, 2),
        })

        if step % 100 == 0 or step == 1:
            acc_str = "Acc" if accepted else "Rej"
            logger.info(
                f"{step:5d} | {T:7.5f} | {len(current_state):5d} | {curr_area:8.2f} | "
                f"{curr_delay:8.2f} | {current_cost:8.4f} | {cooling_factor:.6f} | {acc_str} ({action_str})")

    improvement = (1.0 - best_cost) * 100
    logger.info(
        f"\n[Agent {agent_id}] Best cost: {best_cost:.4f} ({improvement:+.2f}% vs baseline)")
    logger.close()

    return {
        "agent_id": agent_id,
        "seed": seed,
        "n_init": n_init,
        "best_cost": best_cost,
        "best_area": best_area,
        "best_delay": best_delay,
        "best_state": sorted(best_state),
        "improvement_pct": round(improvement, 4),
        "elapsed_s": round(time.time() - start_time, 2),
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args, cfg = parse_args()

    bench_path = os.path.join(_ROOT, args.bench)
    bench_stem = os.path.splitext(os.path.basename(bench_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_base = f"{timestamp}_sa_multi_{args.lib}_{bench_stem}"

    print(
        f"\nStarting {args.n_agents} parallel SA agents — run: {run_name_base}")
    print(
        f"Iterations: {args.iterations} | T0: {args.t0} | T-min: {args.t_min}")
    print(
        f"Init gates: [{args.n_select_min}, {args.n_select_max}] | Adapt interval: {args.adapt_interval}\n")

    worker_args = [(i, args, cfg, run_name_base) for i in range(args.n_agents)]
    with Pool(processes=args.n_agents) as pool:
        results = pool.starmap(_worker, worker_args)

    best = min(results, key=lambda r: r["best_cost"])

    # Write summary
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

    # Cross-agent summary table
    print("\n" + "=" * 72)
    print(f"{'Agent':>5} | {'Seed':>6} | {'InitGates':>9} | {'BestCost':>9} | {'Improvement':>11} | {'Time(s)':>7}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["agent_id"]):
        marker = " *" if r["agent_id"] == best["agent_id"] else ""
        print(f"{r['agent_id']:5d} | {r['seed']:6d} | {r['n_init']:9d} | "
              f"{r['best_cost']:9.4f} | {r['improvement_pct']:+10.2f}% | {r['elapsed_s']:7.1f}{marker}")
    print("=" * 72)
    print(
        f"\nGlobal best: Agent {best['agent_id']} — cost {best['best_cost']:.4f} ({best['improvement_pct']:+.2f}%)")
    print(f"Summary saved to: {summary_dir}/summary.json\n")


if __name__ == "__main__":
    main()
