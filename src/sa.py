"""
Simulated Annealing agent for gate library subset selection.

Maintains a variable-size subset of library cell types to optimize area/delay after
technology mapping. Starts with 50 gates and toggles one random gate per iteration.
Includes robust logging (CSV + JSON config + Log file).
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

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli for Python < 3.11

from abc_mapper import TechMapper

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── LOGGING MANAGER ─────────────────────────────────────────────────────────

class RunLogger:
    """Manages run directories, CSV metrics, and standard logging."""
    def __init__(self, base_dir: str, run_name: str, args_dict: dict):
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Setup file + console logging
        log_file = os.path.join(self.run_dir, "run.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s", # Keeping it clean for terminal readability
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Persist run configuration
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(args_dict, f, indent=2)
            
        # Setup CSV tracker
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.fieldnames = [
            "step", "temperature", "n_selected", "area", "delay", 
            "cost", "best_cost", "action", "accepted", "elapsed_s"
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
    
    p = argparse.ArgumentParser(description="Simulated Annealing for MapTune")
    p.add_argument("--lib", default=default_lib, help="Library name from config.toml")
    p.add_argument("--bench", default=default_bench, help="Path to .bench / .blif file")
    p.add_argument("--out-dir", default=paths.get("gen_newlibs_dir", "gen_newlibs/"), help="Output dir for mapped libraries")
    p.add_argument("--temp-blif", default=os.path.join(paths.get("temp_blifs_dir", "temp_blifs"), "sa_temp.blif"), help="Temporary blif file")
    p.add_argument("--log-dir", default="logs", help="Base directory for run logs")
    
    p.add_argument("--n-select", type=int, default=50, help="Initial number of selected gates")
    p.add_argument("--iterations", type=int, default=1500, help="Total SA iterations")
    p.add_argument("--t0", type=float, default=0.5, help="Initial temperature")
    p.add_argument("--t-min", type=float, default=0.001, help="Final minimum temperature")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args(), cfg

def calculate_cost(delay, area, base_delay, base_area):
    """Normalized Area-Delay Product (ADP). Lower is better."""
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
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.temp_blif), exist_ok=True)
    
    # Init Logger
    # Init Logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_sa_{args.lib}_{bench_stem}"
    logger = RunLogger(os.path.join(_ROOT, args.log_dir), run_name, vars(args))
    
    logger.info(f"Initializing SA for benchmark: {bench_path}")
    logger.info(f"Library: {args.lib} ({genlib_path})")
    
    mapper = TechMapper(genlib_path, bench_path, args.out_dir, args.temp_blif, area_mode=True)
    
    base_delay, base_area = mapper.baseline_delay, mapper.baseline_area
    logger.info(f"Baseline - Area: {base_area:.3f}, Delay: {base_delay:.3f} ps")
    logger.info(f"Total mutable gates available: {mapper.num_arms}\n")
    
    if mapper.num_arms < args.n_select:
        raise ValueError(f"Initial selection size ({args.n_select}) exceeds available mutable gates ({mapper.num_arms}).")

    current_state = set(random.sample(range(mapper.num_arms), args.n_select))
    curr_area, curr_delay = mapper.map_subset(sorted(list(current_state)), tag="sa_init")
    current_cost = calculate_cost(curr_delay, curr_area, base_delay, base_area)
    
    best_state, best_cost = set(current_state), current_cost
    best_area, best_delay = curr_area, curr_delay
    
    T = args.t0
    cooling_factor = (args.t_min / args.t0) ** (1.0 / args.iterations)
    
    logger.info(f"Starting SA for {args.iterations} iterations...")
    logger.info(f"{'Iter':>5} | {'Temp':>6} | {'Gates':>5} | {'Area':>8} | {'Delay':>8} | {'Cost':>8} | {'Action'}")
    logger.info("-" * 72)

    start_time = time.time()

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
        else:
            n_area, n_delay = mapper.map_subset(sorted(list(neighbor_state)), tag=f"sa_{step}")
            neighbor_cost = calculate_cost(n_delay, n_area, base_delay, base_area)

        delta_cost = neighbor_cost - current_cost
        accepted = False
        
        if delta_cost < 0:
            accepted = True
        else:
            prob = math.exp(-delta_cost / T) if T > 0 else 0
            if random.random() < prob:
                accepted = True
                
        if accepted:
            current_state, current_cost = neighbor_state, neighbor_cost
            curr_area, curr_delay = n_area, n_delay
            
            if current_cost < best_cost:
                best_state, best_cost = set(current_state), current_cost
                best_area, best_delay = curr_area, curr_delay

        T *= cooling_factor
        elapsed = time.time() - start_time
        
        # Push to CSV
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
            "elapsed_s": round(elapsed, 2)
        })

        if step % 100 == 0 or step == 1:
            acc_str = "Acc" if accepted else "Rej"
            logger.info(f"{step:5d} | {T:6.4f} | {len(current_state):5d} | {curr_area:8.2f} | {curr_delay:8.2f} | {current_cost:8.4f} | {acc_str} ({action_str})")

    logger.info("\n" + "=" * 72)
    logger.info("Optimization Complete!")
    logger.info(f"Elapsed Time : {time.time() - start_time:.1f} seconds")
    logger.info(f"Baseline ADP : 1.0000 (Area: {base_area:.2f}, Delay: {base_delay:.2f})")
    logger.info(f"Best SA Cost : {best_cost:.4f} (Area: {best_area:.2f}, Delay: {best_delay:.2f})")
    
    improvement = (1.0 - best_cost) * 100
    logger.info(f"ADP Improvement: {improvement:+.2f}%")
    
    logger.info(f"\nBest Selected Gates ({len(best_state)}):")
    for idx in sorted(list(best_state)):
        parts = mapper.mutable_gates[idx].split()
        name, area_val = parts[1], parts[2]
        logger.info(f"  [{idx:3d}] {name:<30s} area={area_val}")
    logger.info("=" * 72)
    
    logger.close()

if __name__ == "__main__":
    main()