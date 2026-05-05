# random_search.py — Random gate-subset search baseline
#
# Usage:
#   python random_search.py <num_sampled_gate> <design> <genlib>
#
#   num_sampled_gate  number of gates randomly sampled per iteration
#   design            path to benchmark (e.g. benchmarks/s838a.bench)
#   genlib            path to cell library  (e.g. sky130.genlib)
#
# Example:
#   python random_search.py 65 benchmarks/s838a.bench sky130.genlib
#
# Must be run from the repo root so relative paths resolve correctly.

import random
import sys
import numpy as np
import subprocess
import re
import time
genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
temp_blif = "temp_blifs/" + design[:-5] + "_random_temp.blif"
lib_path = "gen_newlibs/"

num_iterations = 10000

# ── parse candidate gates (exclude BUF/INV, which are always kept) ────────────
BUF_INV_PREFIXES = (
    "GATE BUF", "GATE INV",
    "GATE sky130_fd_sc_hd__buf", "GATE sky130_fd_sc_hd__inv",
    "GATE gf180mcu_fd_sc_mcu7t5v0__buf", "GATE gf180mcu_fd_sc_mcu7t5v0__inv",
)


def parse_genlib(path):
    candidates, keep = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("GATE"):
                continue
            if any(line.startswith(p) for p in BUF_INV_PREFIXES):
                keep.append(line)
            else:
                candidates.append(line)
    return candidates, keep

# ── abc call ──────────────────────────────────────────────────────────────────


def run_abc(genlib_file):
    cmd = ("read %s;read %s; map -a; write %s; read %s;read -m %s; "
           "ps; topo; upsize; dnsize; stime; "
           % (genlib_file, design, temp_blif, lib_origin, temp_blif))
    res = subprocess.check_output(('abc', '-c', cmd))
    m_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    m_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    if m_d and m_a:
        return float(m_d.group(1)), float(m_a.group(1))
    return float("nan"), float("nan")


def technology_mapper(selected_indices, candidates, keep, tag):
    lines = [candidates[i] for i in selected_indices] + keep
    out_genlib = lib_path + design + "_" + \
        str(len(lines)) + "_" + tag + "_samplelib.genlib"
    with open(out_genlib, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return run_abc(out_genlib)


def calculate_reward(delay, area, max_delay, max_area):
    return -np.sqrt((delay / max_delay) * (area / max_area))


# ── baseline ──────────────────────────────────────────────────────────────────
start = time.time()
print("Running baseline ABC...", flush=True)
max_delay, max_area = run_abc(genlib_origin)
print(f"  {'Baseline delay':<16}: {max_delay:.1f} ps")
print(f"  {'Baseline area':<16}: {max_area:.1f}")

candidates, keep = parse_genlib(genlib_origin)
num_candidates = len(candidates)
print(f"  {'Candidate gates':<16}: {num_candidates}")
print(f"  {'Sampled gates':<16}: {sample_gate}")
print(f"  {'Iterations':<16}: {num_iterations}")

# ── random search ─────────────────────────────────────────────────────────────
best_reward = -float('inf')
best_result = (float('inf'), float('inf'))
best_cells = None

for i in range(num_iterations):
    selected = random.sample(range(num_candidates), sample_gate)
    delay, area = technology_mapper(selected, candidates, keep, "random")

    if np.isnan(delay) or np.isnan(area):
        reward = -float('inf')
    else:
        reward = calculate_reward(delay, area, max_delay, max_area)

    if reward > best_reward:
        best_reward = reward
        best_result = (delay, area)
        best_cells = selected

    print(f"  Iter {i+1:4d}/{num_iterations} | reward={reward:8.4f} | delay={delay:10.1f} ps | area={area:10.1f} | best={best_reward:8.4f}")
    with open(".random_progress", "w") as _pf:
        _pf.write(f"{i+1}/{num_iterations}")

# ── summary ───────────────────────────────────────────────────────────────────
SEP = "=" * 56
print(SEP)
print(f"  {'Genlib':<16}: {genlib_origin}")
print(f"  {'Design':<16}: {design}")
print(f"  {'Sampled gates':<16}: {sample_gate}")
print(f"  {'Iterations':<16}: {num_iterations}")
print(f"  {'Baseline delay':<16}: {max_delay:.1f} ps")
print(f"  {'Baseline area':<16}: {max_area:.1f}")
print(f"  {'Best delay':<16}: {best_result[0]:.1f} ps")
print(f"  {'Best area':<16}: {best_result[1]:.1f}")
print(f"  {'Best reward':<16}: {best_reward:.4f}")
print(f"  {'Total time':<16}: {time.time() - start:.1f} s")
print(SEP)
