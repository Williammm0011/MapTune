"""Run GradMap technology mapping on a benchmark and report metrics.

Usage:
  python quick_map_gradmap.py [bench_path] [--steps N]

Defaults:
  bench_path  benchmarks/c880.bench
  steps       200

Pipeline:
  1. Parse asap7_libcell_info.txt for cell data
  2. Convert .bench -> match file (Python, no patched ABC required)
  3. Write gradmap config
  4. Run gradmap_torch
  5. Print best delay / area / ADP
"""

import argparse
import collections
import os
import re
import subprocess
import sys

GRADMAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "gradmap")
ASAP7_LIB_INFO = os.path.join(GRADMAP_DIR, "libs", "asap7_libcell_info.txt")


# ---------------------------------------------------------------------------
# Step 1: Parse ASAP7 cell library
# ---------------------------------------------------------------------------

def parse_asap7_cells(lib_path: str) -> dict:
    """Return {(gate_type, n_inputs): [(cell_name, area), ...]} sorted by area."""
    cells: dict = {}
    name = area_val = func = None
    with open(lib_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("libcell:"):
                name = line.split(":", 1)[1].strip()
                func = area_val = None
            elif line.startswith("function:") and name:
                func = line.split(":", 1)[1].strip()
            elif line.startswith("area:") and name and func is not None:
                area_val = float(line.split(":", 1)[1].strip())
                key = _classify_func(func)
                if key:
                    cells.setdefault(key, []).append((name, area_val))
                # reset to avoid re-triggering on later area: lines for same cell
                func = None

    # Sort each group by area ascending
    for k in cells:
        cells[k].sort(key=lambda x: x[1])
    return cells


# Map ASAP7 function strings to (gate_type, n_inputs) keys
_FUNC_MAP = {
    "!A":               ("inv",   1),
    "A":                ("buf",   1),
    "(!A) + (!B)":      ("nand",  2),
    "(A * B)":          ("and",   2),
    "(!A * !B)":        ("nor",   2),
    "(!A) + (!B) + (!C)": ("nand", 3),
    "(A * B * C)":      ("and",   3),
    "(!A * !B * !C)":   ("nor",   3),
    "(!A) + (!B) + (!C) + (!D)": ("nand", 4),
    "(A * B * C * D)":  ("and",   4),
    "(!A * !B * !C * !D)": ("nor", 4),
}


def _classify_func(func: str):
    return _FUNC_MAP.get(func)


# ---------------------------------------------------------------------------
# Step 2: .bench -> match file
# ---------------------------------------------------------------------------

def parse_bench(bench_path: str):
    """Return (pis, pos, gates) where:
      pis:   [net_name, ...]  in file order
      pos:   [net_name, ...]  (the driving net for each OUTPUT)
      gates: OrderedDict {out_net: (gate_type, [fanin_nets])}
    """
    pis, pos, gates = [], [], collections.OrderedDict()
    with open(bench_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("INPUT("):
                pis.append(line[6:-1].strip())
            elif line.startswith("OUTPUT("):
                pos.append(line[7:-1].strip())
            elif "=" in line:
                out, rhs = line.split("=", 1)
                out = out.strip()
                rhs = rhs.strip()
                m = re.match(r"(\w+)\((.+)\)$", rhs)
                if m:
                    gtype = m.group(1).upper()
                    fanins = [x.strip() for x in m.group(2).split(",")]
                    gates[out] = (gtype, fanins)
    return pis, pos, gates


def topo_sort(gates: dict) -> list:
    """Return gate output names in topological order."""
    out = []
    visited = set()

    def visit(name):
        if name in visited or name not in gates:
            return
        visited.add(name)
        gtype, fanins = gates[name]
        for f in fanins:
            visit(f)
        out.append(name)

    for name in gates:
        visit(name)
    return out


def generate_match_file(bench_path: str, match_path: str, cells: dict) -> None:
    """Generate gradmap match file from a .bench circuit using ASAP7 cells."""
    pis, pos, gates = parse_bench(bench_path)
    order = topo_sort(gates)

    # Assign even IDs: all signals (PIs then gate outputs, sorted by topo order)
    all_nets = list(pis) + order
    net_pos_id = {name: 2 * (i + 1) for i, name in enumerate(all_nets)}

    def neg_id(name):
        return net_pos_id[name] + 1

    lines = []
    # CONST nodes (required by gradmap parser internals)
    lines.append("0 CONST1 0.0\n")
    lines.append("1 CONST0 0.0\n")

    # PI nodes + INV candidates
    for pi in pis:
        pid = net_pos_id[pi]
        nid = pid + 1
        lines.append(f"{pid} input 0.00\n")
        inv_cells = cells.get(("inv", 1), [])
        best_inv = inv_cells[0] if inv_cells else ("INVx1_ASAP7_75t_L", 0.69984)
        lines.append(f"{nid} {best_inv[0]} {best_inv[1]:.5f} 1 {pid} 0\n")

    # Gate nodes
    for gname in order:
        gtype, fanins = gates[gname]
        node_id = net_pos_id[gname]
        n_in = len(fanins)
        fanin_pos = [net_pos_id[f] for f in fanins]
        fanin_neg = [net_pos_id[f] + 1 for f in fanins]

        # Determine which cells apply and which fanin polarity to use
        if gtype == "OR":
            # OR(A,B) = NAND(NOT A, NOT B) via DeMorgan
            key = ("nand", n_in)
            fanin_ids = fanin_neg
        elif gtype in ("AND", "NAND", "NOR"):
            key = (gtype.lower(), n_in)
            fanin_ids = fanin_pos
        elif gtype in ("NOT",):
            key = ("inv", 1)
            fanin_ids = fanin_pos
        elif gtype in ("BUFF", "BUF"):
            key = ("buf", 1)
            fanin_ids = fanin_pos
        else:
            # Unknown type: fallback to a buffer-like mapping on the first fanin
            key = ("buf", 1)
            fanin_ids = [fanin_pos[0]]

        candidate_cells = cells.get(key, [])
        if not candidate_cells:
            # Fallback: use INV chain if nothing else matches
            key = ("inv", 1)
            candidate_cells = cells.get(key, [("INVx1_ASAP7_75t_L", 0.69984)])
            fanin_ids = [fanin_pos[0]]

        # Emit one candidate line per cell variant
        n_fanins = len(fanin_ids)
        fanin_str = " ".join(str(i) for i in fanin_ids)
        for cname, carea in candidate_cells:
            lines.append(f"{node_id} {cname} {carea:.5f} {n_fanins} {fanin_str} 1 {node_id}\n")

        # INV node for this gate output (needed if it feeds an OR gate downstream)
        inv_cells = cells.get(("inv", 1), [])
        best_inv = inv_cells[0] if inv_cells else ("INVx1_ASAP7_75t_L", 0.69984)
        nid = node_id + 1
        lines.append(f"{nid} {best_inv[0]} {best_inv[1]:.5f} 1 {node_id} 0\n")

        # M line: warm-start with smallest-area cell
        best = candidate_cells[0]
        lines.append(f"M{node_id} {best[0]} {fanin_str}\n")

    # PO nodes
    po_start_id = 2 * (len(all_nets) + 1)
    for i, po_net in enumerate(pos):
        po_node_id = po_start_id + 2 * i
        fanin = net_pos_id.get(po_net, net_pos_id.get(po_net))
        lines.append(f"{po_node_id} output 0.00 {fanin}\n")

    # L lines (topological level per node)
    levels: dict = {}
    for pi in pis:
        levels[net_pos_id[pi]] = 0
        levels[net_pos_id[pi] + 1] = 1
    for gname in order:
        gtype, fanins = gates[gname]
        max_fanin_level = max((levels.get(net_pos_id[f], 0) for f in fanins), default=0)
        levels[net_pos_id[gname]] = max_fanin_level + 1
        levels[net_pos_id[gname] + 1] = max_fanin_level + 2

    for nid, lvl in sorted(levels.items()):
        lines.append(f"L{nid} {lvl}\n")

    os.makedirs(os.path.dirname(match_path), exist_ok=True)
    with open(match_path, "w") as f:
        f.writelines(lines)

    n_candidates = sum(1 for l in lines if l[0].isdigit() and "input" not in l and "output" not in l and "CONST" not in l)
    print(f"      PIs      : {len(pis)}")
    print(f"      Gates    : {len(gates)}")
    print(f"      Candidates: {n_candidates}")
    print(f"      Match file: {match_path}")


# ---------------------------------------------------------------------------
# Step 3: write gradmap config
# ---------------------------------------------------------------------------

def write_gradmap_config(match_path: str, config_path: str, gradmap_dir: str, steps: int, bench_stem: str) -> None:
    rel_match = os.path.relpath(os.path.abspath(match_path), os.path.abspath(gradmap_dir))

    config = f"""\
flow true
testcase.lib libs/asap7_libcell_info.txt
testcase.match {rel_match}
output.verilog verilog_output/{bench_stem}_quick.v

optimizer.method torch
optimizer.eval_backend gpu
optimizer.loss_type ADP
optimizer.area_factor 1.0
optimizer.delay_factor 1.0
optimizer.learning_rate 0.15
optimizer.total_steps {steps}
optimizer.eval_interval 10

optimizer.plateau_enable true
optimizer.plateau_patience 6
optimizer.plateau_threshold 0.003
optimizer.plateau_factor 0.5
optimizer.plateau_min_lr 0.01
optimizer.restore_best_on_plateau true

optimizer.softmax_temperature_start 1.0
optimizer.softmax_temperature_end   0.8

circuit.init_weights_strategy from_abc
circuit.abc_boost_value 2.0
"""
    print(f"[3/5] Writing gradmap config -> {os.path.basename(config_path)}")
    with open(config_path, "w") as f:
        f.write(config)


# ---------------------------------------------------------------------------
# Step 4: run gradmap_torch
# ---------------------------------------------------------------------------

def run_gradmap(config_path: str, gradmap_dir: str) -> str:
    binary = os.path.join(gradmap_dir, "gradmap_torch")
    if not os.path.isfile(binary):
        raise SystemExit(
            f"gradmap_torch not found at {binary}\n"
            f"Build with: cd {gradmap_dir} && bash compile.sh"
        )

    rel_config = os.path.relpath(config_path, gradmap_dir)
    print(f"[4/5] Running gradmap_torch  (config: {rel_config})")
    print("      " + "-" * 60)

    lines = []
    proc = subprocess.Popen(
        [binary, rel_config],
        cwd=gradmap_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print("      " + line, end="")
        lines.append(line)
    proc.wait()

    print("      " + "-" * 60)
    if proc.returncode != 0:
        raise SystemExit(f"gradmap_torch exited with code {proc.returncode}")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Step 5: parse and print summary metrics
# ---------------------------------------------------------------------------

def parse_metrics(output: str) -> None:
    # gradmap prints: Cost=X Area=Y Delay=Z  (ADP cost, not raw adp label)
    cost_pat  = re.compile(r"Cost=([0-9.]+)")
    delay_pat = re.compile(r"Delay=([0-9.]+)")
    area_pat  = re.compile(r"Area=([0-9.]+)")

    costs, delays, areas = [], [], []
    for line in output.splitlines():
        # Only capture from "Best" or final summary lines to avoid training noise
        if "New Best Found" in line or "Restoring best" in line:
            m = cost_pat.search(line)
            if m:
                costs.append(float(m.group(1)))
            m = delay_pat.search(line)
            if m:
                delays.append(float(m.group(1)))
            m = area_pat.search(line)
            if m:
                areas.append(float(m.group(1)))

    print("\n[5/5] Summary")
    if costs:
        print(f"      Best ADP   : {min(costs):.4f}")
    if delays:
        print(f"      Best delay : {min(delays):.4f} ps")
    if areas:
        print(f"      Best area  : {min(areas):.4f}")
    if not (costs or delays or areas):
        print("      (Could not parse numeric metrics — see output above)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench", nargs="?", default="benchmarks/c880.bench",
                        help="Path to .bench file (default: benchmarks/c880.bench)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Optimizer steps (default: 200)")
    args = parser.parse_args()

    if not os.path.isfile(args.bench):
        raise SystemExit(f"Benchmark not found: {args.bench}")

    gradmap_dir = os.path.abspath(GRADMAP_DIR)
    bench_stem = os.path.splitext(os.path.basename(args.bench))[0]
    match_path  = os.path.join(gradmap_dir, "match", f"{bench_stem}_quick.txt")
    config_path = os.path.join(gradmap_dir, "config", f"{bench_stem}_quick_config.txt")

    print(f"Benchmark : {args.bench}")
    print(f"Library   : ASAP7 (full, {gradmap_dir}/libs/asap7_libcell_info.txt)")
    print(f"Steps     : {args.steps}")
    print()

    print(f"[1/5] Parsing ASAP7 cell library")
    cells = parse_asap7_cells(ASAP7_LIB_INFO)
    total_cells = sum(len(v) for v in cells.values())
    print(f"      {total_cells} cells across {len(cells)} function types")

    print(f"[2/5] Generating match file from .bench")
    generate_match_file(args.bench, match_path, cells)

    write_gradmap_config(match_path, config_path, gradmap_dir, args.steps, bench_stem)
    output = run_gradmap(config_path, gradmap_dir)
    parse_metrics(output)


if __name__ == "__main__":
    main()
