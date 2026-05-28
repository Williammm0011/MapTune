"""Run ABC mapping with a specified gate set and report cost.

Usage:
  python quick_map.py <lib> <bench> <gate0,gate1,...>

  lib   — library name from config.toml (e.g. 7nm)
  bench — benchmark name from config.toml (e.g. c880, b20_1)
  gates — comma-separated mutable gate indices

Examples:
  python quick_map.py 7nm c880 0,4,5,9,10
  python quick_map.py 7nm b20_1 0,4,5,9,10,11,12
"""

import os
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from src.abc_mapper import TechMapper


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_library(cfg, name):
    for lib in cfg["library"]:
        if lib["name"] == name:
            return lib["genlib"]
    raise SystemExit(f"Unknown library '{name}'")


def resolve_design(cfg, name):
    all_paths = (
        cfg["benchmarks"].get("bench", [])
        + cfg["benchmarks"].get("bench_large", [])
        + cfg["benchmarks"].get("bench_seq", [])
        + cfg["benchmarks"].get("blif", [])
    )
    for path in all_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem == name:
            return path
    raise SystemExit(f"Unknown benchmark '{name}'")


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    lib_name, bench_name, gates_str = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        gate_indices = sorted(int(x) for x in gates_str.split(","))
    except ValueError:
        raise SystemExit(f"Invalid gate list: {gates_str!r}")

    cfg = load_config()
    genlib = resolve_library(cfg, lib_name)
    design = resolve_design(cfg, bench_name)

    os.makedirs(cfg["paths"]["temp_blifs_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["gen_newlibs_dir"], exist_ok=True)

    temp_blif = f"{cfg['paths']['temp_blifs_dir']}/{bench_name}_quick_temp.blif"
    mapper = TechMapper(genlib, design, cfg["paths"]["gen_newlibs_dir"] + "/", temp_blif)

    print(f"Library  : {genlib}")
    print(f"Design   : {design}")
    print(f"Gates    : {len(gate_indices)} selected  (of {mapper.num_arms} mutable)")
    print(f"Baseline : delay={mapper.baseline_delay:.1f}ps  area={mapper.baseline_area:.1f}\n")

    delay, area = mapper.map_subset(gate_indices, tag="quick")
    cost = mapper.calculate_cost(delay, area)

    if cost == float("-inf"):
        print("Result   : INVALID (mapping failed)")
    else:
        print(f"Delay    : {delay:.1f} ps  (baseline {mapper.baseline_delay:.1f})")
        print(f"Area     : {area:.1f}     (baseline {mapper.baseline_area:.1f})")
        print(f"Cost     : {cost:.6f}  (baseline 1.0)")


if __name__ == "__main__":
    main()
