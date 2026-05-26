"""Iterative library pruning.

Usage: python prune.py

Reads lib and bench from [prune] in config.toml.
Starts with the full library, drops unused gates after each mapping,
and repeats until the gate set converges.
"""

import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from src.pruner import LibraryPruner


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_library(cfg, lib_name):
    for lib in cfg["library"]:
        if lib["name"] == lib_name:
            return lib["genlib"]
    names = [lib["name"] for lib in cfg["library"]]
    raise SystemExit(f"Unknown library '{lib_name}'. Available: {', '.join(names)}")


def resolve_design(cfg, bench_name):
    all_paths = (
        cfg["benchmarks"].get("bench", [])
        + cfg["benchmarks"].get("bench_large", [])
        + cfg["benchmarks"].get("bench_seq", [])
        + cfg["benchmarks"].get("blif", [])
    )
    for path in all_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem == bench_name:
            return path
    stems = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]
    raise SystemExit(f"Unknown benchmark '{bench_name}'. Available: {', '.join(stems)}")


def main():
    cfg = load_config()
    lib_name = cfg["prune"]["lib"]
    bench_name = cfg["prune"]["bench"]

    genlib = resolve_library(cfg, lib_name)
    design = resolve_design(cfg, bench_name)

    os.makedirs(cfg["paths"]["temp_blifs_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["gen_newlibs_dir"], exist_ok=True)

    temp_blif = f"{cfg['paths']['temp_blifs_dir']}/{bench_name}_prune_temp.blif"

    print(f"Library : {genlib}")
    print(f"Design  : {design}")
    print()

    pruner = LibraryPruner(genlib, design, cfg["paths"]["gen_newlibs_dir"] + "/", temp_blif)

    print(f"{'iter':>4}  {'gates':>5}  {'delay(ps)':>10}  {'area':>10}  {'cost':>8}")
    print("-" * 46)

    active, delay, area, cost, log = pruner.prune()

    for entry in log:
        print(
            f"{entry['iter']:>4}  {entry['gates']:>5}  {entry['delay']:>10.1f}"
            f"  {entry['area']:>10.1f}  {entry['cost']:>8.4f}"
        )

    print("-" * 46)
    print(f"Converged: {len(active)} gates  delay={delay:.1f}ps  area={area:.1f}  cost={cost:.4f}")


if __name__ == "__main__":
    main()
