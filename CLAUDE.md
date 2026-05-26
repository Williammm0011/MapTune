# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MapTune is a research project (ICCAD'24) that uses optimization agents to tune ASIC technology library subsets for ABC-based technology mapping. The core idea: given a full genlib (technology library), find a subset of gates that minimizes the normalized Area-Delay Product (ADP) after mapping a circuit.

**ABC** (a logic synthesis tool) is available via the `maptune` conda environment:
```bash
conda activate maptune
```

## Dependencies

```bash
conda activate maptune
pip install -r requirements.txt  # numpy, gymnasium, matplotlib, tomli
```

Python 3.8+ required. `tomli` is used for Python < 3.11; Python 3.11+ uses the built-in `tomllib`.

## Running Agents

All agent scripts live in `src/` and must be run from the repo root so that `config.toml` and `benchmarks/` resolve correctly:

```bash
# Single-agent SA
python src/sa.py --lib 7nm --bench benchmarks/s838a.bench

# Parallel adaptive SA (4 workers, multiprocessing)
python src/sa_multi.py --lib 7nm --bench benchmarks/s838a.bench --n-agents 4

# Random baseline (single)
python src/random_agent.py --lib 7nm --bench benchmarks/s838a.bench

# Parallel random baseline
python src/random_multi.py --lib 7nm --bench benchmarks/s838a.bench

# Full sweep: all (lib × bench × method) combos
python src/run_sweep.py --libs 7nm sky130 --benches benchmarks/c880.bench benchmarks/s838a.bench
```

All `--lib` values must match `name` fields in `config.toml [[library]]`. Available: `7nm`, `nan45`, `sky130`, `gf180mcu_tt_025C`, `gf180mcu_ff_125C`.

All hyperparameters default to `config.toml` values and can be overridden with CLI flags (`--iterations`, `--n-select`, `--t0`, `--t-min`, `--seed`, etc.).

## Plotting

```bash
# Single run: pass log dir(s)
python plot.py logs/20260526_120000_sa_7nm_c880/

# Batch: compare sa_multi vs random_multi, one PNG per (lib, design) pair
python plot_batch.py logs/

# Gate usage analysis across sa_multi_7nm_* runs
python data_analysis/common_gates.py logs/ --n-gates 161
```

## Architecture

**`src/abc_mapper.py` — TechMapper (central abstraction)**
- Parses a `.genlib` into `mutable_gates` (samplable) and `fixed_gates` (BUF/INV, always included).
- `map_subset(gate_indices)` writes a partial genlib, calls ABC, parses delay+area from stdout.
- `calculate_reward()` returns negative geometric-mean normalized ADP (higher is better).
- `baseline_delay/baseline_area` are computed at construction using the full genlib.
- ABC is called via `subprocess.check_output(("abc", "-c", cmd_string))`.

**Agent scripts** (all in `src/`) share the same structure:
1. Load `config.toml`, parse CLI args (args override config defaults).
2. Construct `TechMapper`.
3. Run optimization loop — each iteration calls `mapper.map_subset()` once.
4. Log each step to CSV (`metrics.csv`) and a run log (`run.log`), with config snapshot in `config.json`.

**`sa.py`** — Single SA agent. Starts with `n_select` random gates, toggles one gate per step, accepts worse moves with Boltzmann probability.

**`sa_multi.py`** — Parallel SA with N workers via `multiprocessing.Pool`. Each worker independently adapts its cooling factor based on a sliding-window acceptance rate. When stuck (`restart_patience` steps without improvement), does an exhaustive 1-gate neighborhood sweep; if still stuck, random restarts. Writes per-agent logs + a `summary.json` with the global best.

**`random_agent.py` / `random_multi.py`** — Baselines that sample uniformly random gate subsets each iteration. `random_multi` parallelizes N workers similarly to `sa_multi`.

**`run_sweep.py`** — Orchestrates `sa_multi.py` and `random_multi.py` sequentially over all (lib, bench) combinations. Each sub-script runs its own internal parallelism.

## Output Layout

```
logs/<timestamp>_<method>_<lib>_<bench>/
    run.log          # human-readable progress
    config.json      # run hyperparameters
    metrics.csv      # per-step: step, cost, best_cost, area, delay, ...
    [agent_N/]       # per-agent subdirs for _multi variants
        metrics.csv
        config.json
    summary.json     # _multi only: cross-agent best, full agent results
gen_newlibs/         # partial .genlib files written during each ABC call (gitignored)
temp_blifs/          # intermediate .blif files from ABC (gitignored)
```

## Key Design Decisions

- **BUF/INV gates are always fixed** in every partial library to guarantee ABC can map any circuit. `_FIXED_PREFIXES` in `abc_mapper.py` enumerates the prefixes for all supported libraries.
- **ADP cost** (`delay/baseline_delay × area/baseline_area`) is the objective; lower is better for cost, higher for reward. Failed ABC calls return `(nan, nan)` → `inf` cost.
- **`area_mode=True`** is passed to `TechMapper` in all current agents, which uses `map -a` (area-driven) instead of the default delay-driven `map` in ABC.
- **`.lib` files are gitignored** (large Liberty timing files); only `.genlib` files are tracked.
- `_ROOT` in each script resolves to the repo root via `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` — this is how scripts in `src/` find `config.toml` and `benchmarks/`.
