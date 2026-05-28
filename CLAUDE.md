# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MapTune (ICCAD'24) tunes ASIC technology-mapping by using Reinforcement Learning agents to select subsets of gates from a standard-cell library, then running the ABC EDA tool to map a circuit design and measure delay/area. The core loop: sample gate indices → write a partial `.genlib` → call ABC → parse delay/area → compute reward → update agent.

## Environment requirement

`abc` must be on `PATH`. Set it with:

```bash
export PATH=/your/path/to/ABC:${PATH}
```

Python 3.8+. For DNN agents (DQN/DDQN) CUDA 11.5 is expected.

## Running agents

All agent scripts take the same three positional arguments: `<num_sampled_gate> <design> <genlib>`.

```bash
python MAB_EP.py 65 benchmarks/s838a.bench libs/7nm.genlib
python batched_DQN.py 65 benchmarks/s838a.bench libs/7nm.genlib
```

The `config.toml` lists all available libraries (`[[library]]` entries) and benchmark paths by category (`bench`, `bench_large`, `bench_seq`, `blif`). Load it with:

```python
import tomllib
cfg = tomllib.load(open("config.toml", "rb"))
```

## Key abstraction: `TechMapper` ([src/abc_mapper.py](src/abc_mapper.py))

This is the only file in `src/` right now. It wraps ABC and is what all agents should build on:

- **`TechMapper(genlib_path, design_path, output_lib_dir, temp_blif, area_mode=False)`** — on init, splits the genlib into `mutable_gates` (can be subsampled) and `fixed_gates` (BUF/INV, always included), then runs a baseline mapping.
- **`mapper.map_subset(gate_indices, tag="sample") → (delay, area)`** — writes a partial genlib from selected indices + fixed gates, runs ABC, returns parsed delay/area.
- **`mapper.calculate_reward(delay, area) → float`** — geometric-mean normalised score, `−√(delay/baseline × area/baseline)`. Higher (less negative) is better. Returns `-inf` on invalid results.
- **`mapper.num_arms`** — total count of mutable gates (action space size).
- **`mapper.baseline_delay`, `mapper.baseline_area`** — set at init from the full library.

The ABC command sequence used is:

```
read <genlib>; read <design>; map; write <temp_blif>;
read <lib>; read -m <temp_blif>; ps; topo; upsize; dnsize; stime;
```

`-a` flag on `map` switches to area-driven mode.

## Key abstraction: `GradMapper` ([src/gradmap_mapper.py](src/gradmap_mapper.py))

gradmap-based alternative to `TechMapper`. Uses the gradmap submodule at `third_party/gradmap/`.

- **`GradMapper(bench_path, output_dir)`** — on init, generates an ASAP7 match file via ABC `&nf -Y` (cached in `output_dir`), then runs a baseline mapping.
- **`mapper.map_full(tag) → (delay, area, verilog_path)`** — run `gradmap_torch` with the full ASAP7 match file.
- **`mapper.map_filtered(gate_names, tag) → (delay, area, verilog_path)`** — filter the match file to `gate_names`, run `gradmap_torch`.
- **`mapper.used_gate_names(verilog_path) → set`** — parse a Verilog netlist for instantiated cell names.
- **`mapper.calculate_cost(delay, area) → float`** — normalised ADP; same formula as `TechMapper`.
- **`mapper.num_arms`** — unique gate types in the ASAP7 match for this benchmark.

gradmap requires:
1. `gradmap_torch` built: `cd third_party/gradmap && bash compile.sh`
2. `third_party/gradmap/libs/asap7_libcell_info.txt` present (generate once: `python third_party/gradmap/libs/lut.py -files $GRADMAP_LIBS/asap7*.lib -o third_party/gradmap/libs/asap7_libcell_info.txt`)
3. `GRADMAP_LIBS=/path/` containing `asap7.lib` and `rec6Lib_final_filtered3_recanon.aig`
4. `abc` on PATH (or `ABC_PATH` set)

## Running the gradmap experiment

```bash
export GRADMAP_LIBS=/path/to/libs
python experiment_gradmap.py
# → logs/experiment_gradmap_{bench}.png
```

Reads the same `[experiment]` section from `config.toml`. Runs two passes: full ASAP7 library, then only the gates that appeared in the first pass output. `[experiment.hand_made]` entries must be lists of ASAP7 gate name strings (not genlib integer indices).

## Repository layout

- `src/abc_mapper.py` — `TechMapper` class, ABC-based mapper
- `src/gradmap_mapper.py` — `GradMapper` class, gradmap-based mapper
- `src/agents/` — agent implementations (currently deleted from working tree; was `mab.py` with `EpsilonGreedyMAB` and `UCB_MAB`)
- `libs/` — `.genlib` files tracked in git; paired `.lib` Liberty timing files are **gitignored** (too large)
- `benchmarks/` — `.bench` (ISCAS/ITC) and `.blif` (EPFL) circuit designs
- `third_party/gradmap/` — gradmap submodule; binary not tracked (build locally)
- `config.toml` — canonical list of libraries and benchmarks
- `temp_blifs/` — gitignored; intermediate files (BLIFs, generated match files)
- `gen_newlibs/` — gitignored; generated partial genlibs written by `map_subset`
- `logs/` — gitignored; run logs and plots

## Reward function

Reward = `−√(delay/baseline_delay × area/baseline_area)`.  
A subset library that beats baseline on both metrics gives reward in `(−1, 0)`. Full baseline maps to `−1.0`. Invalid mappings return `−inf`.

## What was refactored out

The original top-level scripts (`MAB_EP.py`, `MAB_UCB.py`, `batched_*.py`, C sources) were deleted in the "refactor for further RL" commit (ca2e711). Their logic is now the responsibility of new code in `src/`. The `TechMapper` class is the stable API to build on.
