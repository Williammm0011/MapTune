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

## Repository layout

- `src/abc_mapper.py` — `TechMapper` class, the shared ABC interface for all agents
- `src/agents/` — agent implementations (currently deleted from working tree; was `mab.py` with `EpsilonGreedyMAB` and `UCB_MAB`)
- `libs/` — `.genlib` files tracked in git; paired `.lib` Liberty timing files are **gitignored** (too large)
- `benchmarks/` — `.bench` (ISCAS/ITC) and `.blif` (EPFL) circuit designs
- `config.toml` — canonical list of libraries and benchmarks
- `temp_blifs/` — gitignored; ABC writes intermediate BLIFs here during mapping
- `gen_newlibs/` — gitignored; generated partial genlibs written by `map_subset`
- `logs/` — gitignored; run logs and plots

## Reward function

Reward = `−√(delay/baseline_delay × area/baseline_area)`.  
A subset library that beats baseline on both metrics gives reward in `(−1, 0)`. Full baseline maps to `−1.0`. Invalid mappings return `−inf`.

## What was refactored out

The original top-level scripts (`MAB_EP.py`, `MAB_UCB.py`, `batched_*.py`, C sources) were deleted in the "refactor for further RL" commit (ca2e711). Their logic is now the responsibility of new code in `src/`. The `TechMapper` class is the stable API to build on.
