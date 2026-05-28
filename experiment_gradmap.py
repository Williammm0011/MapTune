"""Gradmap-based gate-usage experiment.

Usage: python experiment_gradmap.py

Reads [experiment] from config.toml. For the configured benchmark:
  Pass A: map with full ASAP7 library via gradmap_torch
  Pass B: map again using only the gates that appeared in Pass A output

Produces a gate-usage heatmap saved to logs/.

Requirements:
  - gradmap_torch built:  cd third_party/gradmap && bash compile.sh
  - libs/asap7_libcell_info.txt present in third_party/gradmap/libs/
      generate with: python third_party/gradmap/libs/lut.py \\
                       -files $GRADMAP_LIBS/asap7*.lib \\
                       -o third_party/gradmap/libs/asap7_libcell_info.txt
  - GRADMAP_LIBS=/path/  containing asap7.lib + rec6Lib_final_filtered3_recanon.aig
  - abc on PATH  or  ABC_PATH=/path/to/abc/dir

[experiment.hand_made] entries must be lists of ASAP7 gate name strings
(not integer indices, which are genlib-specific). Integer entries are skipped.
"""

import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from src.gradmap_mapper import GradMapper
from experiment import plot_trials, resolve_design


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


def _gate_indices(mapper, gate_names):
    name_set = set(gate_names)
    return sorted(i for i, g in enumerate(mapper._gate_names) if g in name_set)


def run_full_lib(mapper):
    """Two-pass gradmap: full ASAP7 match → used-gates-only match."""
    print("  pass A: full ASAP7 match...")
    delay_a, area_a, v_a = mapper.map_full(tag="full_a")
    cost_before = mapper.calculate_cost(delay_a, area_a)

    used_names = mapper.used_gate_names(v_a)
    used_idx   = _gate_indices(mapper, used_names)
    unused_idx = [i for i in range(mapper.num_arms) if i not in set(used_idx)]
    print(f"    {len(used_idx):>3} used / {len(unused_idx):>3} unused  cost={cost_before:.4f}")

    print("  pass B: used-gates-only match...")
    delay_b, area_b, v_b = mapper.map_filtered(used_names, tag="full_b")
    cost_after = mapper.calculate_cost(delay_b, area_b)

    used_names2 = mapper.used_gate_names(v_b)
    used_idx_b   = _gate_indices(mapper, used_names2)
    unused_idx_b = [i for i in used_idx if mapper._gate_names[i] not in used_names2]
    print(f"    {len(used_idx_b):>3} used / {len(unused_idx_b):>3} unused  cost={cost_after:.4f}")

    return dict(
        selected=list(range(mapper.num_arms)),
        used_idx=used_idx,
        unused_idx=unused_idx,
        cost_before=cost_before,
        used_idx_b=used_idx_b,
        unused_idx_b=unused_idx_b,
        cost_after=cost_after,
    )


def run_hand_made(mapper, gate_names):
    """Two-pass gradmap with a fixed set of ASAP7 gate name strings."""
    gate_set = set(gate_names)
    selected_idx = _gate_indices(mapper, gate_set)

    print("  hand-made pass A...")
    delay_a, area_a, v_a = mapper.map_filtered(gate_set, tag="hm_a")
    cost_before = mapper.calculate_cost(delay_a, area_a)

    used_names = mapper.used_gate_names(v_a)
    used_idx   = [i for i in selected_idx if mapper._gate_names[i] in used_names]
    unused_idx = [i for i in selected_idx if mapper._gate_names[i] not in used_names]
    print(f"    {len(used_idx):>3} used / {len(unused_idx):>3} unused  cost={cost_before:.4f}")

    print("  hand-made pass B...")
    delay_b, area_b, v_b = mapper.map_filtered(used_names, tag="hm_b")
    cost_after = mapper.calculate_cost(delay_b, area_b)

    used_names2 = mapper.used_gate_names(v_b)
    used_idx_b   = [i for i in used_idx if mapper._gate_names[i] in used_names2]
    unused_idx_b = [i for i in used_idx if mapper._gate_names[i] not in used_names2]
    print(f"    {len(used_idx_b):>3} used / {len(unused_idx_b):>3} unused  cost={cost_after:.4f}")

    return dict(
        selected=selected_idx,
        used_idx=used_idx,
        unused_idx=unused_idx,
        cost_before=cost_before,
        used_idx_b=used_idx_b,
        unused_idx_b=unused_idx_b,
        cost_after=cost_after,
    )


def main():
    cfg = load_config()
    ecfg = cfg["experiment"]
    bench_name = ecfg["bench"]

    design = resolve_design(cfg, bench_name)
    os.makedirs(cfg["paths"]["temp_blifs_dir"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"Design : {design}")
    print(f"Library: ASAP7 (gradmap)")
    print()

    mapper = GradMapper(design, cfg["paths"]["temp_blifs_dir"] + "/")
    print(f"Total mutable gate types : {mapper.num_arms}")
    print(f"Baseline  delay={mapper.baseline_delay:.3f}  area={mapper.baseline_area:.3f}\n")

    print("Running full-library mapping (two-pass)...")
    full_lib_trial = run_full_lib(mapper)

    hand_made_trial = None
    hand_made_cfg = ecfg.get("hand_made", {}).get(bench_name)
    if hand_made_cfg:
        if isinstance(hand_made_cfg[0], str):
            print("\nRunning hand-made gate selection...")
            hand_made_trial = run_hand_made(mapper, hand_made_cfg)
        else:
            print(
                "\n[skip] experiment.hand_made entries for gradmap must be ASAP7 gate name "
                "strings, not integer indices (which are genlib/ABC-specific)."
            )

    out_path = f"logs/experiment_gradmap_{bench_name}.png"
    plot_trials([], mapper.num_arms, "asap7", bench_name, out_path,
                full_lib=full_lib_trial, hand_made=hand_made_trial)


if __name__ == "__main__":
    main()
