import random
from src.pruner import _gate_name, _used_gate_names


def run_trials(mapper, n_trials, sample_size):
    trials = []
    for t in range(n_trials):
        selected = sorted(random.sample(range(mapper.num_arms), sample_size))

        delay, area = mapper.map_subset(selected, tag=f"exp_{t}_a")
        cost_before = mapper.calculate_cost(delay, area)

        used_names = _used_gate_names(mapper.temp_blif)
        used_idx   = [i for i in selected if _gate_name(mapper.mutable_gates[i]) in used_names]
        unused_idx = [i for i in selected if _gate_name(mapper.mutable_gates[i]) not in used_names]

        delay2, area2 = mapper.map_subset(used_idx, tag=f"exp_{t}_b")
        cost_after = mapper.calculate_cost(delay2, area2)

        used_names2  = _used_gate_names(mapper.temp_blif)
        used_idx_b   = [i for i in used_idx   if _gate_name(mapper.mutable_gates[i]) in used_names2]
        unused_idx_b = [i for i in used_idx if _gate_name(mapper.mutable_gates[i]) not in used_names2]

        trials.append(dict(
            selected=selected,
            used_idx=used_idx,
            unused_idx=unused_idx,
            cost_before=cost_before,
            used_idx_b=used_idx_b,
            unused_idx_b=unused_idx_b,
            cost_after=cost_after,
        ))
        print(f"  trial {t+1:>2}: {len(used_idx):>3} used / {len(unused_idx):>3} unused  "
              f"cost {cost_before:.4f} → {cost_after:.4f}")

    return trials


def run_full_lib(mapper):
    """Map with all mutable gates (two-pass: before/after pruning unused)."""
    selected = list(range(mapper.num_arms))

    delay, area = mapper.map_subset(selected, tag="full_lib_a")
    cost_before = mapper.calculate_cost(delay, area)

    used_names = _used_gate_names(mapper.temp_blif)
    used_idx   = [i for i in selected if _gate_name(mapper.mutable_gates[i]) in used_names]
    unused_idx = [i for i in selected if _gate_name(mapper.mutable_gates[i]) not in used_names]

    delay2, area2 = mapper.map_subset(used_idx, tag="full_lib_b")
    cost_after = mapper.calculate_cost(delay2, area2)

    used_names2  = _used_gate_names(mapper.temp_blif)
    used_idx_b   = [i for i in used_idx if _gate_name(mapper.mutable_gates[i]) in used_names2]
    unused_idx_b = [i for i in used_idx if _gate_name(mapper.mutable_gates[i]) not in used_names2]

    print(f"  full-lib:  {len(used_idx):>3} used / {len(unused_idx):>3} unused  "
          f"cost {cost_before:.4f} → {cost_after:.4f}")

    return dict(
        selected=selected,
        used_idx=used_idx,
        unused_idx=unused_idx,
        cost_before=cost_before,
        used_idx_b=used_idx_b,
        unused_idx_b=unused_idx_b,
        cost_after=cost_after,
    )


def run_hand_made(mapper, selected):
    """Map with a fixed pre-selected gate set (two-pass: before/after pruning unused)."""
    selected = sorted(selected)

    delay, area = mapper.map_subset(selected, tag="hand_made_a")
    cost_before = mapper.calculate_cost(delay, area)

    used_names = _used_gate_names(mapper.temp_blif)
    used_idx   = [i for i in selected if _gate_name(mapper.mutable_gates[i]) in used_names]
    unused_idx = [i for i in selected if _gate_name(mapper.mutable_gates[i]) not in used_names]

    delay2, area2 = mapper.map_subset(used_idx, tag="hand_made_b")
    cost_after = mapper.calculate_cost(delay2, area2)

    used_names2  = _used_gate_names(mapper.temp_blif)
    used_idx_b   = [i for i in used_idx if _gate_name(mapper.mutable_gates[i]) in used_names2]
    unused_idx_b = [i for i in used_idx if _gate_name(mapper.mutable_gates[i]) not in used_names2]

    print(f"  hand-made: {len(used_idx):>3} used / {len(unused_idx):>3} unused  "
          f"cost {cost_before:.4f} → {cost_after:.4f}")

    return dict(
        selected=selected,
        used_idx=used_idx,
        unused_idx=unused_idx,
        cost_before=cost_before,
        used_idx_b=used_idx_b,
        unused_idx_b=unused_idx_b,
        cost_after=cost_after,
    )
