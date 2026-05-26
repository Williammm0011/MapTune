import re
from src.abc_mapper import TechMapper


def _gate_name(gate_line):
    return gate_line.split()[1]


def _used_gate_names(blif_path):
    used = set()
    with open(blif_path) as f:
        for line in f:
            m = re.match(r"\.gate\s+(\S+)", line)
            if m:
                used.add(m.group(1))
    return used


class LibraryPruner:
    def __init__(self, genlib_path, design_path, output_lib_dir, temp_blif, area_mode=False):
        self.mapper = TechMapper(
            genlib_path, design_path, output_lib_dir, temp_blif, area_mode)

    def prune(self, max_iter=20):
        active = list(range(self.mapper.num_arms))
        log = []

        for i in range(max_iter):
            delay, area = self.mapper.map_subset(active, tag=f"prune_{i}")
            cost = self.mapper.calculate_cost(delay, area)
            log.append({"iter": i, "gates": len(active),
                       "delay": delay, "area": area, "cost": cost})

            used = _used_gate_names(self.mapper.temp_blif)
            next_active = [idx for idx in active if _gate_name(
                self.mapper.mutable_gates[idx]) in used]

            if next_active == active:
                break
            active = next_active

        return active, delay, area, cost, log
