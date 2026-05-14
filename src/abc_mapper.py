import re
import subprocess
import numpy as np

# BUF/INV prefixes that must always be kept in every partial library
_FIXED_PREFIXES = (
    "GATE BUF",
    "GATE INV",
    "GATE sky130_fd_sc_hd__buf",
    "GATE sky130_fd_sc_hd__inv",
    "GATE gf180mcu_fd_sc_mcu7t5v0__buf",
    "GATE gf180mcu_fd_sc_mcu7t5v0__inv",
)


def parse_genlib_gates(genlib_path):
    """Return (mutable_gates, fixed_gates) line lists from a .genlib file.

    mutable_gates: GATE lines that can be subsampled (excludes BUF/INV).
    fixed_gates: BUF/INV lines that are always included.
    """
    mutable, fixed = [], []
    with open(genlib_path) as f:
        for line in f:
            s = line.strip()
            if not s.startswith("GATE"):
                continue
            if s.startswith(_FIXED_PREFIXES):
                fixed.append(s)
            else:
                mutable.append(s)
    return mutable, fixed


def _run_abc(cmd_string):
    return subprocess.check_output(("abc", "-c", cmd_string)).decode("utf-8", errors="replace")


def _parse_delay_area(output):
    m_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", output)
    m_a = re.search(r"Area\s*=\s*([\d.]+)", output)
    if m_d and m_a:
        return float(m_d.group(1)), float(m_a.group(1))
    return float("nan"), float("nan")


class TechMapper:
    """ABC technology-mapping wrapper for gate library subset selection.

    Usage:
        mapper = TechMapper("libs/7nm.genlib", "benchmarks/s838a.bench",
                            "gen_newlibs/", "temp_blifs/s838a_temp.blif")
        delay, area = mapper.map_subset([0, 3, 7], tag="ep")
        reward = mapper.calculate_reward(delay, area)
        n = mapper.num_arms   # number of mutable (non-BUF/INV) gates
    """

    def __init__(self, genlib_path, design_path, output_lib_dir, temp_blif, area_mode=False):
        self.genlib_path = genlib_path
        self.design_path = design_path
        self.output_lib_dir = output_lib_dir
        # Paired .lib file for post-mapping timing (same stem as .genlib)
        self.lib_origin = genlib_path[:-7] + ".lib"
        self.temp_blif = temp_blif
        self._map_cmd = "map -a" if area_mode else "map"
        self.mutable_gates, self.fixed_gates = parse_genlib_gates(genlib_path)
        self.baseline_delay, self.baseline_area = self._compute_baseline()

    def _abc_cmd(self, genlib, design, temp_blif, lib_origin):
        return (
            f"read {genlib};read {design}; {self._map_cmd}; write {temp_blif}; "
            f"read {lib_origin};read -m {temp_blif}; ps; topo; upsize; dnsize; stime; "
        )

    def _compute_baseline(self):
        cmd = self._abc_cmd(self.genlib_path, self.design_path, self.temp_blif, self.lib_origin)
        return _parse_delay_area(_run_abc(cmd))

    def map_subset(self, gate_indices, tag="sample"):
        """Map the design using a subset of mutable gates. Returns (delay, area)."""
        lines = [self.mutable_gates[i] for i in gate_indices] + self.fixed_gates
        design_slug = self.design_path.replace("/", "_")
        out_genlib = f"{self.output_lib_dir}{design_slug}_{len(lines)}_{tag}_samplelib.genlib"
        with open(out_genlib, "w") as f:
            f.write("\n".join(lines) + "\n")
        cmd = self._abc_cmd(out_genlib, self.design_path, self.temp_blif, self.lib_origin)
        try:
            return _parse_delay_area(_run_abc(cmd))
        except subprocess.CalledProcessError:
            return float("nan"), float("nan")

    def calculate_reward(self, delay, area):
        """Geometric-mean normalised reward. Higher is better (range (-inf, 0])."""
        if not (delay > 0 and area > 0):
            return float("-inf")
        return -np.sqrt((delay / self.baseline_delay) * (area / self.baseline_area))

    @property
    def num_arms(self):
        return len(self.mutable_gates)
