import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_GRADMAP_ROOT = Path(__file__).parent.parent / "third_party" / "gradmap"

_BALANCE_SUFFIXES = ("_balance", "_area", "_delay")

# Used for full-match runs (warm-start from ABC's initial match)
_CONFIG_FULL = """\
flow true
regression false
testcase.lib libs/asap7_libcell_info.txt
testcase.match {match_file}
output.verilog {output_placeholder}
optimizer.method torch
optimizer.total_steps 500
optimizer.eval_interval 10
circuit.init_weights_strategy from_abc
circuit.abc_boost_value 2.0
"""

# Used for filtered-match runs (no ABC warm-start; M lines may have been dropped)
_CONFIG_FILTERED = """\
flow true
regression false
testcase.lib libs/asap7_libcell_info.txt
testcase.match {match_file}
output.verilog {output_placeholder}
optimizer.method torch
optimizer.total_steps 500
optimizer.eval_interval 10
circuit.init_weights_strategy average
"""


def _resolve_abc() -> str:
    abc_dir = os.environ.get("ABC_PATH")
    if abc_dir:
        candidate = str(Path(abc_dir) / "abc")
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("abc")
    if found:
        return found
    raise RuntimeError("abc not found — set ABC_PATH or add abc to PATH")


def _testcase_from_match_path(match_path: str) -> str:
    """Extract testcase name the same way gradmap does: strip known mode suffixes."""
    stem = Path(match_path).stem
    for sfx in _BALANCE_SUFFIXES:
        if stem.endswith(sfx):
            return stem[: -len(sfx)]
    return stem


def _actual_verilog(match_path: str, output_dir: Path) -> Path:
    """Compute the path gradmap will write: {output_dir}/{testcase}_best.v"""
    return output_dir / f"{_testcase_from_match_path(match_path)}_best.v"


def _run_gradmap(binary: str, config_path: str, gradmap_root: Path, timeout: int = 600) -> str:
    result = subprocess.run(
        [binary, config_path],
        cwd=gradmap_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gradmap_torch failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _parse_delay_area(stdout: str) -> Tuple[float, float]:
    """Return best (delay, area) from gradmap stdout. Returns (nan, nan) if nothing found."""
    pattern = re.compile(
        r"\[Save\] New Best Found! Cost=[\d.e+\-]+\s+\(Area=([\d.e+\-]+) Delay=([\d.e+\-]+)\)"
    )
    delay, area = float("nan"), float("nan")
    for m in pattern.finditer(stdout):
        area = float(m.group(1))
        delay = float(m.group(2))
    return delay, area


_VERILOG_KEYWORDS = frozenset(
    ("module", "endmodule", "input", "output", "wire", "reg",
     "assign", "always", "begin", "end", "if", "else", "case",
     "endcase", "for", "while", "parameter", "localparam", "inout")
)


def _used_gate_names_from_verilog(verilog_path: str) -> Set[str]:
    """Parse cell instantiation lines from a Verilog netlist."""
    used: Set[str] = set()
    inst_re = re.compile(r"^\s*(\w+)\s+\w+\s*\(")
    with open(verilog_path) as f:
        for line in f:
            m = inst_re.match(line)
            if m and m.group(1) not in _VERILOG_KEYWORDS:
                used.add(m.group(1))
    return used


def _resolve_lib_path() -> Path:
    """Resolve the Liberty .lib file to use for ABC match generation.

    Priority:
      1. GRADMAP_LIB env var (full path to a .lib file)
      2. GRADMAP_LIBS/asap7.lib  (original gradmap convention)
    """
    explicit = os.environ.get("GRADMAP_LIB", "")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise RuntimeError(f"GRADMAP_LIB points to missing file: {p}")
        return p

    gradmap_libs = os.environ.get("GRADMAP_LIBS", "")
    if gradmap_libs:
        p = Path(gradmap_libs) / "asap7.lib"
        if p.exists():
            return p

    raise RuntimeError(
        "No Liberty lib found. Set either:\n"
        "  GRADMAP_LIB=/path/to/file.lib   (single file)\n"
        "  GRADMAP_LIBS=/path/to/dir/       (must contain asap7.lib)"
    )


def _generate_libcell_info(lib_path: Path, out_path: Path, gradmap_root: Path) -> None:
    """Generate asap7_libcell_info.txt from a Liberty .lib file using lut.py."""
    lut_script = gradmap_root / "libs" / "lut.py"
    if not lut_script.exists():
        raise RuntimeError(f"lut.py not found: {lut_script}")
    result = subprocess.run(
        ["python", str(lut_script), "-files", str(lib_path), "-o", str(out_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"lut.py failed (exit {result.returncode}):\n{result.stderr}\n{result.stdout}"
        )


def _generate_match_file(bench_path: str, match_out: str, gradmap_root: Path,
                         lib_path: Path) -> None:
    """Run ABC &nf -Y to produce a match file for gradmap.

    Uses a minimal flow (no rec_start3 / &deepsyn) so that only the Liberty
    .lib file is required — no rec6Lib AIG needed.
    """
    abc = _resolve_abc()
    fd, tcl_path = tempfile.mkstemp(suffix=".tcl", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(
                f"read_lib {lib_path}\n"
                f"read {bench_path}\n"
                f"&get\n"
                f"&nf -Y {match_out}\n"
                f"&put\n"
                f"topo\n"
                f"stime\n"
            )
        result = subprocess.run(
            [abc, "-f", tcl_path],
            cwd=gradmap_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0 or not Path(match_out).exists():
            raise RuntimeError(
                f"ABC match generation failed (exit {result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    finally:
        Path(tcl_path).unlink(missing_ok=True)


def _parse_match_records(match_path: str):
    """Parse match file into tagged records: (kind, data, raw_line)."""
    records = []
    with open(match_path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                records.append(("empty", None, line))
                continue
            parts = stripped.split()
            tok = parts[0]
            if tok.startswith("L") and tok[1:].isdigit():
                records.append(("L", (int(tok[1:]), int(parts[1])), line))
            elif tok.startswith("M") and tok[1:].isdigit():
                gate_type = parts[1] if len(parts) > 1 else ""
                records.append(("M", (int(tok[1:]), gate_type), line))
            elif tok.isdigit():
                gate_type = parts[1] if len(parts) > 1 else ""
                records.append(("gate", (int(tok), gate_type), line))
            else:
                records.append(("other", None, line))
    return records


def _filter_match_file(match_path: str, allowed_gates: Set[str], out_path: str) -> None:
    """Write a filtered match file that keeps only allowed gate types.

    L-line indices are recomputed to account for removed gate options per node.
    M lines for removed gate types are dropped.
    """
    records = _parse_match_records(match_path)

    # Build per-node ordered gate list (index = position in match list for that node)
    node_gates: Dict[int, List[str]] = {}
    for kind, data, _ in records:
        if kind == "gate":
            node_id, gate_type = data
            if gate_type not in ("input", "output"):
                node_gates.setdefault(node_id, []).append(gate_type)

    # old-index → new-index per node after filtering
    node_idx_remap: Dict[int, Dict[int, int]] = {}
    for node_id, gates in node_gates.items():
        mapping: Dict[int, int] = {}
        new_idx = 0
        for old_idx, g in enumerate(gates):
            if g in allowed_gates:
                mapping[old_idx] = new_idx
                new_idx += 1
        node_idx_remap[node_id] = mapping

    out_lines = []
    for kind, data, raw in records:
        if kind == "gate":
            node_id, gate_type = data
            if gate_type in ("input", "output") or gate_type in allowed_gates:
                out_lines.append(raw)
        elif kind == "L":
            node_id, orig_idx = data
            new_idx = node_idx_remap.get(node_id, {}).get(orig_idx, 0)
            out_lines.append(f"L{node_id} {new_idx}")
        elif kind == "M":
            node_id, gate_type = data
            if gate_type in allowed_gates:
                out_lines.append(raw)
            # else: drop; filtered runs use init_weights_strategy=average
        else:
            out_lines.append(raw)

    Path(out_path).write_text("\n".join(out_lines) + "\n")


class GradMapper:
    """gradmap-based technology mapper for gate-usage experiments.

    Usage:
        mapper = GradMapper("benchmarks/c880.bench", output_dir="temp_blifs/")
        delay_a, area_a, v_a = mapper.map_full(tag="pass_a")
        used = mapper.used_gate_names(v_a)
        delay_b, area_b, v_b = mapper.map_filtered(used, tag="pass_b")
        cost = mapper.calculate_cost(delay_b, area_b)
        n = mapper.num_arms   # unique gate types in full ASAP7 match
    """

    def __init__(
        self,
        bench_path: str,
        output_dir: str,
        gradmap_root: Optional[Path] = None,
    ):
        self.bench_path = bench_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradmap_root = gradmap_root or _GRADMAP_ROOT

        binary = os.environ.get(
            "GRADMAP_RUNNER",
            str(self.gradmap_root / "build" / "gradmap_torch"),
        )
        if not Path(binary).is_file():
            raise RuntimeError(
                f"gradmap binary not found: {binary}\n"
                "Build with: cd third_party/gradmap && bash compile.sh"
            )
        self._binary = binary

        self._lib_path = _resolve_lib_path()

        lib_info = self.gradmap_root / "libs" / "asap7_libcell_info.txt"
        if not lib_info.exists():
            print(f"  Generating asap7_libcell_info.txt from {self._lib_path.name}...")
            _generate_libcell_info(self._lib_path, lib_info, self.gradmap_root)

        bench_stem = Path(bench_path).stem
        # _balance suffix so gradmap strips it and names testcase correctly
        self._full_match = str(self.output_dir / f"{bench_stem}_gradmap_balance.txt")

        if not Path(self._full_match).exists():
            print(f"  Generating match file for {bench_stem}...")
            _generate_match_file(bench_path, self._full_match, self.gradmap_root,
                                 self._lib_path)

        self._gate_names: List[str] = self._parse_gate_names()
        self.baseline_delay, self.baseline_area = self._compute_baseline()

    def _parse_gate_names(self) -> List[str]:
        gates: Set[str] = set()
        for kind, data, _ in _parse_match_records(self._full_match):
            if kind == "gate":
                _, gate_type = data
                if gate_type not in ("input", "output"):
                    gates.add(gate_type)
        return sorted(gates)

    def _compute_baseline(self) -> Tuple[float, float]:
        delay, area, _ = self.map_full(tag="baseline")
        return delay, area

    # ------------------------------------------------------------------

    def map_full(self, tag: str = "full") -> Tuple[float, float, str]:
        """Run gradmap with the full ASAP7 match. Returns (delay, area, verilog_path)."""
        return self._run(self._full_match, tag, filtered=False)

    def map_filtered(self, gate_names: Set[str], tag: str = "filtered") -> Tuple[float, float, str]:
        """Run gradmap with match restricted to gate_names. Returns (delay, area, verilog_path)."""
        filtered_match = str(self.output_dir / f"_match_{tag}.txt")
        _filter_match_file(self._full_match, gate_names, filtered_match)
        try:
            return self._run(filtered_match, tag, filtered=True)
        finally:
            Path(filtered_match).unlink(missing_ok=True)

    @staticmethod
    def used_gate_names(verilog_path: str) -> Set[str]:
        return _used_gate_names_from_verilog(verilog_path)

    def calculate_cost(self, delay: float, area: float) -> float:
        """Normalised ADP. Lower is better; baseline = 1.0."""
        if not (delay > 0 and area > 0):
            return float("inf")
        return (delay * area) / (self.baseline_delay * self.baseline_area)

    @property
    def num_arms(self) -> int:
        return len(self._gate_names)

    # ------------------------------------------------------------------

    def _run(self, match_file: str, tag: str, filtered: bool) -> Tuple[float, float, str]:
        template = _CONFIG_FILTERED if filtered else _CONFIG_FULL
        placeholder = str(self.output_dir / "_placeholder.v")
        cfg = str(self.output_dir / f"_cfg_{tag}.txt")
        Path(cfg).write_text(
            template.format(match_file=match_file, output_placeholder=placeholder)
        )
        verilog = str(_actual_verilog(match_file, self.output_dir))
        try:
            stdout = _run_gradmap(self._binary, cfg, self.gradmap_root)
        finally:
            Path(cfg).unlink(missing_ok=True)
        delay, area = _parse_delay_area(stdout)
        return delay, area, verilog
