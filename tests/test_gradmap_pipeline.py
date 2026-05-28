"""
Two-step gradmap pipeline, mirroring what abc_mapper.py does with ABC.

abc_mapper.py (genlib-based):
    read {genlib}; read {design}; map; write {temp_blif};
    read {lib}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;

gradmap equivalent:
    Step 1 — ABC &nf -Y:  generate a match file (replaces `map`)
    Step 2 — gradmap_torch: optimise gate assignment (replaces the stime evaluation)
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

GRADMAP_ROOT = Path(__file__).parent.parent / "third_party" / "gradmap"
TINY_AIG = GRADMAP_ROOT / "regression/cover/tiny.aig"
TINY_MATCH = GRADMAP_ROOT / "regression/cover/tiny_match.txt"
GRADMAP_LIBS = os.environ.get("GRADMAP_LIBS", "")
_BINARY_CANDIDATES = [GRADMAP_ROOT / "gradmap_torch", GRADMAP_ROOT / "build" / "gradmap_torch"]
BINARY = os.environ.get(
    "GRADMAP_RUNNER",
    next((str(p) for p in _BINARY_CANDIDATES if p.is_file()), str(_BINARY_CANDIDATES[0])),
)

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_missing_abc = pytest.mark.skipif(
    shutil.which("abc") is None,
    reason="abc not on PATH",
)
_missing_libs = pytest.mark.skipif(
    not GRADMAP_LIBS
    or not Path(GRADMAP_LIBS, "asap7.lib").exists()
    or not Path(GRADMAP_LIBS, "rec6Lib_final_filtered3_recanon.aig").exists(),
    reason="GRADMAP_LIBS not set or asap7.lib / rec6Lib not found",
)
_missing_binary = pytest.mark.skipif(
    not Path(BINARY).is_file(),
    reason="gradmap binary not built — run compile.sh first",
)

# ---------------------------------------------------------------------------
# Step 1: ABC &nf -Y  (gradmap's counterpart to abc_mapper.py's `map`)
# ---------------------------------------------------------------------------

@_missing_abc
@_missing_libs
def test_abc_generate_match():
    """ABC generates a match file from tiny.aig using &nf -Y (balance mode).

    This is the gradmap equivalent of abc_mapper.py's:
        read {genlib}; read {design}; map; ...
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        match_out = Path(tmpdir) / "tiny_generated.txt"
        tcl = Path(tmpdir) / "run.tcl"
        tcl.write_text(
            f"source abc.rc\n"
            f"read_lib {GRADMAP_LIBS}/asap7.lib\n"
            f"rec_start3 {GRADMAP_LIBS}/rec6Lib_final_filtered3_recanon.aig\n"
            f"read {TINY_AIG}\n"
            f"&get\n"
            f"&if -y -K 6; &put; resyn2; resyn2; &get;\n"
            f"&deepsyn -T 30\n"
            f"&nf -Y {match_out}\n"
            f"&put\n"
            f"topo\n"
            f"stime\n"
        )

        result = subprocess.run(
            ["abc", "-f", str(tcl)],
            cwd=GRADMAP_ROOT,  # so `source abc.rc` resolves
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"abc exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert match_out.exists() and match_out.stat().st_size > 0, (
            "abc did not write a match file"
        )


# ---------------------------------------------------------------------------
# Step 2: gradmap_torch  (gradmap's counterpart to abc_mapper.py's stime eval)
# ---------------------------------------------------------------------------

_GRADMAP_CONFIG = """\
flow true
regression false
testcase.lib libs/asap7_libcell_info.txt
testcase.match {match}
output.verilog {output_verilog}
optimizer.method torch
optimizer.total_steps 5
optimizer.eval_interval 1
circuit.init_weights_strategy average
"""

@_missing_binary
def test_gradmap_torch_with_match():
    """gradmap_torch optimises gate assignment using the pre-built tiny match file.

    This is the gradmap equivalent of abc_mapper.py's:
        read {lib}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config = tmpdir / "config.txt"
        config.write_text(
            _GRADMAP_CONFIG.format(
                match=TINY_MATCH,
                output_verilog=tmpdir / "placeholder.v",
            )
        )

        result = subprocess.run(
            [BINARY, str(config)],
            cwd=GRADMAP_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"gradmap_torch exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        # gradmap writes {outdir}/{testcase}_best.v; tiny_match.txt → tiny_match_best.v
        expected_v = tmpdir / "tiny_match_best.v"
        assert expected_v.exists(), f"gradmap_torch did not write {expected_v}"


@_missing_binary
def test_gradmap_full_lib_used_gates():
    """Run gradmap with the full tiny match file and print which gates are used.

    Verifies the full-lib → used-gates pipeline works on the smallest available fixture.
    """
    import re

    _VERILOG_KEYWORDS = frozenset(
        ("module", "endmodule", "input", "output", "wire", "reg",
         "assign", "always", "begin", "end", "if", "else")
    )

    def used_gates_from_verilog(verilog_path):
        used = set()
        inst_re = re.compile(r"^\s*(\w+)\s+\w+\s*\(")
        with open(verilog_path) as f:
            for line in f:
                m = inst_re.match(line)
                if m and m.group(1) not in _VERILOG_KEYWORDS:
                    used.add(m.group(1))
        return used

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config = tmpdir / "config.txt"
        config.write_text(
            _GRADMAP_CONFIG.format(
                match=TINY_MATCH,
                output_verilog=tmpdir / "placeholder.v",
            )
        )

        result = subprocess.run(
            [BINARY, str(config)],
            cwd=GRADMAP_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"gradmap_torch exited {result.returncode}\n{result.stdout}\n{result.stderr}"
        )

        verilog = tmpdir / "tiny_match_best.v"
        assert verilog.exists(), "gradmap_torch did not write output verilog"

        used = used_gates_from_verilog(str(verilog))

        print(f"\nOutput verilog:\n{verilog.read_text()}")
        print(f"Gates used: {sorted(used) if used else '(none — circuit may be trivial or lib cells mismatched)'}")
