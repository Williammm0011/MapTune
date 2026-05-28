import os
import pathlib
import subprocess
import tempfile

import pytest

GRADMAP_ROOT = pathlib.Path(__file__).parent.parent / "third_party" / "gradmap"
_BINARY_CANDIDATES = [GRADMAP_ROOT / "gradmap_torch", GRADMAP_ROOT / "build" / "gradmap_torch"]
BINARY = os.environ.get(
    "GRADMAP_RUNNER",
    next((str(p) for p in _BINARY_CANDIDATES if p.is_file()), str(_BINARY_CANDIDATES[0])),
)

_CONFIG_TEMPLATE = """\
flow true
regression false
testcase.lib libs/asap7_libcell_info.txt
testcase.match regression/cover/tiny_match.txt
output.verilog {output_verilog}
optimizer.method torch
optimizer.total_steps 5
optimizer.eval_interval 1
circuit.init_weights_strategy average
"""


@pytest.mark.skipif(not os.path.isfile(BINARY), reason="gradmap binary not built — run compile.sh first")
def test_gradmap_runs_on_tiny():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        # gradmap writes {dirname(output.verilog)}/{testcase}_best.v
        # testcase = match file stem with _balance/_area/_delay stripped
        # tiny_match.txt → testcase = "tiny_match" → output = tiny_match_best.v
        config_path = tmpdir / "test_config.txt"
        config_path.write_text(_CONFIG_TEMPLATE.format(output_verilog=tmpdir / "placeholder.v"))

        result = subprocess.run(
            [BINARY, str(config_path)],
            cwd=GRADMAP_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"gradmap exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        expected_v = tmpdir / "tiny_match_best.v"
        assert expected_v.exists(), (
            f"gradmap did not write output verilog at {expected_v}"
        )
