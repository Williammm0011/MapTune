"""GradMapper — class interface for GradMap technology mapping.

Mirrors the TechMapper (ABC) interface: construct with a benchmark + config,
call run() to optimize, call calculate_reward() to score the result.
"""
from __future__ import annotations

import collections
import math
import os
import re
import subprocess

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GradMapper:
    """Wraps the gradmap_torch binary for a single benchmark run."""

    def __init__(self, benchmark: str, cfg: dict, steps: int | None = None):
        """
        benchmark : name from benchmarks_gradmap/ (e.g. "adder") or path to .txt
        cfg       : dict loaded from config.toml via tomllib
        steps     : optimizer steps; falls back to cfg["gradmap"]["default_steps"]
        """
        gcfg = cfg["gradmap"]
        self._gradmap_dir = os.path.join(_REPO_ROOT, gcfg["gradmap_dir"])
        self._lib         = gcfg["lib"]                    # relative to gradmap_dir
        self._steps       = steps if steps is not None else gcfg["default_steps"]
        self._opt         = dict(gcfg["optimizer"])
        self._circuit     = dict(gcfg["circuit"])

        # Resolve match file
        if os.path.isfile(benchmark):
            self._match_path = os.path.abspath(benchmark)
            self.bench_stem  = os.path.splitext(os.path.basename(benchmark))[0]
        else:
            bench_dir        = os.path.join(_REPO_ROOT, gcfg["bench_dir"])
            self._match_path = os.path.join(bench_dir, f"{benchmark}.txt")
            self.bench_stem  = benchmark

        if not os.path.isfile(self._match_path):
            bench_dir = os.path.join(_REPO_ROOT, gcfg["bench_dir"])
            available = ", ".join(
                os.path.splitext(f)[0]
                for f in sorted(os.listdir(bench_dir))
                if f.endswith(".txt")
            )
            raise FileNotFoundError(
                f"Match file not found: {self._match_path}\n"
                f"Available: {available}"
            )

        self._config_path  = os.path.join(
            self._gradmap_dir, "config", f"{self.bench_stem}_quick_config.txt"
        )
        # gradmap derives verilog output name from the match file stem
        self._verilog_path = os.path.join(
            self._gradmap_dir, "verilog_output", f"{self.bench_stem}_best.v"
        )

        self.baseline_delay: float | None = None
        self.baseline_area:  float | None = None
        self.used_cells: collections.Counter = collections.Counter()
        self.last_output: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True, progress: bool = False) -> tuple[float, float]:
        """Run gradmap optimizer. Returns (best_delay, best_area).

        Populates self.baseline_delay, self.baseline_area, self.used_cells.
        """
        self._write_config()
        output = self._run_binary(verbose=verbose, progress=progress)
        self.last_output = output
        self._parse_output(output)
        self.used_cells = self._parse_used_cells(self._verilog_path)
        if self.baseline_delay is None:
            raise RuntimeError("gradmap did not report a baseline delay — check output above")
        if self.baseline_area is None:
            raise RuntimeError("gradmap did not report a baseline area — check output above")
        best_delay = self._best_delay
        best_area  = self._best_area
        return best_delay, best_area

    def calculate_reward(self, delay: float, area: float) -> float:
        """−√(delay/baseline_delay × area/baseline_area). Returns -inf on invalid input."""
        if self.baseline_delay is None or self.baseline_area is None:
            raise RuntimeError("Call run() first to establish baseline")
        if delay <= 0 or area <= 0 or not math.isfinite(delay) or not math.isfinite(area):
            return float("-inf")
        return -math.sqrt((delay / self.baseline_delay) * (area / self.baseline_area))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_config(self) -> None:
        rel_match = os.path.relpath(
            os.path.abspath(self._match_path), os.path.abspath(self._gradmap_dir)
        )
        o = self._opt
        c = self._circuit
        config = (
            f"flow true\n"
            f"testcase.lib {self._lib}\n"
            f"testcase.match {rel_match}\n"
            f"output.verilog verilog_output/{self.bench_stem}_quick_best.v\n\n"
            f"optimizer.method torch\n"
            f"optimizer.eval_backend {o['eval_backend']}\n"
            f"optimizer.loss_type {o['loss_type']}\n"
            f"optimizer.area_factor {o['area_factor']}\n"
            f"optimizer.delay_factor {o['delay_factor']}\n"
            f"optimizer.learning_rate {o['learning_rate']}\n"
            f"optimizer.total_steps {self._steps}\n"
            f"optimizer.eval_interval {o['eval_interval']}\n\n"
            f"optimizer.plateau_enable {str(o['plateau_enable']).lower()}\n"
            f"optimizer.plateau_patience {o['plateau_patience']}\n"
            f"optimizer.plateau_threshold {o['plateau_threshold']}\n"
            f"optimizer.plateau_factor {o['plateau_factor']}\n"
            f"optimizer.plateau_min_lr {o['plateau_min_lr']}\n"
            f"optimizer.restore_best_on_plateau {str(o['restore_best_on_plateau']).lower()}\n\n"
            f"optimizer.softmax_temperature_start {o['softmax_temperature_start']}\n"
            f"optimizer.softmax_temperature_end   {o['softmax_temperature_end']}\n\n"
            f"circuit.init_weights_strategy {c['init_weights_strategy']}\n"
            f"circuit.abc_boost_value {c['abc_boost_value']}\n"
        )
        with open(self._config_path, "w") as f:
            f.write(config)

    def _run_binary(self, verbose: bool, progress: bool) -> str:
        binary = os.path.join(self._gradmap_dir, "gradmap_torch")
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"gradmap_torch not found at {binary}\n"
                f"Build with: cd {self._gradmap_dir} && bash compile.sh"
            )
        rel_config = os.path.relpath(self._config_path, self._gradmap_dir)
        proc = subprocess.Popen(
            [binary, rel_config],
            cwd=self._gradmap_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        for line in proc.stdout:
            if verbose:
                print(line, end="", flush=True)
            elif progress and self._is_progress_line(line):
                print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
        output = "".join(lines)
        if proc.returncode != 0:
            self.last_output = output
            raise RuntimeError(f"gradmap_torch exited with code {proc.returncode}")
        return output

    @staticmethod
    def _is_progress_line(line: str) -> bool:
        progress_patterns = (
            r"\[Flow\] Auto-calculated baseline_(delay|area):",
            r"\[TorchOptimizer\] Loop Start",
            r"^\s*\d+\s+\| .* \[Hard\] Loss=",
            r"\[Save\] New Best Found!",
            r"\[TorchOptimizer\] Finished\.",
            r"-> Restoring best result:",
        )
        return any(re.search(pattern, line) for pattern in progress_patterns)

    def _parse_output(self, output: str) -> None:
        baseline_delay_pat = re.compile(r"\[Flow\] Auto-calculated baseline_delay:\s*([0-9.]+)")
        baseline_area_pat  = re.compile(r"\[Flow\] Auto-calculated baseline_area:\s*([0-9.]+)")
        cost_pat  = re.compile(r"Cost=([0-9.]+)")
        delay_pat = re.compile(r"Delay=([0-9.]+)")
        area_pat  = re.compile(r"Area=([0-9.]+)")

        best_record = None
        fallback_record = None
        for line in output.splitlines():
            m = baseline_delay_pat.search(line)
            if m:
                self.baseline_delay = float(m.group(1))
            m = baseline_area_pat.search(line)
            if m:
                self.baseline_area = float(m.group(1))
            if "New Best Found" in line or "Restoring best" in line:
                cost = cost_pat.search(line)
                delay = delay_pat.search(line)
                area = area_pat.search(line)
                if cost and delay and area:
                    record = (
                        float(cost.group(1)),
                        float(delay.group(1)),
                        float(area.group(1)),
                    )
                    if "Restoring best" in line:
                        best_record = record
                    else:
                        fallback_record = record

        record = best_record or fallback_record
        if record is None:
            self._best_adp = float("inf")
            self._best_delay = float("inf")
            self._best_area = float("inf")
            return

        self._best_adp, self._best_delay, self._best_area = record

    @staticmethod
    def _parse_used_cells(verilog_path: str) -> collections.Counter:
        counts: collections.Counter = collections.Counter()
        if not os.path.isfile(verilog_path):
            return counts
        cell_pat = re.compile(r"^(\w+)\s+g\d+\s*\(")
        with open(verilog_path) as f:
            for line in f:
                m = cell_pat.match(line.strip())
                if m:
                    counts[m.group(1)] += 1
        return counts
