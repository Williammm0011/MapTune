"""
Driver script: runs sa_multi.py and random_multi.py on every (lib, bench) combo.

Default: 2 libs × 10 benchmarks × 2 methods = 40 sequential runs.
Each sub-script already parallelises internally (--n-agents workers).
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

_SRC = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LIBS = ["7nm", "sky130"]

DEFAULT_BENCHES = [
    # ISCAS-85 combinational
    "benchmarks/c880.bench",
    "benchmarks/c1238.bench",
    "benchmarks/c1355.bench",
    "benchmarks/c5315.bench",
    # ITC-99 larger combinational
    "benchmarks/b10.bench",
    "benchmarks/b12.bench",
    "benchmarks/b14.bench",
    "benchmarks/b20_1.bench",
    # ISCAS-89 sequential (first 2)
    "benchmarks/s838a.bench",
    "benchmarks/s1488.bench",
]

METHODS = ["sa_multi.py", "random_multi.py"]


def parse_args():
    p = argparse.ArgumentParser(description="Sweep sa_multi + random_multi over libs × benches")
    p.add_argument("--libs", nargs="+", default=DEFAULT_LIBS,
                   help="Library names to sweep")
    p.add_argument("--benches", nargs="+", default=DEFAULT_BENCHES,
                   help="Benchmark paths to sweep")
    p.add_argument("--iterations", type=int, default=None,
                   help="Forwarded to sub-scripts (omit to use their defaults)")
    p.add_argument("--n-agents", type=int, default=None,
                   help="Forwarded to sub-scripts (omit to use their defaults)")
    return p.parse_args()


def run_one(script: str, lib: str, bench: str, log_dir: str, extra_args: list):
    cmd = [sys.executable, os.path.join(_SRC, script),
           "--lib", lib, "--bench", bench,
           "--log-dir", log_dir, *extra_args]
    t0 = time.time()
    result = subprocess.run(cmd)
    return result.returncode, round(time.time() - t0, 1)


def main():
    args = parse_args()

    extra = []
    if args.iterations is not None:
        extra += ["--iterations", str(args.iterations)]
    if args.n_agents is not None:
        extra += ["--n-agents", str(args.n_agents)]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_dir = f"logs/{timestamp}_batch_run"

    combos = [
        (lib, bench, method)
        for lib in args.libs
        for bench in args.benches
        for method in METHODS
    ]
    total = len(combos)

    print(f"\nSweep: {len(args.libs)} libs × {len(args.benches)} benches × {len(METHODS)} methods = {total} runs")
    print(f"Logs:  {batch_log_dir}/")
    print(f"Extra args: {extra if extra else '(none)'}\n")

    results = []
    for i, (lib, bench, method) in enumerate(combos, 1):
        bench_stem = os.path.splitext(os.path.basename(bench))[0]
        method_name = method.replace(".py", "")
        print(f"[{i:>3}/{total}] {lib:<20} | {bench_stem:<12} | {method_name} ...", flush=True)
        rc, elapsed = run_one(method, lib, bench, batch_log_dir, extra)
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"         → {status}  ({elapsed}s)\n")
        results.append((lib, bench_stem, method_name, status, elapsed))

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Lib':<22} | {'Bench':<14} | {'Method':<14} | {'Status':<10} | {'Time(s)':>7}")
    print("-" * 72)
    for lib, bench_stem, method_name, status, elapsed in results:
        print(f"{lib:<22} | {bench_stem:<14} | {method_name:<14} | {status:<10} | {elapsed:>7.1f}")
    print("=" * 72)

    failures = [r for r in results if r[3] != "OK"]
    if failures:
        print(f"\n{len(failures)} run(s) FAILED:")
        for lib, bench_stem, method_name, status, _ in failures:
            print(f"  {lib} / {bench_stem} / {method_name} → {status}")
        sys.exit(1)
    else:
        print(f"\nAll {total} runs completed successfully.")


if __name__ == "__main__":
    main()
