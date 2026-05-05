#!/usr/bin/env bash
# Run random_search.py over all active design × genlib combinations.
# Usage: bash experiment/random/run.sh <num_sampled_gate> [design] [genlib]
#   num_sampled_gate  required
#   design            optional – run only this design (e.g. benchmarks/s838a.bench)
#   genlib            optional – run only this genlib  (e.g. sky130.genlib)
# Must be run from the repo root so relative paths resolve correctly.

set -euo pipefail

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

# ── designs ───────────────────────────────────────────────────────────────────
DESIGNS=(
    # benchmarks/b10.bench
    # benchmarks/b12.bench
    # benchmarks/b14.bench
    # benchmarks/b20_1.bench
    # benchmarks/bar.blif
    # benchmarks/c880.bench
    benchmarks/c1238.bench
    # benchmarks/c1355.bench
    # benchmarks/c5315.bench
    # benchmarks/multiplier.blif
    # benchmarks/ode.abc.blif
    # benchmarks/priority.blif
    # benchmarks/s838a.bench
    # benchmarks/s1488.bench
    # benchmarks/s1494.bench
    # benchmarks/s9234.bench
    # benchmarks/s35932.bench
    # benchmarks/sin.blif
    # benchmarks/sqrt.blif
    # benchmarks/voter.blif
)

# ── genlibs ───────────────────────────────────────────────────────────────────
GENLIBS=(
    7nm.genlib
    # gf180mcu_ff_125C.genlib
    # gf180mcu_tt_025C.genlib
    # nan45.genlib
    # sky130.genlib
)

# ── argument handling ─────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: bash experiment/random/run.sh <num_sampled_gate> [design] [genlib]"
    exit 1
fi

NUM_SAMPLED_GATE="$1"
FILTER_DESIGN="${2:-}"
FILTER_GENLIB="${3:-}"
NUM_ITERATIONS=1000   # must match num_iterations in random_search.py

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR="experiment/random/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DESCRIPTION="random_gates${NUM_SAMPLED_GATE}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${DESCRIPTION}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

SEP="========================================================"

# ── helpers ───────────────────────────────────────────────────────────────────
run_one() {
    local design="$1"
    local genlib="$2"

    echo "$SEP"
    echo "  Script        : experiment/random/random_search.py"
    echo "  Design        : ${design}"
    echo "  Genlib        : ${genlib}"
    echo "  Sampled gates : ${NUM_SAMPLED_GATE}"
    echo "  Iterations    : ${NUM_ITERATIONS}"
    echo "  Started       : $(date)"
    echo "$SEP"

    echo ">>> python experiment/random/random_search.py ${NUM_SAMPLED_GATE} ${design} ${genlib}"
    rm -f .random_progress
    local t0
    t0=$(date +%s)
    python experiment/random/random_search.py "$NUM_SAMPLED_GATE" "$design" "$genlib"
    rm -f .random_progress
    local elapsed=$(( $(date +%s) - t0 ))
    echo "$SEP"
    echo "  Elapsed       : ${elapsed}s"
    echo "$SEP"
}

# ── main loop ─────────────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)
for design in "${DESIGNS[@]}"; do
    [[ -n "$FILTER_DESIGN" && "$design" != "$FILTER_DESIGN" ]] && continue
    for genlib in "${GENLIBS[@]}"; do
        [[ -n "$FILTER_GENLIB" && "$genlib" != "$FILTER_GENLIB" ]] && continue
        run_one "$design" "$genlib"
    done
done
echo "Total elapsed : $(( $(date +%s) - TOTAL_START ))s"
