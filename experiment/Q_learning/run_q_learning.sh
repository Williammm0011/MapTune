#!/usr/bin/env bash
# Run batched_DDQN over explicit (design, genlib, num_sampled_gate) combinations.
# Usage: bash experiment/run_q_learning.sh
# Must be run from the repo root so relative paths resolve correctly.
#
# Library cell counts
# ┌─────────────────┬────────────────┐
# │ Library         │ Cell count     │
# ├─────────────────┼────────────────┤
# │ ASAP7  (7nm)    │  45 – 135      │
# │ NAN45           │  35 –  75      │
# │ SKY130          │ 220 – 310      │
# │ GF180           │  40 – 130      │
# └─────────────────┴────────────────┘
#
# Available designs (for reference):
#   benchmarks/b10.bench        benchmarks/b12.bench
#   benchmarks/b14.bench        benchmarks/b20_1.bench
#   benchmarks/bar.blif         benchmarks/c880.bench
#   benchmarks/c1238.bench      benchmarks/c1355.bench
#   benchmarks/c5315.bench      benchmarks/multiplier.blif
#   benchmarks/ode.abc.blif     benchmarks/priority.blif
#   benchmarks/s838a.bench      benchmarks/s1488.bench
#   benchmarks/s1494.bench      benchmarks/s9234.bench
#   benchmarks/s35932.bench     benchmarks/sin.blif
#   benchmarks/sqrt.blif        benchmarks/voter.blif
#
# Available genlibs (for reference):
#   7nm.genlib  gf180mcu_ff_125C.genlib  gf180mcu_tt_025C.genlib
#   nan45.genlib  sky130.genlib

set -euo pipefail

trap 'kill $(jobs -p) 2>/dev/null || true; rm -f .q_learning_progress' EXIT INT TERM

# ── runs: "design genlib num_sampled_gate algo" ───────────────────────────────
# algo: DQN | DDQN
RUNS=(
    # "benchmarks/c1238.bench    7nm.genlib   50 DDQN"
    "benchmarks/priority.blif 7nm.genlib   50 DDQN"
    # "benchmarks/s35932.bench   7nm.genlib   50 DDQN"
    # "benchmarks/c1238.bench    nan45.genlib 50 DDQN"
    # "benchmarks/priority.blif nan45.genlib 50 DDQN"
    # "benchmarks/s35932.bench   nan45.genlib 50 DDQN"
)

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR="experiment/q_learning/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SEP="========================================================"

# ── helpers ───────────────────────────────────────────────────────────────────
run_one() {
    local design="$1"
    local genlib="$2"
    local num_sampled_gate="$3"
    local algo="$4"
    local script="batched_${algo}.py"

    echo "$SEP"
    echo "  Script        : ${script}"
    echo "  Design        : ${design}"
    echo "  Genlib        : ${genlib}"
    echo "  Sampled gates : ${num_sampled_gate}"
    echo "  Algorithm     : ${algo}"
    echo "  Started       : $(date)"
    echo "$SEP"

    echo ">>> python ${script} ${num_sampled_gate} ${design} ${genlib}"
    rm -f .q_learning_progress
    local t0
    t0=$(date +%s)
    python "$script" "$num_sampled_gate" "$design" "$genlib"
    rm -f .q_learning_progress
    local elapsed=$(( $(date +%s) - t0 ))
    echo "$SEP"
    echo "  Elapsed       : ${elapsed}s"
    echo "$SEP"
}

# ── main loop ─────────────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)
for run in "${RUNS[@]}"; do
    read -r design genlib num_sampled_gate algo <<< "$run"
    design_base="${design##*/}"; design_base="${design_base%.*}"
    lib_base="${genlib%.*}"
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${algo}_${design_base}_${lib_base}_${num_sampled_gate}.txt"
    echo "Log: $LOG_FILE"
    run_one "$design" "$genlib" "$num_sampled_gate" "$algo" 2>&1 | tee -a "$LOG_FILE"
done
echo "Total elapsed : $(( $(date +%s) - TOTAL_START ))s"
