#!/bin/bash
# submit.sh
# Usage: ./submit.sh <script.sh> [args...]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script.sh> [args...]"
    exit 1
fi

SCRIPT=$1
shift

JOB_NAME=$(basename "$SCRIPT" .sh)
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$(pwd)/slurm_job_log"
mkdir -p "$LOG_DIR"

sbatch \
    --job-name="$JOB_NAME" \
    --output="${LOG_DIR}/${JOB_NAME}_${TS}_%j.log" \
    --error="${LOG_DIR}/${JOB_NAME}_${TS}_%j.err" \
    --ntasks=8 \
    --time=95:59:00 \
    --mem=32G \
    --partition=standard \
    --account=b12901074 \
    "$SCRIPT" "$@"

echo "Submitted: $JOB_NAME"
echo "Log: ${LOG_DIR}/${JOB_NAME}_${TS}_<job_id>.log"