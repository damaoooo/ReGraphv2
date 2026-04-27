#!/usr/bin/env bash
set -euo pipefail

SHARDS="${1:-${SHARDS:-2}}"
INPUT="${INPUT:-/scratch/zhoul0e/Dataset-1}"
OUTPUT="${OUTPUT:-/scratch/zhoul0e/Dataset-1-lift}"
PARTITION="${PARTITION:-workq}"
WORKERS="${WORKERS:-384}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
TIME_LIMIT="${TIME_LIMIT:-23:00:00}"
HOME_FOR_IDA="${HOME_FOR_IDA:-/scratch/zhoul0e}"
IDA_DIR="${IDADIR:-/scratch/zhoul0e/ida-pro-9.3}"
RELL_PYTHON="${REGRAPH_RELL_PYTHON:-/scratch/zhoul0e/miniconda3/envs/ReLL/bin/python}"
FORCE="${FORCE:-0}"

if ! [[ "$SHARDS" =~ ^[0-9]+$ ]] || (( SHARDS < 1 )); then
  echo "SHARDS must be a positive integer; got: $SHARDS" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/task1_lift_multinode.py"
LOG_DIR="$SCRIPT_DIR/slurm_logs/task1_multinode"
mkdir -p "$LOG_DIR"

if [[ "$FORCE" != "1" ]]; then
  active="$(squeue -h -u "$USER" -o '%i %j %t' | awk '$2 ~ /^regraph_task1_/ && ($3 == "R" || $3 == "PD") {print}')"
  if [[ -n "$active" ]]; then
    echo "Existing task1 Slurm job(s) found; refusing to submit overlapping shards." >&2
    echo "$active" >&2
    echo "Wait for them to finish/cancel them, or rerun with FORCE=1." >&2
    exit 3
  fi
fi

last=$((SHARDS - 1))
echo "Submitting task1 multi-node lift: shards=$SHARDS workers_per_node=$WORKERS input=$INPUT output=$OUTPUT"

sbatch --job-name=regraph_task1_mn \
  --partition="$PARTITION" \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$WORKERS" \
  --mem=0 \
  --time="$TIME_LIMIT" \
  --array=0-"$last" \
  --output="$LOG_DIR/task1_mn-%A_%a.out" \
  --error="$LOG_DIR/task1_mn-%A_%a.err" \
  --export=ALL,HOME="$HOME_FOR_IDA",REGRAPH_IDA_HOME="$HOME_FOR_IDA",IDADIR="$IDA_DIR",REGRAPH_RELL_PYTHON="$RELL_PYTHON",REGRAPH_TASK1_WORKERS="$WORKERS",REGRAPH_TASK1_TIMEOUT_SECONDS="$TIMEOUT_SECONDS",REGRAPH_TASK1_ALLOW_FAILURES=1 \
  --wrap="cd '$REPO_ROOT' && '$RELL_PYTHON' '$RUNNER' --input-path '$INPUT' --output '$OUTPUT' --num-shards '$SHARDS' --shard-index \$SLURM_ARRAY_TASK_ID --workers '$WORKERS' --timeout-seconds '$TIMEOUT_SECONDS' --allow-failures --resume"
