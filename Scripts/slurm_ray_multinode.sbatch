#!/bin/bash
#SBATCH --job-name=regraph-ray
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-ray-%j.out
#SBATCH --error=logs/slurm-ray-%j.err

set -euo pipefail

# =========================
# User-tunable parameters
# =========================
RAY_PORT="${RAY_PORT:-6379}"
INPUT_PATH="${INPUT_PATH:-/path/to/Binaries/DataProcess-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/output}"
START_FROM="${START_FROM:-1}"
ENABLE_RECOMPILE="${ENABLE_RECOMPILE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RAY_TASK_CPUS="${RAY_TASK_CPUS:-1}"
TASK2_TIMEOUT="${TASK2_TIMEOUT:-3600}"
TASK3_TIMEOUT="${TASK3_TIMEOUT:-3600}"
TASK4_TIMEOUT="${TASK4_TIMEOUT:-3600}"
SIF_IMAGE="${SIF_IMAGE:-}"
HEAD_IP_OVERRIDE="${HEAD_IP_OVERRIDE:-}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs

# If you use conda/venv, activate it here.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ReLL
# If needed on IBEX:
# module load singularity

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"

if [ -n "$HEAD_IP_OVERRIDE" ]; then
  HEAD_IP="$HEAD_IP_OVERRIDE"
else
  HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "getent hosts \"$HEAD_NODE\" | awk 'NR==1{print \\$1}'")
  if [ -z "$HEAD_IP" ]; then
    echo "[ERROR] Failed to resolve HEAD_IP for $HEAD_NODE"
    exit 1
  fi
fi

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export SLURM_RAY_HEAD_IP="$HEAD_IP"

echo "[INFO] Nodes: ${NODES[*]}"
echo "[INFO] Head node: $HEAD_NODE"
echo "[INFO] Ray address: $RAY_ADDRESS"

RAY_LAUNCH_PREFIX=()
PYTHON_LAUNCH_PREFIX=()
if [ -n "$SIF_IMAGE" ]; then
  echo "[INFO] Using container image: $SIF_IMAGE"
  if command -v apptainer >/dev/null 2>&1; then
    RAY_LAUNCH_PREFIX=(apptainer exec --cleanenv "$SIF_IMAGE")
    PYTHON_LAUNCH_PREFIX=(apptainer exec --cleanenv "$SIF_IMAGE")
  else
    RAY_LAUNCH_PREFIX=(singularity exec --cleanenv "$SIF_IMAGE")
    PYTHON_LAUNCH_PREFIX=(singularity exec --cleanenv "$SIF_IMAGE")
  fi
fi

cleanup() {
  echo "[INFO] Stopping Ray on all nodes"
  if [ -n "$SIF_IMAGE" ]; then
    if command -v apptainer >/dev/null 2>&1; then
      srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
        apptainer exec --cleanenv "$SIF_IMAGE" ray stop --force || true
    else
      srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
        singularity exec --cleanenv "$SIF_IMAGE" ray stop --force || true
    fi
  else
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" ray stop --force || true
  fi
}
trap cleanup EXIT

# Start Ray head (background)
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
  "${RAY_LAUNCH_PREFIX[@]}" ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --block \
  > "logs/ray-head-${SLURM_JOB_ID}.log" 2>&1 &

sleep 10

# Start Ray workers on remaining nodes
if [ "${#NODES[@]}" -gt 1 ]; then
  for NODE in "${NODES[@]:1}"; do
    echo "[INFO] Starting worker on $NODE"
    srun --nodes=1 --ntasks=1 -w "$NODE" \
      "${RAY_LAUNCH_PREFIX[@]}" ray start --address="$RAY_ADDRESS" --block \
      > "logs/ray-worker-${NODE}-${SLURM_JOB_ID}.log" 2>&1 &
  done
fi

sleep 10

PIPELINE_CMD=(
  "${PYTHON_LAUNCH_PREFIX[@]}" python3 Scripts/pipeline_ray.py
  --input-path "$INPUT_PATH"
  --output "$OUTPUT_ROOT"
  --start-from "$START_FROM"
  --ray-mode slurm
  --batch-size "$BATCH_SIZE"
  --ray-task-cpus "$RAY_TASK_CPUS"
  --task2-timeout "$TASK2_TIMEOUT"
  --task3-timeout "$TASK3_TIMEOUT"
  --task4-timeout "$TASK4_TIMEOUT"
)

if [ "$ENABLE_RECOMPILE" = "1" ]; then
  PIPELINE_CMD+=(--enable-recompile)
fi

echo "[INFO] Running pipeline on head node"
echo "[INFO] ${PIPELINE_CMD[*]}"

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" "${PIPELINE_CMD[@]}"
