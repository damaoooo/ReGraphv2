#!/bin/bash
#SBATCH --job-name=regraph_qwen_text
#SBATCH --partition=workq
#SBATCH --exclusive
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=384
#SBATCH --mem=0
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/zhoul0e/ReGraphv2/Scripts/ray_opt_ablation/slurm_logs/regraph_qwen_text-%j.out
#SBATCH --error=/scratch/zhoul0e/ReGraphv2/Scripts/ray_opt_ablation/slurm_logs/regraph_qwen_text-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/zhoul0e/ReGraphv2}"
INNER="${REPO_ROOT}/Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch"
OPT_ARG="${1:-${OPT_LEVEL:-}}"

export TASK3_RECORD_FORMAT="qwen-text"
export SPLITS="${SPLITS:-test}"
export FINAL_UNINFORMATIVE_FILTER="${FINAL_UNINFORMATIVE_FILTER:-1}"
export FINAL_UNINFORMATIVE_FILTER_SPLITS="${FINAL_UNINFORMATIVE_FILTER_SPLITS:-test}"
QWEN_PREFILTER_UNINFORMATIVE="${QWEN_PREFILTER_UNINFORMATIVE:-1}"
QWEN_EXACT_DEDUP="${QWEN_EXACT_DEDUP:-1}"

if [[ -z "${OUTPUT_PATH:-}" && -n "${OPT_ARG}" ]]; then
  if [[ -n "${DATASET_PATH:-}" ]]; then
    qwen_dataset_path="${DATASET_PATH}"
  elif [[ "${SMOKE:-0}" == "1" ]]; then
    qwen_dataset_path="${DEFAULT_SMOKESET_PATH:-/scratch/zhoul0e/Dataset-smoketest}"
  else
    qwen_dataset_path="${DEFAULT_DATASET_PATH:-/scratch/zhoul0e/Dataset-1}"
  fi
  opt_token="${OPT_ARG#-}"
  opt_token="${opt_token//\//_}"
  OUTPUT_PATH="$(dirname "${qwen_dataset_path}")/$(basename "${qwen_dataset_path}")-${opt_token}-qwen-text-fused"
  export OUTPUT_PATH
fi

qwen_args=()
if [[ "${QWEN_PREFILTER_UNINFORMATIVE}" == "1" ]]; then
  qwen_args+=(--task3-prefilter-uninformative)
fi
if [[ "${QWEN_EXACT_DEDUP}" == "1" ]]; then
  qwen_args+=(--task3-dedup-input-ids)
fi
if [[ -n "${DRIVER_EXTRA_ARGS:-}" ]]; then
  DRIVER_EXTRA_ARGS="${qwen_args[*]} ${DRIVER_EXTRA_ARGS}"
else
  DRIVER_EXTRA_ARGS="${qwen_args[*]}"
fi
export DRIVER_EXTRA_ARGS

exec "${INNER}" "$@"
