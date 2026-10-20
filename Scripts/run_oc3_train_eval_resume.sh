#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/damaoooo/Downloads/regraphv2}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/IR/Dataset-1-Oc3-fused}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/dataset1_oc3_fused}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TRAIN_SET="${TRAIN_SET:-}"
if [[ -z "${TRAIN_SET}" ]]; then
  for candidate in \
    "${DATASET_ROOT}/train_final_set_common_oc3" \
    "${DATASET_ROOT}/train_final_set_common_oc" \
    "${DATASET_ROOT}/train_final_set_len128_hashdedup" \
    "${DATASET_ROOT}/train_final_set"; do
    if [[ -d "${candidate}" ]]; then
      TRAIN_SET="${candidate}"
      break
    fi
  done
fi
if [[ -z "${TRAIN_SET}" ]]; then
  TRAIN_SET="${DATASET_ROOT}/train_final_set"
fi
VALIDATION_SET="${VALIDATION_SET:-}"
if [[ -z "${VALIDATION_SET}" ]]; then
  for candidate in \
    "${DATASET_ROOT}/validation_final_set_common_oc3" \
    "${DATASET_ROOT}/validation_final_set_common_oc" \
    "${DATASET_ROOT}/validation_final_set_len128_hashdedup" \
    "${DATASET_ROOT}/validation_final_set"; do
    if [[ -d "${candidate}" ]]; then
      VALIDATION_SET="${candidate}"
      break
    fi
  done
fi
if [[ -z "${VALIDATION_SET}" ]]; then
  VALIDATION_SET="${DATASET_ROOT}/validation_final_set"
fi
TEST_SET="${TEST_SET:-}"
if [[ -z "${TEST_SET}" ]]; then
  for candidate in \
    "${DATASET_ROOT}/test_final_set_common_oc3" \
    "${DATASET_ROOT}/test_final_set_common_oc" \
    "${DATASET_ROOT}/test_final_set_len128_hashdedup" \
    "${DATASET_ROOT}/test_final_set"; do
    if [[ -d "${candidate}" ]]; then
      TEST_SET="${candidate}"
      break
    fi
  done
fi
if [[ -z "${TEST_SET}" ]]; then
  TEST_SET="${DATASET_ROOT}/test_final_set"
fi

TEST_TAG="$(basename "${TEST_SET}")"
TEST_POOL="${TEST_SET}/train_dataset_pool"
TEST_MAP="${TEST_SET}/train_positive_map.pkl"

CHECKPOINT_PREFIX="${RUN_ROOT}/checkpoints"
TENSORBOARD_PREFIX="${RUN_ROOT}/tensorboard"
MODEL_PREFIX="${RUN_ROOT}/model"
MODEL_DIR="${MODEL_PREFIX}_cfg_ddg"
EMBEDDINGS_PATH="${RUN_ROOT}/${TEST_TAG}_full_embeddings_cfg_ddg_${RUN_ID}.pth"
MARKDOWN_PATH="${RUN_ROOT}/${TEST_TAG}_full_results_${RUN_ID}.md"

MAX_LENGTH="${MAX_LENGTH:-2048}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
MAX_STEPS="${MAX_STEPS:-300000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"
POOL_SAMPLES="${POOL_SAMPLES:-0}"

set +u
source "/home/damaoooo/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-ReLL}"
set -u

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"

for required_path in "${TRAIN_SET}" "${VALIDATION_SET}" "${TEST_POOL}" "${TEST_MAP}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Error: required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

echo "[OC3] dataset_root=${DATASET_ROOT}"
echo "[OC3] run_root=${RUN_ROOT}"
echo "[OC3] checkpoint_dir=${CHECKPOINT_PREFIX}_cfg_ddg"
echo "[OC3] model_dir=${MODEL_DIR}"
echo "[OC3] train_set=${TRAIN_SET}"
echo "[OC3] validation_set=${VALIDATION_SET}"
echo "[OC3] test_set=${TEST_SET}"
echo "[OC3] embeddings_path=${EMBEDDINGS_PATH}"
echo "[OC3] markdown_path=${MARKDOWN_PATH}"
echo "[OC3] max_length=${MAX_LENGTH}"
echo "[OC3] max_steps=${MAX_STEPS}"

python -m Pretrain.run_pretrain train \
  --dataset-dir "${TRAIN_SET}" \
  --validation-dataset-dir "${VALIDATION_SET}" \
  --cfg \
  --ddg \
  --resume \
  --set "max_seq_length=${MAX_LENGTH}" \
  --set "save_steps=${SAVE_STEPS}" \
  --set "max_steps=${MAX_STEPS}" \
  --set "output_dir=${CHECKPOINT_PREFIX}" \
  --set "logging_dir=${TENSORBOARD_PREFIX}" \
  --set "final_model_dir=${MODEL_PREFIX}" \
  --set "report_to=tensorboard"

python evaluation.py "${TEST_MAP}" \
  --dataset-path "${TEST_POOL}" \
  --model-path "${MODEL_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --gpu-batch-size "${GPU_BATCH_SIZE}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --pool-samples "${POOL_SAMPLES}" \
  --embeddings-path "${EMBEDDINGS_PATH}" \
  --markdown-output "${MARKDOWN_PATH}" \
  --cfg \
  --ddg \
  --bf16

echo "[OC3] done"
echo "[OC3] markdown_path=${MARKDOWN_PATH}"
