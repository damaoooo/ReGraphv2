#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/damaoooo/Downloads/regraphv2}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/IR/Dataset-1-Oc2-fused}"
REFERENCE_TEST_SET="${REFERENCE_TEST_SET:-${REPO_ROOT}/IR/Dataset-1-new/Dataset-1-Oc-fused/test_final_set}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/dataset1_oc2_fused}"

TRAIN_SET="${DATASET_ROOT}/train_final_set"
VALIDATION_SET="${DATASET_ROOT}/validation_final_set"
RAW_TEST_SET="${DATASET_ROOT}/test_final_set"
COMMON_TEST_SET="${DATASET_ROOT}/test_final_set_common_oc"

CHECKPOINT_PREFIX="${RUN_ROOT}/checkpoints"
TENSORBOARD_PREFIX="${RUN_ROOT}/tensorboard"
MODEL_PREFIX="${RUN_ROOT}/model"
MODEL_DIR="${MODEL_PREFIX}_cfg_ddg"
EMBEDDINGS_PATH="${RUN_ROOT}/test_final_set_common_oc_full_embeddings_cfg_ddg.pth"
MARKDOWN_PATH="${RUN_ROOT}/test_final_set_common_oc_full_results.md"

MAX_STEPS="${MAX_STEPS:-300000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-256}"
REBUILD_COMMON_TEST="${REBUILD_COMMON_TEST:-1}"

set +u
source "/home/damaoooo/miniconda3/etc/profile.d/conda.sh"
conda activate ReLL
set -u

cd "${REPO_ROOT}"

for required_path in "${TRAIN_SET}" "${VALIDATION_SET}" "${RAW_TEST_SET}" "${REFERENCE_TEST_SET}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}"

echo "[OC2 fair] repo_root=${REPO_ROOT}"
echo "[OC2 fair] dataset_root=${DATASET_ROOT}"
echo "[OC2 fair] reference_test_set=${REFERENCE_TEST_SET}"
echo "[OC2 fair] run_root=${RUN_ROOT}"
echo "[OC2 fair] checkpoint_dir=${CHECKPOINT_PREFIX}_cfg_ddg"
echo "[OC2 fair] model_dir=${MODEL_DIR}"
echo "[OC2 fair] max_steps=${MAX_STEPS}"
echo "[OC2 fair] embeddings_path=${EMBEDDINGS_PATH}"
echo "[OC2 fair] markdown_path=${MARKDOWN_PATH}"

if [[ "${REBUILD_COMMON_TEST}" == "1" || ! -d "${COMMON_TEST_SET}/train_dataset_pool" ]]; then
  echo "[OC2 fair] rebuilding common testset: ${COMMON_TEST_SET}"
  python Scripts/ray_opt_ablation/filter_final_set_by_reference.py \
    "${RAW_TEST_SET}" \
    "${REFERENCE_TEST_SET}" \
    --output "${COMMON_TEST_SET}" \
    --reference-kind final-set \
    --match-mode exact \
    --workers 0 \
    --batch-size 50000 \
    --overwrite
else
  echo "[OC2 fair] reusing existing common testset: ${COMMON_TEST_SET}"
fi

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

python evaluation.py \
  "${COMMON_TEST_SET}/train_positive_map.pkl" \
  --dataset-path "${COMMON_TEST_SET}/train_dataset_pool" \
  --model-path "${MODEL_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --gpu-batch-size "${GPU_BATCH_SIZE}" \
  --eval-samples 0 \
  --pool-samples 0 \
  --embeddings-path "${EMBEDDINGS_PATH}" \
  --markdown-output "${MARKDOWN_PATH}" \
  --cfg \
  --ddg \
  --bf16

echo "[OC2 fair] done"
echo "[OC2 fair] markdown_path=${MARKDOWN_PATH}"
