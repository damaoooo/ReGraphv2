#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/path/to/rell"
DATASET_ROOT="${REPO_ROOT}/IR/Dataset-1-Oc2-fused"
RUN_ROOT="${REPO_ROOT}/runs/dataset1_oc2_fused"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TRAIN_SET="${DATASET_ROOT}/train_final_set"
VALIDATION_SET="${DATASET_ROOT}/validation_final_set"
TEST_SET="${DATASET_ROOT}/test_final_set_common_oc"
if [[ ! -d "${TEST_SET}" ]]; then
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

set +u
source "/path/to/miniconda3/etc/profile.d/conda.sh"
conda activate ReLL
set -u

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"

echo "[OC2] dataset_root=${DATASET_ROOT}"
echo "[OC2] run_root=${RUN_ROOT}"
echo "[OC2] checkpoint_dir=${CHECKPOINT_PREFIX}_cfg_ddg"
echo "[OC2] model_dir=${MODEL_DIR}"
echo "[OC2] test_set=${TEST_SET}"
echo "[OC2] embeddings_path=${EMBEDDINGS_PATH}"
echo "[OC2] markdown_path=${MARKDOWN_PATH}"

python -m Pretrain.run_pretrain train \
  --dataset-dir "${TRAIN_SET}" \
  --validation-dataset-dir "${VALIDATION_SET}" \
  --cfg \
  --ddg \
  --resume \
  --set "max_seq_length=2048" \
  --set "save_steps=10000" \
  --set "output_dir=${CHECKPOINT_PREFIX}" \
  --set "logging_dir=${TENSORBOARD_PREFIX}" \
  --set "final_model_dir=${MODEL_PREFIX}" \
  --set "report_to=tensorboard"

python evaluation.py "${TEST_MAP}" \
  --dataset-path "${TEST_POOL}" \
  --model-path "${MODEL_DIR}" \
  --max-length 2048 \
  --batch-size 16 \
  --gpu-batch-size 256 \
  --eval-samples 0 \
  --pool-samples 0 \
  --embeddings-path "${EMBEDDINGS_PATH}" \
  --markdown-output "${MARKDOWN_PATH}" \
  --cfg \
  --ddg \
  --bf16

echo "[OC2] done"
echo "[OC2] markdown_path=${MARKDOWN_PATH}"
