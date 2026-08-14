#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGRAPH_ROOT="${REGRAPH_ROOT:-/path/to/rell}"
CONDA_ENV="${CONDA_ENV:-ml}"

ASM_DATASET_ROOT="${ASM_DATASET_ROOT:-${REGRAPH_ROOT}/IR/dataset-1-asm}"
ASM_QWEN_ROOT="${ASM_QWEN_ROOT:-${REGRAPH_ROOT}/IR/Dataset-1-ASM-Qwen-text}"

TRAIN_FINAL_SET="${ASM_QWEN_ROOT}/train_final_set"
VALIDATION_FINAL_SET="${ASM_QWEN_ROOT}/validation_final_set"
OUTPUT_DIR="${OUTPUT_DIR:-${RELL_ROOT}/experiments/qwen-asm-finetune-hf}"
MERGED_DIR="${MERGED_DIR:-${RELL_ROOT}/experiments/qwen-asm-finetune-hf-merged}"
CONFIG_PATH="${CONFIG_PATH:-${RELL_ROOT}/configs/train_asm_config.yaml}"

NUM_PROC="${NUM_PROC:-32}"
INSTRUCTION="${INSTRUCTION:-Represent this assembly function for searching for similar functions:}"

cd "${RELL_ROOT}"

conda run -n "${CONDA_ENV}" --no-capture-output \
  python -u "${SCRIPT_DIR}/build_qwen_asm_text_dataset.py" \
  --input-final-set "${ASM_DATASET_ROOT}/train_final_set" \
  --output-final-set "${TRAIN_FINAL_SET}" \
  --num-proc "${NUM_PROC}"

conda run -n "${CONDA_ENV}" --no-capture-output \
  python -u "${SCRIPT_DIR}/build_qwen_asm_text_dataset.py" \
  --input-final-set "${ASM_DATASET_ROOT}/validation_final_set" \
  --output-final-set "${VALIDATION_FINAL_SET}" \
  --num-proc "${NUM_PROC}"

TRAIN_ARGS=()
if find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
  TRAIN_ARGS+=(--resume)
fi

CONDA_ENV="${CONDA_ENV}" \
OC_ROOT="${ASM_QWEN_ROOT}" \
CONFIG_PATH="${CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_ID="qwen_asm_finetune" \
INSTRUCTION="${INSTRUCTION}" \
bash "${SCRIPT_DIR}/run_oc_qwen_train.sh" --dry-run "${TRAIN_ARGS[@]}"

TRAIN_COMMAND=(
  conda run -n "${CONDA_ENV}" --no-capture-output
  python "${RELL_ROOT}/train.py" "${CONFIG_PATH}"
)
if (( ${#TRAIN_ARGS[@]} > 0 )); then
  TRAIN_COMMAND+=(--resume)
fi
"${TRAIN_COMMAND[@]}"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  conda run -n "${CONDA_ENV}" python "${SCRIPT_DIR}/04_merge_and_export.py" \
    "${OUTPUT_DIR}" \
    "${MERGED_DIR}"
fi

echo "Training and merge complete."
echo "Adapter: ${OUTPUT_DIR}"
echo "Merged model: ${MERGED_DIR}"
echo "Run scripts/run_asm_qwen_eval.sh after serving the merged model with TEI."
