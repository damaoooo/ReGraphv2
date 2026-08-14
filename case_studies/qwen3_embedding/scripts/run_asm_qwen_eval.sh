#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_FINAL_SET="${DATASET_FINAL_SET:-/path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text}"
DATASET_POOL="${DATASET_POOL:-${DATASET_FINAL_SET}/train_dataset_pool}"
POSITIVE_MAP="${POSITIVE_MAP:-${DATASET_FINAL_SET}/train_positive_map.pkl}"
TEI_ENDPOINT="${TEI_ENDPOINT:-http://127.0.0.1:18081}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3-Embedding-0.6B}"
INSTRUCTION="${INSTRUCTION:-Represent this assembly function for searching for similar functions:}"

RUN_ID="${RUN_ID:-qwen3_0p6b_asm_ft}"
MODEL_ID="${MODEL_ID:-Qwen3-Embedding-0.6B ASM fine-tuned}"
SUMMARY_NAME="${SUMMARY_NAME:-Qwen3-0.6B-ft ASM}"
REPORT_DIR="${REPORT_DIR:-${RELL_ROOT}/experiments/eval_reports}"
CACHE_PATH="${CACHE_PATH:-${RELL_ROOT}/experiments/${RUN_ID}.bin.npy}"
LOG_PATH="${LOG_PATH:-${REPORT_DIR}/logs/${RUN_ID}.log}"
MD_PATH="${MD_PATH:-${REPORT_DIR}/${RUN_ID}.md}"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-512}"
KS="${KS:-1,5,10,15,20,25,30,35,40,45,50}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${CACHE_PATH}")"

python "${SCRIPT_DIR}/generate_tei_embeddings_resume.py" \
  "${DATASET_POOL}" \
  --output "${CACHE_PATH}" \
  --tei-endpoint "${TEI_ENDPOINT}" \
  --tokenizer-name "${TOKENIZER_NAME}" \
  --instruction "${INSTRUCTION}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}"

EVAL_COMMAND=(
  python "${RELL_ROOT}/evaluate.py"
  "${DATASET_POOL}"
  "${POSITIVE_MAP}"
  --tei-endpoint "${TEI_ENDPOINT}"
  --ks "${KS}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --gpu-batch-size "${GPU_BATCH_SIZE}"
  --eval-samples 0
  --embeddings-path "${CACHE_PATH}"
  --seed "${SEED}"
  --gpu
)

"${EVAL_COMMAND[@]}" >"${LOG_PATH}" 2>&1

python "${SCRIPT_DIR}/render_rell_eval_markdown.py" \
  --title "${MODEL_ID} on ASM common OC CSV hashdedup test_final_set" \
  --summary-name "${SUMMARY_NAME}" \
  --log "${LOG_PATH}" \
  --output "${MD_PATH}" \
  --endpoint "${TEI_ENDPOINT}" \
  --model-id "${MODEL_ID}" \
  --dataset "${DATASET_POOL}" \
  --positive-map "${POSITIVE_MAP}" \
  --cache "${CACHE_PATH}" \
  --command "${EVAL_COMMAND[*]}"

echo "Markdown report: ${MD_PATH}"
