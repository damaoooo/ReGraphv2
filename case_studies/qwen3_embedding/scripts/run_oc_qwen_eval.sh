#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGRAPH_ROOT="${REGRAPH_ROOT:-$(cd "${RELL_ROOT}/../regraphv2" && pwd)}"

OC_FINAL_SET="${OC_FINAL_SET:-${REGRAPH_ROOT}/IR/Dataset-1-Oc-fused/test_final_set}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REGRAPH_ROOT}/Tokenizer/output_tokenizer/llvm_ir_bpe.json}"
OUTPUT_DATASET="${OUTPUT_DATASET:-${RELL_ROOT}/data/oc_eval_dataset}"

TEI_ENDPOINT="${TEI_ENDPOINT:-http://127.0.0.1:8080}"
MODEL_ID="${MODEL_ID:-fine-tuned Qwen3-Embedding-0.6B}"
SUMMARY_NAME="${SUMMARY_NAME:-Qwen3-0.6B-ft OC}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3-Embedding-0.6B}"

BATCH_SIZE="${BATCH_SIZE:-128}"
TEI_WORKERS="${TEI_WORKERS:-4}"
TEI_TIMEOUT="${TEI_TIMEOUT:-180}"
TEI_MAX_RETRIES="${TEI_MAX_RETRIES:-12}"
TEI_RETRY_BASE_DELAY="${TEI_RETRY_BASE_DELAY:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-512}"
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"
KS="${KS:-1,5,10,15,20,25,30,35,40,45,50}"
SEED="${SEED:-42}"
USE_GPU_FLAG="${USE_GPU_FLAG:---gpu}"

CONDA_ENV="${CONDA_ENV:-}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

RUN_ID="${RUN_ID:-qwen3_0p6b_ft_oc}"
REPORT_DIR="${REPORT_DIR:-${RELL_ROOT}/experiments/eval_reports}"
LOG_DIR="${LOG_DIR:-${REPORT_DIR}/logs}"
CACHE_PATH="${CACHE_PATH:-${RELL_ROOT}/experiments/${RUN_ID}.bin.npy}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_ID}.log}"
MD_PATH="${MD_PATH:-${REPORT_DIR}/${RUN_ID}.md}"

cd "${RELL_ROOT}"

if [[ -n "${CONDA_ENV}" ]]; then
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Cannot find conda hook: ${CONDA_SH}" >&2
    exit 1
  fi
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
fi

set -u

mkdir -p "${LOG_DIR}" "$(dirname "${CACHE_PATH}")"

if [[ ! -d "${OUTPUT_DATASET}/train_dataset_pool" || ! -f "${OUTPUT_DATASET}/train_positive_map.pkl" || "${FORCE_REBUILD_DATASET:-0}" == "1" ]]; then
  python "${SCRIPT_DIR}/build_regraph_oc_eval_dataset.py" \
    --source-final-set "${OC_FINAL_SET}" \
    --regraph-root "${REGRAPH_ROOT}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --output "${OUTPUT_DATASET}" \
    ${FORCE_REBUILD_DATASET:+--force}
fi

python - <<PY
import requests
endpoint = "${TEI_ENDPOINT}".rstrip("/")
try:
    response = requests.post(
        endpoint + "/embed",
        json={"inputs": ["define i32 @func0() { ret i32 0 }"]},
        timeout=10,
    )
    response.raise_for_status()
except Exception as exc:
    raise SystemExit(f"TEI endpoint is not ready: {endpoint}/embed ({exc})")
print(f"TEI endpoint is ready: {endpoint}/embed")
PY

if [[ ! -f "${CACHE_PATH}" || "${FORCE_REBUILD_EMBEDDINGS:-0}" == "1" ]]; then
  python "${SCRIPT_DIR}/generate_tei_embeddings_resume.py" \
    "${OUTPUT_DATASET}/train_dataset_pool" \
    --output "${CACHE_PATH}" \
    --tei-endpoint "${TEI_ENDPOINT}" \
    --tokenizer-name "${TOKENIZER_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --timeout "${TEI_TIMEOUT}" \
    --max-retries "${TEI_MAX_RETRIES}" \
    --retry-base-delay "${TEI_RETRY_BASE_DELAY}"
fi

export COLUMNS="${COLUMNS:-240}"
EVAL_COMMAND=(
  python evaluate.py
  "${OUTPUT_DATASET}/train_dataset_pool"
  "${OUTPUT_DATASET}/train_positive_map.pkl"
  --tei-endpoint "${TEI_ENDPOINT}"
  --ks "${KS}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --tei-workers "${TEI_WORKERS}"
  --tei-timeout "${TEI_TIMEOUT}"
  --tei-max-retries "${TEI_MAX_RETRIES}"
  --tei-retry-base-delay "${TEI_RETRY_BASE_DELAY}"
  --gpu-batch-size "${GPU_BATCH_SIZE}"
  --eval-samples "${EVAL_SAMPLES}"
  --embeddings-path "${CACHE_PATH}"
  --seed "${SEED}"
  "${USE_GPU_FLAG}"
)

"${EVAL_COMMAND[@]}" > "${LOG_PATH}" 2>&1

python "${SCRIPT_DIR}/render_rell_eval_markdown.py" \
  --title "${MODEL_ID} on OC test_final_set" \
  --summary-name "${SUMMARY_NAME}" \
  --log "${LOG_PATH}" \
  --output "${MD_PATH}" \
  --endpoint "${TEI_ENDPOINT}" \
  --model-id "${MODEL_ID}" \
  --dataset "${OUTPUT_DATASET}/train_dataset_pool" \
  --positive-map "${OUTPUT_DATASET}/train_positive_map.pkl" \
  --cache "${CACHE_PATH}" \
  --command "${EVAL_COMMAND[*]}"

echo "Markdown report: ${MD_PATH}"
