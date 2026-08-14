#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-ml}"

MERGED_DIR="${MERGED_DIR:-${RELL_ROOT}/experiments/qwen-asm-finetune-hf-merged}"
TEI_IMAGE="${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:120-1.9}"
TEI_DATA_DIR="${TEI_DATA_DIR:-${RELL_ROOT}/tei_data}"
TEI_PORT="${TEI_PORT:-18081}"
TEI_ENDPOINT="http://127.0.0.1:${TEI_PORT}"
CONTAINER_NAME="${CONTAINER_NAME:-rell-qwen-asm-eval}"
POLL_SECONDS="${POLL_SECONDS:-300}"
INSTRUCTION="${INSTRUCTION:-Represent this assembly function for searching for similar functions:}"

wait_for_merged_model() {
  while [[ ! -f "${MERGED_DIR}/config.json" ]]; do
    echo "Waiting for merged model: ${MERGED_DIR}"
    sleep "${POLL_SECONDS}"
  done
}

stop_tei() {
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi
}

wait_for_tei() {
  conda run -n "${CONDA_ENV}" --no-capture-output python -c "
import time
import requests

endpoint = '${TEI_ENDPOINT}/embed'
last_error = None
for _ in range(120):
    try:
        response = requests.post(
            endpoint,
            json={'inputs': ['MOV R0, R1\\nBX LR']},
            timeout=10,
        )
        response.raise_for_status()
        print(f'TEI ready: {endpoint}')
        break
    except Exception as exc:
        last_error = exc
        time.sleep(5)
else:
    raise SystemExit(f'TEI did not become ready: {last_error}')
"
}

start_merged_tei() {
  stop_tei
  docker run --gpus all --rm -d \
    --name "${CONTAINER_NAME}" \
    -p "127.0.0.1:${TEI_PORT}:80" \
    -v "${MERGED_DIR}:/model:ro" \
    -v "${TEI_DATA_DIR}:/data" \
    "${TEI_IMAGE}" \
    --model-id /model \
    --pooling last-token \
    --max-client-batch-size 128 \
    --max-batch-tokens 131072 \
    --dtype float16 >/dev/null
  wait_for_tei
}

start_official_tei() {
  stop_tei
  docker run --gpus all --rm -d \
    --name "${CONTAINER_NAME}" \
    -p "127.0.0.1:${TEI_PORT}:80" \
    -v "${TEI_DATA_DIR}:/data" \
    "${TEI_IMAGE}" \
    --model-id Qwen/Qwen3-Embedding-0.6B \
    --pooling last-token \
    --max-client-batch-size 128 \
    --max-batch-tokens 131072 \
    --dtype float16 >/dev/null
  wait_for_tei
}

run_finetuned_eval() {
  conda run -n "${CONDA_ENV}" --no-capture-output \
    env \
      TEI_ENDPOINT="${TEI_ENDPOINT}" \
      TOKENIZER_NAME="${MERGED_DIR}" \
      INSTRUCTION="${INSTRUCTION}" \
      RUN_ID="qwen3_0p6b_asm_ft_original_train_asm_prompt" \
      MODEL_ID="Qwen3-Embedding-0.6B ASM QLoRA (original ASM train)" \
      SUMMARY_NAME="Qwen3-0.6B-ft ASM original train" \
      bash "${SCRIPT_DIR}/run_asm_qwen_eval.sh"
}

run_official_eval() {
  conda run -n "${CONDA_ENV}" --no-capture-output \
    env \
      TEI_ENDPOINT="${TEI_ENDPOINT}" \
      TOKENIZER_NAME="Qwen/Qwen3-Embedding-0.6B" \
      INSTRUCTION="${INSTRUCTION}" \
      RUN_ID="qwen3_0p6b_official_asm_prompt" \
      MODEL_ID="Qwen/Qwen3-Embedding-0.6B official with ASM prompt" \
      SUMMARY_NAME="Qwen3-0.6B-official ASM prompt" \
      bash "${SCRIPT_DIR}/run_asm_qwen_eval.sh"
}

trap stop_tei EXIT
wait_for_merged_model
start_merged_tei
run_finetuned_eval
stop_tei
start_official_tei
run_official_eval
stop_tei

echo "Fine-tuned report:"
echo "${RELL_ROOT}/experiments/eval_reports/qwen3_0p6b_asm_ft_original_train_asm_prompt.md"
echo "Official report:"
echo "${RELL_ROOT}/experiments/eval_reports/qwen3_0p6b_official_asm_prompt.md"
