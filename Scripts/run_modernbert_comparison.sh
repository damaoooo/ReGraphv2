#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-ReLL}"
MODE="${1:---cached-gnn}"
case "${MODE}" in
  --cached-gnn|--matched-100k) ;;
  *) echo "Usage: $0 [--cached-gnn|--matched-100k]" >&2; exit 2 ;;
esac
if [[ "${MODE}" == "--matched-100k" ]]; then
  RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/modernbert_gnn_100k_len2048}"
  BASELINE_MEMBER="db1_gnn_100K_test.pkl.npy"
  STEPS=100000
  LENGTH=2048
else
  RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/modernbert_both_30000}"
  BASELINE_MEMBER="db1_gnn_both_test.pkl.npy"
  STEPS=30000
  LENGTH=4096
fi
TOKENIZER="${ROOT}/IR/dataset-1/train_corpus_tokenizer/llvm_ir_bpe.json"
TRAIN="${ROOT}/IR/dataset-1/train_final_set"
TEST="${ROOT}/IR/dataset-1/test_final_set"
MODEL="${RUN_ROOT}/model_cfg_ddg"
PYTHON=(conda run -n "${CONDA_ENV}" --no-capture-output python -u)
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mnt/nas/ReLL-archives/2026-10-21-roformer-retirement}"
BASELINE_ARCHIVE="${BASELINE_ARCHIVE:-${ARCHIVE_ROOT}/roformer-gnn.tar}"
if [[ -n "${BASELINE_EMBEDDINGS:-}" ]]; then
  [[ -f "${BASELINE_EMBEDDINGS}" ]] || { echo "Missing baseline: ${BASELINE_EMBEDDINGS}" >&2; exit 1; }
else
  [[ -f "${BASELINE_ARCHIVE}" ]] || { echo "Set BASELINE_EMBEDDINGS or BASELINE_ARCHIVE to an existing file." >&2; exit 1; }
fi
mkdir -p "${RUN_ROOT}"
cd "${ROOT}"
trap 'echo "Pipeline failed at line ${LINENO}; rerun this script to resume." >&2' ERR

if [[ ! -f "${RUN_ROOT}/training.complete" ]]; then
  "${PYTHON[@]}" -m Pretrain.run_pretrain train \
    --dataset-dir "${TRAIN}" --cfg --ddg --no-validation --resume \
    --set "tokenizer_path=${TOKENIZER}" \
    --set attention_backend=flash_attention_2 \
    --set "max_steps=${STEPS}" --set "max_seq_length=${LENGTH}" \
    --set per_device_train_batch_size=4 --set gradient_checkpointing=false \
    --set learning_rate=0.00005 --set warmup_steps=1000 \
    --set optim=paged_adamw_8bit --set weight_decay=0.01 --set bf16=true \
    --set graph_layers=1 --set graph_attention_heads=4 --set graph_dropout=0.1 \
    --set moco_buffer_size=65536 --set moco_momentum=0.999 --set moco_temperature=0.07 \
    --set mlm_probability=0.15 \
    --set save_steps=1000 --set save_total_limit=3 --set logging_steps=100 \
    --set dataloader_num_workers=16 --set skip_bad_batch=false \
    --set "output_dir=${RUN_ROOT}/checkpoints" \
    --set "logging_dir=${RUN_ROOT}/tensorboard" \
    --set "final_model_dir=${RUN_ROOT}/model"
  touch "${RUN_ROOT}/training.complete"
fi

"${PYTHON[@]}" evaluation.py "${TEST}/train_positive_map.pkl" --export-only \
  --model-path "${MODEL}" --dataset-path "${TEST}/train_dataset_pool" \
  --tokenizer-path "${TOKENIZER}" --max-length "${LENGTH}" --cfg --ddg --bf16 \
  --batch-size 4 --embeddings-path "${RUN_ROOT}/modernbert_test.npy"

if [[ "${MODE}" == "--matched-100k" ]]; then
  LABEL="RoFormer-GNN-100K"
  NOTE="Both runs use Dataset-1 IR, 100000 training steps, max length 2048, batch size 4, and CFG/DDG GAT branches (1 layer, 4 heads). Optimizer, learning rate, warmup and MoCo settings match the historical TensorBoard configuration. ModernBERT is trained from scratch, not resumed from the 30K/4096 model. Baseline: db1_model_gnn_100K_cfg_ddg/checkpoint-100000. Historical evaluation commands associate its cache with this test pool; the old cache lacks fingerprint metadata. Encoder-specific architecture and defaults, and library versions, differ. This is not bitwise identical replay of the historical training."
  REPORT="${RUN_ROOT}/comparison_100k_len2048.md"
else
  LABEL="RoFormer-GNN"
  NOTE="Historical commands associate the baseline with db1_model_cfg_ddg and the same test dataset path, max length 2048. ModernBERT uses max length 4096. The old cache has no fingerprint metadata, so exact historical dataset identity and row order are not independently verified. Training budgets are not verified to match."
  REPORT="${RUN_ROOT}/comparison.md"
fi
REPORT="${REPORT_OUTPUT:-${REPORT}}"

if [[ -z "${BASELINE_EMBEDDINGS:-}" ]]; then
  TEMP_BASELINE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rell-baseline.XXXXXXXX")"
  cleanup_baseline() {
    rm -f -- "${TEMP_BASELINE_DIR}/baseline.npy"
    rmdir -- "${TEMP_BASELINE_DIR}"
  }
  trap cleanup_baseline EXIT
  BASELINE_EMBEDDINGS="${TEMP_BASELINE_DIR}/baseline.npy"
  # One NAS file is opened; only the selected vector cache is staged locally.
  tar --occurrence=1 -xOf "${BASELINE_ARCHIVE}" "${BASELINE_MEMBER}" > "${BASELINE_EMBEDDINGS}"
  NOTE="${NOTE} Baseline restored temporarily from ${BASELINE_ARCHIVE}, member ${BASELINE_MEMBER}."
fi

"${PYTHON[@]}" evaluation.py "${TEST}/train_positive_map.pkl" \
  --paired-baseline "${BASELINE_EMBEDDINGS}" \
  --embeddings-path "${RUN_ROOT}/modernbert_test.npy" \
  --baseline-label "${LABEL}" \
  --comparison-note "${NOTE}" \
  --markdown-output "${REPORT}"
echo "Completed: ${REPORT}"
