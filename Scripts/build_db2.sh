#!/bin/bash

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=(uv run --project "${REPO_ROOT}" --locked --extra pipeline python)
DATASET_PATH="/home/damaoooo/Downloads/regraphv2/IR/Dataset-2-IR"
OUTPUT_PATH="/home/damaoooo/Downloads/regraphv2/IR/Dataset-2"

cd "${REPO_ROOT}"

"${PYTHON[@]}" -m GraphBuilder.graph_generator "${DATASET_PATH}" # 这个命令没有问题，第二个就有问题了，需要debug

"${PYTHON[@]}" -m DataProcess.cli "${DATASET_PATH}" "${OUTPUT_PATH}/db2_raw_dataset" --num-processes 32

"${PYTHON[@]}" -m DataProcess.dataset_wash --dataset-path "${OUTPUT_PATH}/db2_raw_dataset" --output-path "${OUTPUT_PATH}/db2_wash_dataset" --tokenizer-path "Tokenizer/output_tokenizer/llvm_ir_bpe.json"

"${PYTHON[@]}" -m Pretrain.split_train_validation "${OUTPUT_PATH}/db2_wash_dataset" --base-path "${DATASET_PATH}" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/db2_final_set"
