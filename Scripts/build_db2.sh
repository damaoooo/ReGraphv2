#!/bin/bash

PYTHON_PATH="/path/to/miniconda3/envs/ReLL/bin/python"
DATASET_PATH="/path/to/rell/IR/Dataset-2-IR"
OUTPUT_PATH="/path/to/rell/IR/Dataset-2"

cd "/path/to/rell"

${PYTHON_PATH} -m GraphBuilder.graph_generator "${DATASET_PATH}" # 这个命令没有问题，第二个就有问题了，需要debug

${PYTHON_PATH} -m DataProcess.cli "${DATASET_PATH}" "${OUTPUT_PATH}/db2_raw_dataset" --num-processes 32 --parallel  --no-cache --use-hf

${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${OUTPUT_PATH}/db2_raw_dataset" --output-path "${OUTPUT_PATH}/db2_wash_dataset" --tokenizer-path "/path/to/datasets/IR/Dataset-1-all/train_corpus_tokenizer/llvm_ir_bpe.json"

${PYTHON_PATH} -m Pretrain.split_train_validation "${OUTPUT_PATH}/db2_wash_dataset" --base-path "${DATASET_PATH}" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/db2_final_set"
