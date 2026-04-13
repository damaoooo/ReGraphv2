#!/bin/bash

PYTHON_PATH="/home/damaoooo/miniconda3/envs/ReLL/bin/python"
DATASET_PATH="/home/damaoooo/Datasets/IR/Dataset-1-asm/Dataset-1"
TOKENIZER_PATH="/home/damaoooo/Datasets/IR/Dataset-1-asm/Dataset-1/train_corpus_tokenizer/asm_bpe.json"

cd "/home/damaoooo/Downloads/regraphv2"

${PYTHON_PATH} -m DataProcess.cli "${DATASET_PATH}/train" "${DATASET_PATH}/train_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf
${PYTHON_PATH} -m DataProcess.cli "${DATASET_PATH}/validation" "${DATASET_PATH}/validation_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf
${PYTHON_PATH} -m DataProcess.cli "${DATASET_PATH}/test" "${DATASET_PATH}/test_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf


${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${DATASET_PATH}/train_raw_dataset" --output-path "${DATASET_PATH}/train_wash_dataset" --tokenizer-path "${TOKENIZER_PATH}"
${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${DATASET_PATH}/validation_raw_dataset" --output-path "${DATASET_PATH}/validation_wash_dataset" --tokenizer-path "${TOKENIZER_PATH}"
${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${DATASET_PATH}/test_raw_dataset" --output-path "${DATASET_PATH}/test_wash_dataset" --tokenizer-path "${TOKENIZER_PATH}"

${PYTHON_PATH} -m Pretrain.split_train_validation "${DATASET_PATH}/train_wash_dataset" --base-path "${DATASET_PATH}/train" --train-ratio 1.0 --output-dir "${DATASET_PATH}/train_final_set"
${PYTHON_PATH} -m Pretrain.split_train_validation "${DATASET_PATH}/validation_wash_dataset" --base-path "${DATASET_PATH}/validation" --train-ratio 1.0 --output-dir "${DATASET_PATH}/validation_final_set"
${PYTHON_PATH} -m Pretrain.split_train_validation "${DATASET_PATH}/test_wash_dataset" --base-path "${DATASET_PATH}/test" --train-ratio 1.0 --output-dir "${DATASET_PATH}/test_final_set"