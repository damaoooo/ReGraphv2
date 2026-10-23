#!/bin/bash
# Dataset-2 preprocessing after Task 3: graphs -> raw HF dataset -> wash -> final set.
#
# usage: Scripts/build_db2.sh <function IR dir> <output dir> [workers]
#   <function IR dir>  directory holding the Task 3 output (<binary>_functions/ with function_map.csv)
#   <output dir>       receives db2_raw_dataset/, db2_wash_dataset/ and db2_final_set/
#   [workers]          worker processes for every stage; without it each stage uses the CPUs allocated
#                      to it (affinity, capped by SLURM_CPUS_PER_TASK)
#
# Environment:
#   REGRAPH_PYTHON_CMD  interpreter command, split on whitespace, so its words cannot contain spaces
#                       (default: uv run --project <repo> --locked --extra pipeline python)
#   TOKENIZER_PATH      tokenizer JSON (default: <repo>/Tokenizer/output_tokenizer/llvm_ir_bpe.json)
# Relative paths are taken from the directory the script is started in.
#
# Prerequisites: an environment with the `pipeline` extra, LLVM 18 `opt` on PATH, and LLVMTools built
# (cmake -S LLVMTools -B LLVMTools/build -DCMAKE_BUILD_TYPE=Release && cmake --build LLVMTools/build -j 4).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    sed -n '2,17p' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
WORKERS="${3:-}"
if [ -n "${WORKERS}" ] && ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "workers must be a positive integer, got '${WORKERS}'" >&2
    exit 2
fi
# Resolve every caller path before changing into the repository
DATASET_PATH="$(cd "$1" && pwd)"
mkdir -p "$2"
OUTPUT_PATH="$(cd "$2" && pwd)"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/Tokenizer/output_tokenizer/llvm_ir_bpe.json}"
case "${TOKENIZER_PATH}" in
    /*) ;;
    *) TOKENIZER_PATH="${PWD}/${TOKENIZER_PATH}" ;;
esac
if [ -n "${REGRAPH_PYTHON_CMD:-}" ]; then
    read -r -a PYTHON <<< "${REGRAPH_PYTHON_CMD}"
else
    PYTHON=(uv run --project "${REPO_ROOT}" --locked --extra pipeline python)
fi

# Pass a worker count only when one was given, so each stage otherwise applies its own CPU budget
JOBS=() PROCESSES=() NUM_PROC=()
if [ -n "${WORKERS}" ]; then
    JOBS=(--jobs "${WORKERS}") PROCESSES=(--num-processes "${WORKERS}") NUM_PROC=(--num-proc "${WORKERS}")
fi
cd "${REPO_ROOT}"

"${PYTHON[@]}" -m GraphBuilder.graph_generator "${DATASET_PATH}" "${JOBS[@]}"

"${PYTHON[@]}" -m DataProcess "${DATASET_PATH}" "${OUTPUT_PATH}/db2_raw_dataset" \
    --tokenizer-path "${TOKENIZER_PATH}" "${PROCESSES[@]}"

"${PYTHON[@]}" -m DataProcess.wash --dataset-path "${OUTPUT_PATH}/db2_raw_dataset" \
    --output-path "${OUTPUT_PATH}/db2_wash_dataset" --tokenizer-path "${TOKENIZER_PATH}" "${NUM_PROC[@]}"

# function_map.csv is found next to each function file, so no --base-path is needed
"${PYTHON[@]}" -m Pretrain.split_train_validation "${OUTPUT_PATH}/db2_wash_dataset" \
    --naming dataset2 --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/db2_final_set"
