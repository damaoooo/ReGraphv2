
cd "/home/damaoooo/Downloads/regraphv2"
PYTHON_PATH="$(python -c 'import sys; print(sys.executable)')"
DATASET_PATH="/home/damaoooo/Downloads/regraphv2/IR/Dataset-1"
if [ -z "$1" ]; then
    echo "Usage: $0 <OPT_LEVEL>"
    exit 1
fi
OPT_LEVEL="$1"
OUTPUT_PATH="/home/damaoooo/Downloads/regraphv2/IR/Dataset-1-${OPT_LEVEL}"

${PYTHON_PATH} Scripts/pipeline.py pipeline --input-path ${DATASET_PATH} --output ${OUTPUT_PATH} --start-from 2 --opt-level ${OPT_LEVEL} --resume

${PYTHON_PATH} -m GraphBuilder.graph_generator "${OUTPUT_PATH}/train"
${PYTHON_PATH} -m GraphBuilder.graph_generator "${OUTPUT_PATH}/validation"
${PYTHON_PATH} -m GraphBuilder.graph_generator "${OUTPUT_PATH}/test"

${PYTHON_PATH} -m DataProcess.cli "${OUTPUT_PATH}/train" "${OUTPUT_PATH}/train_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf
${PYTHON_PATH} -m DataProcess.cli "${OUTPUT_PATH}/validation" "${OUTPUT_PATH}/validation_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf
${PYTHON_PATH} -m DataProcess.cli "${OUTPUT_PATH}/test" "${OUTPUT_PATH}/test_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf

${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${OUTPUT_PATH}/train_raw_dataset" --output-path "${OUTPUT_PATH}/train_wash_dataset" --tokenizer-path "/home/damaoooo/Datasets/IR/Dataset-1-all/train_corpus_tokenizer/llvm_ir_bpe.json"
${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${OUTPUT_PATH}/validation_raw_dataset" --output-path "${OUTPUT_PATH}/validation_wash_dataset" --tokenizer-path "/home/damaoooo/Datasets/IR/Dataset-1-all/train_corpus_tokenizer/llvm_ir_bpe.json"
${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${OUTPUT_PATH}/test_raw_dataset" --output-path "${OUTPUT_PATH}/test_wash_dataset" --tokenizer-path "/home/damaoooo/Datasets/IR/Dataset-1-all/train_corpus_tokenizer/llvm_ir_bpe.json"

${PYTHON_PATH} -m Pretrain.split_train_validation "${OUTPUT_PATH}/train_wash_dataset" --base-path "${OUTPUT_PATH}/train" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/train_final_set"
${PYTHON_PATH} -m Pretrain.split_train_validation "${OUTPUT_PATH}/validation_wash_dataset" --base-path "${OUTPUT_PATH}/validation" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/validation_final_set"
${PYTHON_PATH} -m Pretrain.split_train_validation "${OUTPUT_PATH}/test_wash_dataset" --base-path "${OUTPUT_PATH}/test" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/test_final_set"
