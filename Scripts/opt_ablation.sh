
cd "/path/to/rell"
PYTHON_PATH="$(python -c 'import sys; print(sys.executable)')"
DATASET_PATH="/path/to/rell/IR/Dataset-1"
if [ -z "$1" ]; then
    echo "Usage: $0 <OPT_LEVEL>"
    exit 1
fi
OPT_LEVEL="$1"
OUTPUT_PATH="/path/to/rell/IR/Dataset-1-${OPT_LEVEL}"

${PYTHON_PATH} Scripts/pipeline.py pipeline --input-path ${DATASET_PATH} --output ${DATASET_PATH} --start-from 2 --opt-level ${OPT_LEVEL} --resume

${PYTHON_PATH} -m GraphBuilder.graph_generator "${DATASET_PATH}/train"
${PYTHON_PATH} -m GraphBuilder.graph_generator "${DATASET_PATH}/validation"
${PYTHON_PATH} -m GraphBuilder.graph_generator "${DATASET_PATH}/test"

${PYTHON_PATH} -m DataProcess.cli "${DATASET_PATH}/train" "${DATASET_PATH}/train_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf
${PYTHON_PATH} -m DataProcess.cli "${DATASET_PATH}/validation" "${DATASET_PATH}/validation_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf
${PYTHON_PATH} -m DataProcess.cli "${DATASET_PATH}/test" "${DATASET_PATH}/test_raw_dataset" --parallel --num-processes 32 --no-cache --use-hf

${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${DATASET_PATH}/train_raw_dataset" --output-path "${DATASET_PATH}/train_wash_dataset"
${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${DATASET_PATH}/validation_raw_dataset" --output-path "${DATASET_PATH}/validation_wash_dataset"
${PYTHON_PATH} -m DataProcess.dataset_wash --dataset-path "${DATASET_PATH}/test_raw_dataset" --output-path "${DATASET_PATH}/test_wash_dataset"

${PYTHON_PATH} -m Pretrain.split_train_validation "${DATASET_PATH}/train_wash_dataset" --base-path "${DATASET_PATH}/train" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/train_final_set"
${PYTHON_PATH} -m Pretrain.split_train_validation "${DATASET_PATH}/validation_wash_dataset" --base-path "${DATASET_PATH}/validation" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/validation_final_set"
${PYTHON_PATH} -m Pretrain.split_train_validation "${DATASET_PATH}/test_wash_dataset" --base-path "${DATASET_PATH}/test" --train-ratio 1.0 --output-dir "${OUTPUT_PATH}/test_final_set"
