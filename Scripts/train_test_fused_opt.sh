#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/damaoooo/Downloads/regraphv2"
PYTHON_CMD=${PYTHON_CMD:-$(which python)}

steps=300000
max_length=2048
eval_steps=1000
save_steps=1000
batch_size=16
gpu_batch_size=256
eval_samples=0
pool_samples=0
reuse_embeddings=0
resume=0
skip_train=0
output_root=""

print_help() {
	cat <<'EOF'
Usage:
  bash Scripts/train_test_fused_opt.sh OPT_LEVEL [OPTIONS]

OPT_LEVEL:
  O0, O1, O2, or just 0/1/2.

What it does:
  1. Train with CFG+DDG on IR/Dataset-1-OPT-fused/train_final_set.
  2. Use IR/Dataset-1-OPT-fused/validation_final_set only for eval_loss during training.
  3. Run retrieval evaluation on IR/Dataset-1-OPT-fused/test_final_set.
  4. Save the test result as Markdown.

Options:
  --steps N              Max training steps. Default: 300000
  --max-length N         Train/eval max sequence length. Default: 2048
  --eval-steps N         Validation loss interval. Default: 1000
  --save-steps N         Checkpoint save interval. Default: 1000
  --batch-size N         Evaluation embedding batch size. Default: 16
  --gpu-batch-size N     Evaluation similarity GPU batch size. Default: 256
  --eval-samples N       Test anchors to sample. 0 means all. Default: 0
  --pool-samples N       Retrieval pool samples. 0 means full pool. Default: 0
  --output-root DIR      Output directory. Default: ./runs/dataset1_<opt>_fused
  --resume               Resume training from the latest checkpoint.
  --skip-train           Only run test evaluation using the existing model.
  --reuse-embeddings     Reuse existing test embedding cache if present.
  -h, --help             Show this help.

Environment:
	PYTHON_CMD             Python runner. Default: output of `which python`

Example:
  bash Scripts/train_test_fused_opt.sh O0
  bash Scripts/train_test_fused_opt.sh O2 --steps 50000 --max-length 4096
EOF
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
	print_help
	exit 0
fi

if [[ $# -lt 1 ]]; then
	print_help
	exit 1
fi

opt_level="$1"
shift

case "${opt_level}" in
	0|O0|o0) opt_level="O0" ;;
	1|O1|o1) opt_level="O1" ;;
	2|O2|o2) opt_level="O2" ;;
	*)
		echo "Error: OPT_LEVEL must be one of O0/O1/O2 or 0/1/2."
		exit 1
		;;
esac

while [[ $# -gt 0 ]]; do
	case "$1" in
		--steps)
			steps="$2"
			shift 2
			;;
		--max-length)
			max_length="$2"
			shift 2
			;;
		--eval-steps)
			eval_steps="$2"
			shift 2
			;;
		--save-steps)
			save_steps="$2"
			shift 2
			;;
		--batch-size)
			batch_size="$2"
			shift 2
			;;
		--gpu-batch-size)
			gpu_batch_size="$2"
			shift 2
			;;
		--eval-samples)
			eval_samples="$2"
			shift 2
			;;
		--pool-samples)
			pool_samples="$2"
			shift 2
			;;
		--output-root)
			output_root="$2"
			shift 2
			;;
		--resume)
			resume=1
			shift
			;;
		--skip-train)
			skip_train=1
			shift
			;;
		--reuse-embeddings)
			reuse_embeddings=1
			shift
			;;
		-h|--help)
			print_help
			exit 0
			;;
		*)
			echo "Error: Unknown argument '$1'"
			echo
			print_help
			exit 1
			;;
	esac
done

for numeric_arg in steps max_length eval_steps save_steps batch_size gpu_batch_size eval_samples pool_samples; do
	value="${!numeric_arg}"
	if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
		echo "Error: --${numeric_arg//_/-} must be a non-negative integer."
		exit 1
	fi
done

cd "${REPO_ROOT}"

opt_lower="$(tr '[:upper:]' '[:lower:]' <<< "${opt_level}")"
dataset_root="${REPO_ROOT}/IR/Dataset-1-${opt_level}-fused"
train_dir="${dataset_root}/train_final_set"
validation_dir="${dataset_root}/validation_final_set"
test_dir="${dataset_root}/test_final_set"
test_pool="${test_dir}/train_dataset_pool"
test_map="${test_dir}/train_positive_map.pkl"

if [[ -z "${output_root}" ]]; then
	output_root="${REPO_ROOT}/runs/dataset1_${opt_lower}_fused"
fi

checkpoint_prefix="${output_root}/checkpoints"
tensorboard_prefix="${output_root}/tensorboard"
model_prefix="${output_root}/model"
model_dir="${model_prefix}_cfg_ddg"
embedding_cache="${output_root}/test_embeddings.npy"
markdown_result="${output_root}/test_results.md"
train_log="${output_root}/train.log"
test_log="${output_root}/test_evaluation.log"

for required_path in "${train_dir}" "${validation_dir}" "${test_pool}" "${test_map}"; do
	if [[ ! -e "${required_path}" ]]; then
		echo "Error: required path does not exist: ${required_path}"
		exit 1
	fi
done

mkdir -p "${output_root}"

echo "[INFO] optimization=${opt_level}"
echo "[INFO] dataset_root=${dataset_root}"
echo "[INFO] output_root=${output_root}"
echo "[INFO] model_dir=${model_dir}"
echo "[INFO] markdown_result=${markdown_result}"
echo "[INFO] cfg=true, ddg=true"
echo "[INFO] max_length=${max_length}, steps=${steps}, eval_steps=${eval_steps}"

read -r -a PYTHON <<< "${PYTHON_CMD}"

if [[ "${skip_train}" == "0" ]]; then
	train_cmd=(
		"${PYTHON[@]}" -m Pretrain.run_pretrain train
		--dataset-dir "${train_dir}"
		--validation-dataset-dir "${validation_dir}"
		--set "max_steps=${steps}"
		--set "max_seq_length=${max_length}"
		--set "eval_steps=${eval_steps}"
		--set "save_steps=${save_steps}"
		--set "output_dir=${checkpoint_prefix}"
		--set "logging_dir=${tensorboard_prefix}"
		--set "final_model_dir=${model_prefix}"
		--set "report_to=tensorboard"
	)

	if [[ "${resume}" == "1" ]]; then
		train_cmd+=(--resume)
	fi

	echo "[INFO] Starting training..."
	"${train_cmd[@]}" 2>&1 | tee "${train_log}"
else
	echo "[INFO] Skipping training."
fi

if [[ ! -d "${model_dir}" ]]; then
	echo "Error: model directory does not exist: ${model_dir}"
	echo "Run without --skip-train first, or check the training output."
	exit 1
fi

if [[ "${reuse_embeddings}" == "0" && -f "${embedding_cache}" ]]; then
	echo "[INFO] Removing stale embedding cache: ${embedding_cache}"
	rm -f "${embedding_cache}"
fi

eval_cmd=(
	"${PYTHON[@]}" evaluation.py
	"${test_pool}"
	"${test_map}"
	"${model_dir}"
	--max-length "${max_length}"
	--batch-size "${batch_size}"
	--gpu-batch-size "${gpu_batch_size}"
	--eval-samples "${eval_samples}"
	--pool-samples "${pool_samples}"
	--embeddings-path "${embedding_cache}"
	--markdown-output "${markdown_result}"
	--cfg
	--ddg
	--bf16
)

echo "[INFO] Starting test evaluation..."
"${eval_cmd[@]}" 2>&1 | tee "${test_log}"

echo "[INFO] Done."
echo "[INFO] Test Markdown: ${markdown_result}"
echo "[INFO] Test log: ${test_log}"
