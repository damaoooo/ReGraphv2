#!/bin/bash

set -euo pipefail

cd "/home/damaoooo/Downloads/regraphv2"

use_cfg=1
use_ddg=1
steps=""
output_file=""

build_graph_tag() {
	if [[ "${use_cfg}" == "1" && "${use_ddg}" == "1" ]]; then
		echo "cfg_ddg"
	elif [[ "${use_cfg}" == "1" ]]; then
		echo "cfg"
	elif [[ "${use_ddg}" == "1" ]]; then
		echo "ddg"
	else
		echo "plain"
	fi
}

print_help() {
	cat <<'EOF'
Usage: bash Scripts/ablation.sh [OPTIONS]

Options:
	--cfg / --no-cfg      Enable or disable CFG graph branch.
												Default: enabled
	--ddg / --no-ddg      Enable or disable DDG graph branch.
												Default: enabled
	--steps N             Optional max training steps.
												Default: disabled; train 1 epoch.
	--output_file FILE    Write evaluation output (stdout/stderr) to FILE.
								Training output stays in terminal.
												If omitted, defaults to
												./logs_ablation<step_tag>_<graph_tag>.log
	-h, --help            Show this help message and exit.

Example:
	bash Scripts/ablation.sh --no-cfg --ddg --steps 40000 \
		--output_file ./logs/my_ablation.log
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--cfg)
			use_cfg=1
			shift
			;;
		--no-cfg)
			use_cfg=0
			shift
			;;
		--ddg)
			use_ddg=1
			shift
			;;
		--no-ddg)
			use_ddg=0
			shift
			;;
		--steps)
			if [[ $# -lt 2 ]]; then
				echo "Error: --steps requires a value."
				exit 1
			fi
			steps="$2"
			shift 2
			;;
		--output_file)
			if [[ $# -lt 2 ]]; then
				echo "Error: --output_file requires a value."
				exit 1
			fi
			output_file="$2"
			shift 2
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

if [[ -n "${steps}" && ! "${steps}" =~ ^[0-9]+$ ]]; then
	echo "Error: --steps must be a positive integer."
	exit 1
fi

step_tag="${steps:-1epoch}"
graph_tag="$(build_graph_tag)"
if [[ -z "${output_file}" ]]; then
	output_file="./logs_ablation${step_tag}_${graph_tag}.log"
fi

mkdir -p "$(dirname "${output_file}")"

echo "[INFO] cfg=${use_cfg}, ddg=${use_ddg}, max_steps=${steps:-disabled}"
echo "[INFO] evaluation output_file=${output_file}"

run_tag="ablation${step_tag}_${graph_tag}"
model_dir="./db1_model_${run_tag}"

train_graph_flags=()
eval_graph_flags=()
if [[ "${use_cfg}" == "1" ]]; then
	train_graph_flags+=(--cfg)
	eval_graph_flags+=(--cfg)
else
	train_graph_flags+=(--no-cfg)
	eval_graph_flags+=(--no-cfg)
fi
if [[ "${use_ddg}" == "1" ]]; then
	train_graph_flags+=(--ddg)
	eval_graph_flags+=(--ddg)
else
	train_graph_flags+=(--no-ddg)
	eval_graph_flags+=(--no-ddg)
fi

train_cmd=(
	python -m Pretrain.run_pretrain train "${train_graph_flags[@]}"
	--set output_dir=./output_${run_tag}
	--set final_model_dir=${model_dir}
	--set logging_dir=./logs_${run_tag}
	--set report_to=tensorboard
)
if [[ -n "${steps}" ]]; then
	train_cmd+=(--set max_steps=${steps})
fi
"${train_cmd[@]}"

if [[ -f "./db1_${graph_tag}_test.pkl.npy" ]]; then
    echo "[INFO] Removing existing evaluation results: ./db1_${graph_tag}_test.pkl.npy"
    rm "./db1_${graph_tag}_test.pkl.npy"
fi

echo "===== evaluation: dataset-1 =====" | tee -a "${output_file}"
python evaluation.py /home/damaoooo/Downloads/regraphv2/IR/dataset-1/train_final_set/train_dataset_pool  /home/damaoooo/Downloads/regraphv2/IR/dataset-1/test_final_set/train_positive_map.pkl ${model_dir} --max-length 4096 -e ./db1_${graph_tag}_test.pkl.npy -b 16 --gpu-batch-size 256 "${eval_graph_flags[@]}" --bf16 2>&1 | tee >(sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' | tr '\r' '\n' >> "${output_file}")

if [[ -f "./db2_${graph_tag}_test.pkl.npy" ]]; then
    echo "[INFO] Removing existing evaluation results: ./db2_${graph_tag}_test.pkl.npy"
    rm "./db2_${graph_tag}_test.pkl.npy"
fi

echo "===== evaluation: dataset-2 =====" | tee -a "${output_file}"
python evaluation.py /home/damaoooo/Downloads/regraphv2/IR/Dataset-2/db2_final_set/train_dataset_pool  /home/damaoooo/Downloads/regraphv2/IR/Dataset-2/db2_final_set/train_positive_map.pkl ${model_dir} --max-length 4096 -e ./db2_${graph_tag}_test.pkl.npy -b 16 --gpu-batch-size 256 "${eval_graph_flags[@]}" --bf16 2>&1 | tee >(sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' | tr '\r' '\n' >> "${output_file}")
