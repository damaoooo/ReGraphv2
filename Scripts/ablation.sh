#!/bin/bash

set -euo pipefail

cd "/home/damaoooo/Downloads/regraphv2"

graph_mode="cfg_as_ddg_no_ddg"
steps=30000
output_file=""

print_help() {
	cat <<'EOF'
Usage: bash Scripts/ablation.sh [OPTIONS]

Options:
	--graph_mode MODE     Graph mode for training/evaluation.
												Default: cfg_as_ddg_no_ddg
	--steps N             Max training steps.
												Default: 30000
	--output_file FILE    Write evaluation output (stdout/stderr) to FILE.
								Training output stays in terminal.
												If omitted, defaults to
												./logs_ablation<steps>_<graph_mode>.log
	-h, --help            Show this help message and exit.

Example:
	bash Scripts/ablation.sh --graph_mode cfg_as_ddg --steps 40000 \
		--output_file ./logs/my_ablation.log
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--graph_mode)
			if [[ $# -lt 2 ]]; then
				echo "Error: --graph_mode requires a value."
				exit 1
			fi
			graph_mode="$2"
			shift 2
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

if ! [[ "${steps}" =~ ^[0-9]+$ ]]; then
	echo "Error: --steps must be a positive integer."
	exit 1
fi

step_tag="${steps}"
if [[ -z "${output_file}" ]]; then
	output_file="./logs_ablation${step_tag}_${graph_mode}.log"
fi

mkdir -p "$(dirname "${output_file}")"

echo "[INFO] graph_mode=${graph_mode}, steps=${steps}"
echo "[INFO] evaluation output_file=${output_file}"

run_tag="ablation${step_tag}_${graph_mode}"
model_dir="./db1_model_${run_tag}"

python -m Pretrain.run_pretrain train --graph-mode ${graph_mode} --set max_steps=${steps} --set output_dir=./output_${run_tag} --set final_model_dir=${model_dir} --set logging_dir=./logs_${run_tag} --set report_to=tensorboard 

# if ./db1_${graph_mode}_test.pkl.npy exists, delete it to avoid using stale evaluation results
if [[ -f "./db1_${graph_mode}_test.pkl.npy" ]]; then
    echo "[INFO] Removing existing evaluation results: ./db1_${graph_mode}_test.pkl.npy"
    rm "./db1_${graph_mode}_test.pkl.npy"
fi

echo "===== evaluation: dataset-1 =====" | tee -a "${output_file}"
python evaluation.py /home/damaoooo/Downloads/regraphv2/IR/dataset-1/train_final_set/train_dataset_pool  /home/damaoooo/Downloads/regraphv2/IR/dataset-1/test_final_set/train_positive_map.pkl ${model_dir} --max-length 4096 -e ./db1_${graph_mode}_test.pkl.npy -b 16 --gpu-batch-size 256 --graph-mode ${graph_mode} --fp16 2>&1 | tee >(sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' | tr '\r' '\n' >> "${output_file}")

if [[ -f "./db2_${graph_mode}_test.pkl.npy" ]]; then
    echo "[INFO] Removing existing evaluation results: ./db2_${graph_mode}_test.pkl.npy"
    rm "./db2_${graph_mode}_test.pkl.npy"
fi

echo "===== evaluation: dataset-2 =====" | tee -a "${output_file}"
python evaluation.py /home/damaoooo/Downloads/regraphv2/IR/Dataset-2/db2_final_set/train_dataset_pool  /home/damaoooo/Downloads/regraphv2/IR/Dataset-2/db2_final_set/train_positive_map.pkl ${model_dir} --max-length 4096 -e ./db2_${graph_mode}_test.pkl.npy -b 16 --gpu-batch-size 256 --graph-mode ${graph_mode} --fp16 2>&1 | tee >(sed -r 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' | tr '\r' '\n' >> "${output_file}")