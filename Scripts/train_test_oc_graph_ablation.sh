#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_CMD="${PYTHON_CMD:-python}"
read -r -a PYTHON <<< "${PYTHON_CMD}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/IR/Dataset-1-Oc-fused}"
TEST_SET="${TEST_SET:-${DATASET_ROOT}/test_final_set_len128_hashdedup}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/dataset1_oc_graph_ablation}"

MAX_STEPS="${MAX_STEPS:-300000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
EVAL_STEPS="${EVAL_STEPS:-}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"
POOL_SAMPLES="${POOL_SAMPLES:-0}"
RESUME=1
SKIP_TRAIN=0
REUSE_EMBEDDINGS=0
USE_CONDA=1

print_help() {
  cat <<'EOF'
Usage:
  bash Scripts/train_test_oc_graph_ablation.sh MODE [OPTIONS]

MODE:
  ddg       Train/evaluate with DDG only: --no-cfg --ddg
  cfg       Train/evaluate with CFG only: --cfg --no-ddg
  plain     Train/evaluate with no graph branches: --no-cfg --no-ddg
  all       Run ddg, cfg, and plain sequentially.

Options:
  --steps N              Max training steps. Default: 300000
  --save-steps N         Checkpoint interval. Default: 10000
  --eval-steps N         Eval-loss interval. Default: trainer default/save interval
  --max-length N         Train/eval max sequence length. Default: 2048
  --batch-size N         Embedding generation batch size. Default: 16
  --gpu-batch-size N     Retrieval GPU batch size. Default: 256
  --eval-samples N       Test anchors. 0 means all. Default: 0
  --pool-samples N       Retrieval pool samples. 0 means full pool. Default: 0
  --dataset-root DIR     OC fused dataset root.
  --test-set DIR         Test final_set directory. Default: DATASET_ROOT/test_final_set_len128_hashdedup
  --output-root DIR      Output root. Default: runs/dataset1_oc_graph_ablation
  --conda-env NAME       Conda env to activate. Default: keep current shell environment
  --fresh                Do not resume training.
  --skip-train           Only run evaluation with existing model.
  --reuse-embeddings     Reuse existing embedding cache.
  --no-conda             Do not activate CONDA_ENV.
  -h, --help             Show this help.

Environment:
  REPO_ROOT              Repository root.
  CONDA_ENV              Conda env to activate. Empty means keep current shell environment.
  PYTHON_CMD             Python command after env activation. Default: python
  DATASET_ROOT           Same as --dataset-root.
  TEST_SET               Same as --test-set.
  OUTPUT_ROOT            Same as --output-root.

Examples:
  bash Scripts/train_test_oc_graph_ablation.sh ddg
  CONDA_ENV=myenv bash Scripts/train_test_oc_graph_ablation.sh ddg
  bash Scripts/train_test_oc_graph_ablation.sh cfg --steps 50000
  bash Scripts/train_test_oc_graph_ablation.sh all --reuse-embeddings
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  print_help
  [[ $# -lt 1 ]] && exit 1 || exit 0
fi

MODE="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --save-steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --eval-steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --gpu-batch-size)
      GPU_BATCH_SIZE="$2"
      shift 2
      ;;
    --eval-samples)
      EVAL_SAMPLES="$2"
      shift 2
      ;;
    --pool-samples)
      POOL_SAMPLES="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --test-set)
      TEST_SET="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --fresh)
      RESUME=0
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --reuse-embeddings)
      REUSE_EMBEDDINGS=1
      shift
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

for numeric_arg in MAX_STEPS SAVE_STEPS MAX_LENGTH EVAL_BATCH_SIZE GPU_BATCH_SIZE EVAL_SAMPLES POOL_SAMPLES; do
  value="${!numeric_arg}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "Error: ${numeric_arg} must be a non-negative integer, got '${value}'." >&2
    exit 1
  fi
done
if [[ -n "${EVAL_STEPS}" && ! "${EVAL_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "Error: EVAL_STEPS must be a non-negative integer, got '${EVAL_STEPS}'." >&2
  exit 1
fi

case "${MODE}" in
  ddg|cfg|plain)
    MODES=("${MODE}")
    ;;
  all)
    MODES=(ddg cfg plain)
    ;;
  *)
    echo "Error: MODE must be one of ddg/cfg/plain/all, got '${MODE}'." >&2
    exit 1
    ;;
esac

if [[ "${USE_CONDA}" == "1" && -n "${CONDA_ENV}" ]]; then
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Error: cannot find conda. Set CONDA_SH=/path/to/conda.sh or use --no-conda." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"
elif [[ "${USE_CONDA}" == "1" && -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[INFO] Using active conda env: ${CONDA_DEFAULT_ENV}"
elif [[ "${USE_CONDA}" == "1" ]]; then
  echo "[INFO] CONDA_ENV is empty; using current PATH python. Set CONDA_ENV=name or pass --conda-env name to activate one."
fi

cd "${REPO_ROOT}"

TRAIN_SET="${DATASET_ROOT}/train_final_set"
VALIDATION_SET="${DATASET_ROOT}/validation_final_set"
TEST_POOL="${TEST_SET}/train_dataset_pool"
TEST_MAP="${TEST_SET}/train_positive_map.pkl"

for required_path in "${TRAIN_SET}" "${VALIDATION_SET}" "${TEST_POOL}" "${TEST_MAP}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Error: required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

graph_flags_for_mode() {
  local mode="$1"
  case "${mode}" in
    ddg)
      echo "--no-cfg --ddg"
      ;;
    cfg)
      echo "--cfg --no-ddg"
      ;;
    plain)
      echo "--no-cfg --no-ddg"
      ;;
    *)
      return 1
      ;;
  esac
}

run_one_mode() {
  local mode="$1"
  local tag="${mode}"
  local mode_root="${OUTPUT_ROOT}/${mode}"
  local checkpoint_prefix="${mode_root}/checkpoints"
  local tensorboard_prefix="${mode_root}/tensorboard"
  local model_prefix="${mode_root}/model"
  local model_dir="${model_prefix}_${tag}"
  local embeddings_path="${mode_root}/test_full_embeddings_${tag}.pth"
  local markdown_path="${mode_root}/test_full_results_${tag}.md"
  local graph_flags

  read -r -a graph_flags <<< "$(graph_flags_for_mode "${mode}")"

  mkdir -p "${mode_root}"

  echo
  echo "===== OC graph ablation: ${mode} ====="
  echo "[INFO] repo_root=${REPO_ROOT}"
  echo "[INFO] dataset_root=${DATASET_ROOT}"
  echo "[INFO] test_set=${TEST_SET}"
  echo "[INFO] output_root=${mode_root}"
  echo "[INFO] model_dir=${model_dir}"
  echo "[INFO] max_steps=${MAX_STEPS}, max_length=${MAX_LENGTH}"
  echo "[INFO] graph_flags=${graph_flags[*]}"
  echo "[INFO] markdown=${markdown_path}"

  if [[ "${SKIP_TRAIN}" == "0" ]]; then
    train_cmd=(
      "${PYTHON[@]}" -m Pretrain.run_pretrain train
      --dataset-dir "${TRAIN_SET}"
      --validation-dataset-dir "${VALIDATION_SET}"
      "${graph_flags[@]}"
      --set "max_seq_length=${MAX_LENGTH}"
      --set "save_steps=${SAVE_STEPS}"
      --set "max_steps=${MAX_STEPS}"
      --set "output_dir=${checkpoint_prefix}"
      --set "logging_dir=${tensorboard_prefix}"
      --set "final_model_dir=${model_prefix}"
      --set "report_to=tensorboard"
    )
    if [[ -n "${EVAL_STEPS}" ]]; then
      train_cmd+=(--set "eval_steps=${EVAL_STEPS}")
    fi
    if [[ "${RESUME}" == "1" ]]; then
      train_cmd+=(--resume)
    fi

    echo "[INFO] Starting training..."
    "${train_cmd[@]}"
  else
    echo "[INFO] Skipping training."
  fi

  if [[ ! -d "${model_dir}" ]]; then
    echo "Error: model directory does not exist: ${model_dir}" >&2
    echo "Run without --skip-train first, or check training output." >&2
    exit 1
  fi

  if [[ "${REUSE_EMBEDDINGS}" == "0" && -f "${embeddings_path}" ]]; then
    echo "[INFO] Removing stale embedding cache: ${embeddings_path}"
    rm -f "${embeddings_path}"
  fi

  echo "[INFO] Starting full-pool test evaluation..."
  "${PYTHON[@]}" evaluation.py \
    "${TEST_MAP}" \
    --dataset-path "${TEST_POOL}" \
    --model-path "${model_dir}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --gpu-batch-size "${GPU_BATCH_SIZE}" \
    --eval-samples "${EVAL_SAMPLES}" \
    --pool-samples "${POOL_SAMPLES}" \
    --embeddings-path "${embeddings_path}" \
    --markdown-output "${markdown_path}" \
    "${graph_flags[@]}" \
    --bf16

  echo "[INFO] Done: ${mode}"
  echo "[INFO] Result: ${markdown_path}"
}

for mode in "${MODES[@]}"; do
  run_one_mode "${mode}"
done

summary_path="${OUTPUT_ROOT}/summary.md"
"${PYTHON[@]}" - "${OUTPUT_ROOT}" "${summary_path}" "${MODES[@]}" <<'PY'
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
modes = sys.argv[3:]

rows = []
for mode in modes:
    path = output_root / mode / f"test_full_results_{mode}.md"
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    recall = None
    mrr = None
    for line in text.splitlines():
        if line.startswith("| 10,000 |") and recall is None:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                recall = cells[1]
        elif line.startswith("| 10,000 |") and recall is not None and mrr is None:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                mrr = cells[1]
    rows.append((mode, recall or "NA", mrr or "NA", str(path)))

if rows:
    lines = [
        "# OC Graph Ablation Summary",
        "",
        "| Mode | Recall@1 Pool10000 | MRR@P Pool10000 | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for mode, recall, mrr, path in rows:
        lines.append(f"| {mode} | {recall} | {mrr} | `{path}` |")
    lines.append("")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Summary: {summary_path}")
PY
