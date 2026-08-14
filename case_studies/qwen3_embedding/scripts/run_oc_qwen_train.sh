#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_oc_qwen_train.sh [--resume] [--dry-run]

Environment overrides:
  OC_ROOT              Dataset root. Default:
                       /path/to/rell/IR/Dataset-1-Oc-qwen-text-fused
  OUTPUT_DIR           Fine-tuned adapter output dir. Default:
                       $RELL_ROOT/experiments/qwen-llvm-finetune-hf-oc
  CONFIG_PATH          Generated training config path. Default:
                       $RELL_ROOT/configs/train_oc_config.yaml
  CONDA_ENV            Conda env for training. Default: ml
  STEPS_PER_EPOCH      Training samples per epoch. Default: 600000
  STEPS_PER_EPOCH_EVAL Validation samples per eval pass. Default: 128
  TRAIN_BATCH_SIZE     Per-device train batch size. Default: 16
  EVAL_BATCH_SIZE      Per-device eval batch size. Default: 8
  EVAL_STRATEGY        Evaluation strategy. Default: no
  SAVE_STEPS           Checkpoint save interval. Default: 500
EOF
}

RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume|-r)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OC_ROOT="${OC_ROOT:-/path/to/rell/IR/Dataset-1-Oc-qwen-text-fused}"
TRAIN_FINAL_SET="${TRAIN_FINAL_SET:-${OC_ROOT}/train_final_set}"
VALIDATION_FINAL_SET="${VALIDATION_FINAL_SET:-${OC_ROOT}/validation_final_set}"
TRAIN_POOL="${TRAIN_POOL:-${TRAIN_FINAL_SET}/train_dataset_pool}"
TRAIN_MAP="${TRAIN_MAP:-${TRAIN_FINAL_SET}/train_positive_map.pkl}"
VALIDATION_POOL="${VALIDATION_POOL:-${VALIDATION_FINAL_SET}/train_dataset_pool}"
VALIDATION_MAP="${VALIDATION_MAP:-${VALIDATION_FINAL_SET}/train_positive_map.pkl}"

BASE_CONFIG="${BASE_CONFIG:-${RELL_ROOT}/configs/train_config.yaml}"
CONFIG_PATH="${CONFIG_PATH:-${RELL_ROOT}/configs/train_oc_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${RELL_ROOT}/experiments/qwen-llvm-finetune-hf-oc}"

RUN_ID="${RUN_ID:-qwen_llvm_finetune_oc}"
LOG_DIR="${LOG_DIR:-${RELL_ROOT}/experiments/train_logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log}"

CONDA_ENV="${CONDA_ENV:-ml}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
QUANTIZATION_BITS="${QUANTIZATION_BITS:-4}"
INSTRUCTION="${INSTRUCTION:-Represent this LLVM IR for searching for similar functions:}"

STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-600000}"
STEPS_PER_EPOCH_EVAL="${STEPS_PER_EPOCH_EVAL:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-false}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
TRIPLET_MARGIN="${TRIPLET_MARGIN:-1.0}"
SEED="${SEED:-42}"
BF16="${BF16:-true}"
FP16="${FP16:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

die() {
  echo "Error: $*" >&2
  exit 1
}

[[ -f "${BASE_CONFIG}" ]] || die "base config not found: ${BASE_CONFIG}"
[[ -d "${TRAIN_POOL}" ]] || die "train dataset pool not found: ${TRAIN_POOL}"
[[ -f "${TRAIN_MAP}" ]] || die "train positive map not found: ${TRAIN_MAP}"
[[ -d "${VALIDATION_POOL}" ]] || die "validation dataset pool not found: ${VALIDATION_POOL}"
[[ -f "${VALIDATION_MAP}" ]] || die "validation positive map not found: ${VALIDATION_MAP}"

cd "${RELL_ROOT}"

if [[ -n "${CONDA_ENV}" ]]; then
  [[ -f "${CONDA_SH}" ]] || die "conda hook not found: ${CONDA_SH}"
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
fi

mkdir -p "$(dirname "${CONFIG_PATH}")" "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ "${RESUME}" != "1" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
  if find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
    die "found existing checkpoints in ${OUTPUT_DIR}; rerun with --resume or ALLOW_EXISTING_OUTPUT=1"
  fi
fi

python - <<PY
import pickle
from datasets import load_from_disk

checks = [
    ("train", "${TRAIN_POOL}", "${TRAIN_MAP}"),
    ("validation", "${VALIDATION_POOL}", "${VALIDATION_MAP}"),
]

for name, pool_path, map_path in checks:
    dataset = load_from_disk(pool_path)
    with open(map_path, "rb") as handle:
        positive_map = pickle.load(handle)
    if len(dataset) == 0:
        raise SystemExit(f"{name} dataset is empty: {pool_path}")
    if len(positive_map) == 0:
        raise SystemExit(f"{name} positive map is empty: {map_path}")
    print(f"{name}: {len(dataset):,} rows, {len(positive_map):,} anchors")
PY

export BASE_CONFIG CONFIG_PATH OUTPUT_DIR
export TRAIN_POOL TRAIN_MAP VALIDATION_POOL VALIDATION_MAP
export MODEL_NAME MAX_LENGTH QUANTIZATION_BITS INSTRUCTION
export STEPS_PER_EPOCH STEPS_PER_EPOCH_EVAL NUM_TRAIN_EPOCHS
export TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE WARMUP_RATIO LOGGING_STEPS EVAL_STRATEGY EVAL_STEPS
export SAVE_STRATEGY SAVE_STEPS SAVE_TOTAL_LIMIT LOAD_BEST_MODEL_AT_END
export DATALOADER_NUM_WORKERS TRIPLET_MARGIN SEED
export BF16 FP16 GRADIENT_CHECKPOINTING

python - <<'PY'
import os
from pathlib import Path

import yaml


def as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def getenv_int(key: str) -> int:
    return int(os.environ[key])


def getenv_float(key: str) -> float:
    return float(os.environ[key])


base_config = Path(os.environ["BASE_CONFIG"])
config_path = Path(os.environ["CONFIG_PATH"])

with base_config.open("r") as handle:
    config = yaml.safe_load(handle)

config["model"]["name"] = os.environ["MODEL_NAME"]
config["model"]["max_length"] = getenv_int("MAX_LENGTH")
config["model"]["quantization_bits"] = getenv_int("QUANTIZATION_BITS")
config["model"]["instruction"] = os.environ["INSTRUCTION"]

config["data"]["train_dataset_pool_path"] = os.environ["TRAIN_POOL"]
config["data"]["train_positive_map_path"] = os.environ["TRAIN_MAP"]
config["data"]["validation_dataset_pool_path"] = os.environ["VALIDATION_POOL"]
config["data"]["validation_positive_map_path"] = os.environ["VALIDATION_MAP"]
config["data"]["steps_per_epoch"] = getenv_int("STEPS_PER_EPOCH")
config["data"]["steps_per_epoch_eval"] = getenv_int("STEPS_PER_EPOCH_EVAL")

training = config["training"]
training["output_dir"] = os.environ["OUTPUT_DIR"]
training["num_train_epochs"] = getenv_float("NUM_TRAIN_EPOCHS")
training["per_device_train_batch_size"] = getenv_int("TRAIN_BATCH_SIZE")
training["per_device_eval_batch_size"] = getenv_int("EVAL_BATCH_SIZE")
training["gradient_accumulation_steps"] = getenv_int("GRADIENT_ACCUMULATION_STEPS")
training["learning_rate"] = getenv_float("LEARNING_RATE")
training["warmup_ratio"] = getenv_float("WARMUP_RATIO")
training["logging_steps"] = getenv_int("LOGGING_STEPS")
training["eval_strategy"] = os.environ["EVAL_STRATEGY"]
training["eval_steps"] = getenv_int("EVAL_STEPS")
training["save_strategy"] = os.environ["SAVE_STRATEGY"]
training["save_steps"] = getenv_int("SAVE_STEPS")
training["save_total_limit"] = getenv_int("SAVE_TOTAL_LIMIT")
training["load_best_model_at_end"] = as_bool(os.environ["LOAD_BEST_MODEL_AT_END"])
training["dataloader_num_workers"] = getenv_int("DATALOADER_NUM_WORKERS")
training["triplet_margin"] = getenv_float("TRIPLET_MARGIN")
training["seed"] = getenv_int("SEED")
training["bf16"] = as_bool(os.environ["BF16"])
training["fp16"] = as_bool(os.environ["FP16"])
training["gradient_checkpointing"] = as_bool(os.environ["GRADIENT_CHECKPOINTING"])

if training["eval_strategy"] == "no":
    training["do_eval"] = False
    training["load_best_model_at_end"] = False
    training.pop("metric_for_best_model", None)
    training.pop("greater_is_better", None)

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

print(f"Wrote training config: {config_path}")
PY

TRAIN_COMMAND=(python "${RELL_ROOT}/train.py" "${CONFIG_PATH}")
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_COMMAND+=(--resume)
fi

echo "Output dir: ${OUTPUT_DIR}"
echo "Log path: ${LOG_PATH}"
echo "Command: ${TRAIN_COMMAND[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete; training was not started."
  exit 0
fi

"${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"
