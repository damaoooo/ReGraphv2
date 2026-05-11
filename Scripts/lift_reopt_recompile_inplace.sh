#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_PATH="${PYTHON_PATH:-/home/damaoooo/miniconda3/envs/ReLL/bin/python}"
CONDA_ENV="${CONDA_ENV:-ReLL}"
DEFAULT_WORKERS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
WORKERS="${WORKERS:-${DEFAULT_WORKERS}}"
OPT_LEVEL="${OPT_LEVEL:-O3}"
ARCH="${ARCH:-auto}"
RESUME=1
START_FROM_STEP2=0
FAIL_FAST=0
INPUT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  Scripts/lift_reopt_recompile_inplace.sh INPUT_DIR [options]

Runs the existing pipeline stages in place:
  1. task1_lift.py         binary/.i64 -> .ll
  2. task2_reoptimize.py   .ll -> .bc
  3. task4_recompile.py    .bc -> .re

The final .re files are written next to the source files under INPUT_DIR.

Options:
  --workers N           Worker process count. Default: CPU count
  --opt-level LEVEL     Reoptimization level for Task 2. Default: O3
  --arch MODE           auto, m32, or m64. Default: auto
  --resume              Skip completed outputs when matching state exists. Default
  --no-resume           Re-run existing outputs
  --fail-fast           Stop after a stage returns non-zero. Default keeps going
  --start-from-step2    Skip IDA .i64 generation and lift existing .i64 files
  --python PATH         Python executable. Default: /home/damaoooo/miniconda3/envs/ReLL/bin/python
  -h, --help            Show this help

Examples:
  Scripts/lift_reopt_recompile_inplace.sh Binaries/coreutils --workers 32 --arch auto
  Scripts/lift_reopt_recompile_inplace.sh Binaries/coreutils --start-from-step2 --resume
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      [[ $# -ge 2 ]] || { echo "Missing value for --workers" >&2; exit 2; }
      WORKERS="$2"
      shift 2
      ;;
    --opt-level)
      [[ $# -ge 2 ]] || { echo "Missing value for --opt-level" >&2; exit 2; }
      OPT_LEVEL="$2"
      shift 2
      ;;
    --arch)
      [[ $# -ge 2 ]] || { echo "Missing value for --arch" >&2; exit 2; }
      ARCH="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --start-from-step2)
      START_FROM_STEP2=1
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 2; }
      PYTHON_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${INPUT_DIR}" ]]; then
        echo "Only one INPUT_DIR is allowed: got '${INPUT_DIR}' and '$1'" >&2
        exit 2
      fi
      INPUT_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "${INPUT_DIR}" ]]; then
  usage >&2
  exit 2
fi

if ! RESOLVED_INPUT_DIR="$(realpath -e "${INPUT_DIR}")"; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 1
fi
INPUT_DIR="${RESOLVED_INPUT_DIR}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input path is not a directory: ${INPUT_DIR}" >&2
  exit 1
fi

INPUT_PARENT="$(dirname "${INPUT_DIR}")"

if [[ -x "${PYTHON_PATH}" ]]; then
  PYTHON_CMD=("${PYTHON_PATH}")
  ENV_BIN_DIR="$(dirname "${PYTHON_PATH}")"
  export PATH="${ENV_BIN_DIR}:${PATH}"

  if [[ "${PYTHON_PATH}" == */envs/*/bin/python ]]; then
    CONDA_ROOT="${PYTHON_PATH%%/envs/*}"
    if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
      export PATH="${CONDA_ROOT}/bin:${PATH}"
    fi
  fi
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
  echo "Python not executable and conda not found: ${PYTHON_PATH}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required because task1_lift.py runs ida2llvm through 'conda run -n ReLL'." >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "${RESUME}" -eq 1 ]]; then
  RESUME_ARGS=(--resume)
fi

START_ARGS=()
if [[ "${START_FROM_STEP2}" -eq 1 ]]; then
  START_ARGS=(--start-from-step2)
fi

cd "${REPO_ROOT}"

FAILED_STAGES=()

run_stage() {
  local stage_name="$1"
  shift

  set +e
  "$@"
  local status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "${stage_name} returned ${status}; continuing with remaining stages." >&2
    FAILED_STAGES+=("${stage_name}:${status}")
    if [[ "${FAIL_FAST}" -eq 1 ]]; then
      exit "${status}"
    fi
  fi

  return 0
}

echo "[1/3] Lifting to LLVM IR in place"
run_stage "task1_lift" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/task1_lift.py" \
  --input-path "${INPUT_DIR}" \
  --output "${INPUT_PARENT}" \
  --workers "${WORKERS}" \
  "${RESUME_ARGS[@]}" \
  "${START_ARGS[@]}"

echo "[2/3] Reoptimizing .ll to .bc"
run_stage "task2_reoptimize" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/task2_reoptimize.py" \
  --input-path "${INPUT_DIR}" \
  --workers "${WORKERS}" \
  --opt-level "${OPT_LEVEL}" \
  --arch "${ARCH}" \
  "${RESUME_ARGS[@]}"

echo "[3/3] Recompiling .bc to .re"
run_stage "task4_recompile" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/task4_recompile.py" \
  --input-path "${INPUT_DIR}" \
  --workers "${WORKERS}" \
  --arch "${ARCH}" \
  "${RESUME_ARGS[@]}"

RE_COUNT="$(find "${INPUT_DIR}" -type f -name '*.re' | wc -l | tr -d ' ')"
echo "Done. Found ${RE_COUNT} .re files under ${INPUT_DIR}"

if [[ "${#FAILED_STAGES[@]}" -gt 0 ]]; then
  echo "Stages with failures: ${FAILED_STAGES[*]}" >&2
fi
