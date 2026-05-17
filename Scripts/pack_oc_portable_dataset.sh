#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/IR/Dataset-1-Oc-fused}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/IR/Dataset-1-Oc-fused-portable.tar.zst}"
INCLUDE_UNFILTERED_TEST=0
DRY_RUN=0

print_help() {
  cat <<'EOF'
Usage:
  bash Scripts/pack_oc_portable_dataset.sh [OPTIONS]

Packages the portable OC dataset needed for train/eval on another machine.
By default it includes:
  Dataset-1-Oc-fused/train_final_set
  Dataset-1-Oc-fused/validation_final_set
  Dataset-1-Oc-fused/test_final_set_len128_hashdedup

Options:
  --dataset-root DIR          Dataset root. Default: REPO_ROOT/IR/Dataset-1-Oc-fused
  --output FILE               Output archive. Default: REPO_ROOT/IR/Dataset-1-Oc-fused-portable.tar.zst
  --include-unfiltered-test   Also include Dataset-1-Oc-fused/test_final_set
  --dry-run                   Print what would be packaged.
  -h, --help                  Show this help.

Unpack on another machine:
  tar -I zstd -xf Dataset-1-Oc-fused-portable.tar.zst -C /path/to/regraphv2/IR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --include-unfiltered-test)
      INCLUDE_UNFILTERED_TEST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Error: dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

dataset_name="$(basename "${DATASET_ROOT}")"
dataset_parent="$(cd "$(dirname "${DATASET_ROOT}")" && pwd)"

include_paths=(
  "${dataset_name}/train_final_set"
  "${dataset_name}/validation_final_set"
  "${dataset_name}/test_final_set_len128_hashdedup"
)

if [[ "${INCLUDE_UNFILTERED_TEST}" == "1" ]]; then
  include_paths+=("${dataset_name}/test_final_set")
fi

for path in "${include_paths[@]}"; do
  if [[ ! -e "${dataset_parent}/${path}" ]]; then
    echo "Error: required path does not exist: ${dataset_parent}/${path}" >&2
    exit 1
  fi
done

echo "[INFO] Dataset root: ${DATASET_ROOT}"
echo "[INFO] Output archive: ${OUTPUT}"
echo "[INFO] Included paths:"
printf '  %s\n' "${include_paths[@]}"
du -sh "${include_paths[@]/#/${dataset_parent}/}" 2>/dev/null || true

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

mkdir -p "$(dirname "${OUTPUT}")"

if command -v zstd >/dev/null 2>&1; then
  tar -I "zstd -T0 -3" -cf "${OUTPUT}" -C "${dataset_parent}" "${include_paths[@]}"
else
  fallback_output="${OUTPUT%.zst}.gz"
  echo "[WARN] zstd not found; writing gzip archive instead: ${fallback_output}" >&2
  tar -czf "${fallback_output}" -C "${dataset_parent}" "${include_paths[@]}"
  OUTPUT="${fallback_output}"
fi

du -h "${OUTPUT}"
echo "[INFO] Done."
