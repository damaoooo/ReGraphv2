#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifact_packages}"
OUTPUT_NAME="${OUTPUT_NAME:-rell_artifact_core.tar.zst}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/IR/Dataset-1-Oc-fused}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/runs/dataset1_oc_fused/model_cfg_ddg}"
PYTHON="${PYTHON:-python}"
INCLUDE_RESULTS=1
DRY_RUN=0

print_help() {
  cat <<'EOF'
Usage:
  bash Scripts/pack_artifact_release.sh [OPTIONS]

Packages the compact ReLL artifact data/model bundle.  The archive is
intended for external storage, not for committing to GitHub.

Included by default:
  IR/Dataset-1-Oc-fused/train_final_set
  IR/Dataset-1-Oc-fused/validation_final_set
  IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup
  runs/dataset1_oc_fused/model_cfg_ddg
  selected result Markdown snapshots

Options:
  --output-dir DIR       Directory where archive and manifest are written.
  --output-name NAME     Archive filename. Default: rell_artifact_core.tar.zst
  --dataset-root DIR     Dataset root. Default: REPO_ROOT/IR/Dataset-1-Oc-fused
  --model-dir DIR        Model dir. Default: REPO_ROOT/runs/dataset1_oc_fused/model_cfg_ddg
  --python PATH          Python executable with the datasets package installed.
  --no-results           Do not include result Markdown snapshots.
  --dry-run              Print included paths and sizes without writing archive.
  -h, --help             Show this help.

Example:
  bash Scripts/pack_artifact_release.sh --output-dir /home/user/rell_artifacts
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --no-results)
      INCLUDE_RESULTS=0
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

DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

relative_to_repo() {
  local path="$1"
  case "${path}" in
    "${REPO_ROOT}"/*) printf '%s\n' "${path#"${REPO_ROOT}/"}" ;;
    *) echo "Error: path is outside repo root: ${path}" >&2; exit 1 ;;
  esac
}

dataset_rel="$(relative_to_repo "${DATASET_ROOT}")"
model_rel="$(relative_to_repo "${MODEL_DIR}")"

include_paths=(
  "${dataset_rel}/train_final_set"
  "${dataset_rel}/validation_final_set"
  "${dataset_rel}/test_final_set_len128_hashdedup"
  "${model_rel}"
)

if [[ "${INCLUDE_RESULTS}" == "1" ]]; then
  optional_results=(
    "runs/dataset1_oc_fused/oc_test_results_len128_hashdedup.md"
    "runs/dataset1_oc_fused/common_oc_csv_results.md"
    "runs/dataset1_oc_graph_ablation/seed_eval_10_summary.md"
    "runs/dataset_vulnerability_regraph/results_bitnorm_fusion_max.md"
    "runs/dataset_vulnerability_regraph/vuln_search_big_table.md"
  )
  for path in "${optional_results[@]}"; do
    if [[ -f "${REPO_ROOT}/${path}" ]]; then
      include_paths+=("${path}")
    else
      echo "[WARN] result snapshot missing, skipping: ${path}" >&2
    fi
  done
fi

for path in "${include_paths[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
    echo "Error: required artifact path does not exist: ${REPO_ROOT}/${path}" >&2
    exit 1
  fi
done

archive="${OUTPUT_DIR}/${OUTPUT_NAME}"
manifest="${OUTPUT_DIR}/${OUTPUT_NAME%.*}.MANIFEST.md"

echo "[INFO] Repository root: ${REPO_ROOT}"
echo "[INFO] Output archive: ${archive}"
echo "[INFO] Manifest: ${manifest}"
echo "[INFO] Included paths:"
printf '  %s\n' "${include_paths[@]}"
du -sh "${include_paths[@]/#/${REPO_ROOT}/}" 2>/dev/null || true

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] Dry run complete."
  exit 0
fi

stage_root="$(mktemp -d "${OUTPUT_DIR}/.rell-artifact-stage.XXXXXX")"
cleanup() {
  rm -rf -- "${stage_root}"
}
trap cleanup EXIT

echo "[INFO] Staging files for path anonymization."
for path in "${include_paths[@]}"; do
  mkdir -p "${stage_root}/$(dirname "${path}")"
  cp -a --reflink=auto "${REPO_ROOT}/${path}" "${stage_root}/${path}"
done

# This training-state pickle is unnecessary for evaluation and can contain
# local output-directory names.  The model weights and tokenizer are retained.
rm -f -- "${stage_root}/${model_rel}/training_args.bin"

"${PYTHON}" "${SCRIPT_DIR}/sanitize_artifact_paths.py" "${stage_root}"

if command -v zstd >/dev/null 2>&1; then
  tar -I "zstd -T0 -3" -cf "${archive}" -C "${stage_root}" "${include_paths[@]}"
else
  archive="${archive%.zst}.gz"
  echo "[WARN] zstd not found; writing gzip archive instead: ${archive}" >&2
  tar -czf "${archive}" -C "${stage_root}" "${include_paths[@]}"
fi

archive_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
{
  echo "# ReLL Artifact Core Manifest"
  echo
  echo "- Archive: \`$(basename "${archive}")\`"
  echo "- SHA-256: \`${archive_sha256}\`"
  echo "- Generated at: $(date '+%F %T %z')"
  echo
  echo "## Included Paths"
  echo
  for path in "${include_paths[@]}"; do
    echo "- \`${path}\`"
  done
  echo
  echo "## Sizes"
  echo
  for path in "${include_paths[@]}"; do
    size="$(du -sh "${stage_root}/${path}" 2>/dev/null | awk '{print $1}')"
    echo "- \`${path}\`: ${size}"
  done
} > "${manifest}"

du -h "${archive}" "${manifest}"
echo "[INFO] SHA-256: ${archive_sha256}"
echo "[INFO] Done."
