#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifact_packages}"
OUTPUT_NAME="${OUTPUT_NAME:-regraphv2_artifact_core.tar.zst}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/IR/Dataset-1-Oc-fused}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/runs/dataset1_oc_fused/model_cfg_ddg}"
INCLUDE_RESULTS=1
DRY_RUN=0

print_help() {
  cat <<'EOF'
Usage:
  bash Scripts/pack_artifact_release.sh [OPTIONS]

Packages the compact ReGraphv2 artifact data/model bundle.  The archive is
intended for external storage, not for committing to GitHub.

Included by default:
  IR/Dataset-1-Oc-fused/train_final_set
  IR/Dataset-1-Oc-fused/validation_final_set
  IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup
  runs/dataset1_oc_fused/model_cfg_ddg
  selected result Markdown snapshots

Options:
  --output-dir DIR       Directory where archive and manifest are written.
  --output-name NAME     Archive filename. Default: regraphv2_artifact_core.tar.zst
  --dataset-root DIR     Dataset root. Default: REPO_ROOT/IR/Dataset-1-Oc-fused
  --model-dir DIR        Model dir. Default: REPO_ROOT/runs/dataset1_oc_fused/model_cfg_ddg
  --no-results           Do not include result Markdown snapshots.
  --dry-run              Print included paths and sizes without writing archive.
  -h, --help             Show this help.

Example:
  bash Scripts/pack_artifact_release.sh --output-dir /home/user/regraphv2_artifacts
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

{
  echo "# ReGraphv2 Artifact Core Manifest"
  echo
  echo "- Repository root: \`${REPO_ROOT}\`"
  echo "- Archive: \`${archive}\`"
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
  du -sh "${include_paths[@]/#/${REPO_ROOT}/}" 2>/dev/null || true
} > "${manifest}"

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

if command -v zstd >/dev/null 2>&1; then
  tar -I "zstd -T0 -3" -cf "${archive}" -C "${REPO_ROOT}" "${include_paths[@]}"
else
  archive="${archive%.zst}.gz"
  echo "[WARN] zstd not found; writing gzip archive instead: ${archive}" >&2
  tar -czf "${archive}" -C "${REPO_ROOT}" "${include_paths[@]}"
fi

du -h "${archive}" "${manifest}"
echo "[INFO] Done."
