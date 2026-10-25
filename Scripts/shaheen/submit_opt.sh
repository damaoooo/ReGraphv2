#!/bin/bash
# Submit steps 3-6 for one or more opt levels: N shard nodes, then 3 final nodes (one per split).
#   Scripts/shaheen/submit_opt.sh /scratch/zhoul0e/Dataset-1-ll /scratch/zhoul0e/Dataset-1-{OPT}-fused 8 O0 O1 O2 O3 Os Og Oz Oc
# {OPT} in the out pattern is replaced by each opt level. Extra sbatch args go in SBATCH_ARGS
# (e.g. SBATCH_ARGS="-t 01:00:00"). Rerunning resumes; it keeps each level's shard plan.
set -euo pipefail
LL_ROOT=${1:?usage: submit_opt.sh ll_root out_pattern nshards OPT...}
OUT_PATTERN=${2:?}
NSHARDS=${3:?}
shift 3
REPO=/scratch/zhoul0e/ReGraphv2
SB="$REPO/Scripts/shaheen/opt_stage.sbatch"

for OPT in "$@"; do
    OUT=${OUT_PATTERN//\{OPT\}/$OPT}
    # Resume only our own runs: task2 skips existing .bc, so a foreign directory (e.g. an older
    # pipeline's output at the same path) would leak stale modules into the dataset.
    if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && [ ! -d "$OUT/shards" ]; then
        echo "$OUT exists and was not made by this pipeline (no shards/); move it away first" >&2
        exit 1
    fi
    "$REPO/.venv/bin/python" "$REPO/Scripts/shaheen/plan_shards.py" "$LL_ROOT" "$OUT" "$NSHARDS"
    # shellcheck disable=SC2086
    s=$(sbatch --parsable ${SBATCH_ARGS:-} --job-name="$OPT-shard" --array=0-$((NSHARDS - 1)) "$SB" shard "$OPT" "$LL_ROOT" "$OUT")
    # shellcheck disable=SC2086
    f=$(sbatch --parsable ${SBATCH_ARGS:-} --job-name="$OPT-final" --array=0-2 --dependency=afterok:$s "$SB" final "$OPT" "$LL_ROOT" "$OUT")
    echo "$OPT: shard=$s final=$f -> $OUT"
done
