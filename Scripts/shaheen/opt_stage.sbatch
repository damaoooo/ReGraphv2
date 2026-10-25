#!/bin/bash
# Steps 3-6 for ONE opt level, inside the LLVM 18 SIF with the uv .venv. Submit via
# Scripts/shaheen/submit_opt.sh, which chains the phases with a Slurm dependency:
#   shard (array 0..N-1): task2_bc.py   .ll -> <out>/bc (+ hardlinked X.ll, X.ll.lift.json)
#                         fused_task3.py -> <out>/parquet/<split>/... and <out>/functions/.../function_map.csv
#   select (one node)   : select_level.py, IR dedup + Steps 5-6 over all splits -> <out>/selection/<split>.txt
#   final (array 0..2)  : build_split.py, one split per node -> <out>/<split>_final_set
#                         (only the selected rows when <out>/selection exists)
# Every phase resumes: shards skip finished modules (task2 outputs, fused manifest); final redoes a split.
#SBATCH --job-name=regraph-opt
#SBATCH --partition=workq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=384
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/zhoul0e/iops/projects/ida-batch/logs/%x-%A_%a.out

set -euo pipefail
USAGE="usage: opt_stage.sbatch {shard|select|final|recompile|corpus|retok} OPT_LEVEL ll_root out_root"
PHASE=${1:?$USAGE}
OPT=${2:?$USAGE}
LL_ROOT=$(realpath "${3:?$USAGE}")
OUT=$(realpath -m "${4:?$USAGE}")
TASK=${SLURM_ARRAY_TASK_ID:-0}
WORKERS=${WORKERS:-384}
CMD_TIMEOUT=${CMD_TIMEOUT:-28800}
BC_ROOT=${BC_ROOT:-$OUT/bc}  # retok reads the .bc of an earlier run of the same level
CORPUS_ROOT=${CORPUS_ROOT:-/scratch/zhoul0e/bpe-corpus}  # corpus phase: lists/<OPT>/part_<K>.txt in, <OPT>/ out
SELECT_ROOT=${SELECT_ROOT:-}  # select_functions.py output: shard/retok extract only the selected functions
NAMING=${NAMING:-dataset1}  # split_train_validation naming of select and final (dataset1-xm: XM positives)

REPO=/scratch/zhoul0e/ReGraphv2
SIF=/scratch/zhoul0e/regraph-data-env-llvm18.1.3.sif
LOCAL=/tmp/regraph-opt.${SLURM_JOB_ID}.${TASK}
mkdir -p "$OUT" "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT

module load singularity >/dev/null 2>&1
echo "node=$(hostname) phase=$PHASE task=$TASK opt=$OPT ll_root=$LL_ROOT out=$OUT workers=$WORKERS"

singularity exec --bind /scratch/zhoul0e:/scratch/zhoul0e --bind /tmp:/tmp "$SIF" \
    env PHASE="$PHASE" OPT="$OPT" LL_ROOT="$LL_ROOT" OUT="$OUT" TASK="$TASK" WORKERS="$WORKERS" BC_ROOT="$BC_ROOT" CORPUS_ROOT="$CORPUS_ROOT" \
        SELECT_ROOT="$SELECT_ROOT" NAMING="$NAMING" \
        CMD_TIMEOUT="$CMD_TIMEOUT" REPO="$REPO" LOCAL="$LOCAL" TMPDIR="$LOCAL" \
        HF_DATASETS_CACHE="$LOCAL/hf_cache" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash -euo pipefail -c '
cd "$REPO"
PY=$REPO/.venv/bin/python
SELECT=${SELECT_ROOT:+--select-root $SELECT_ROOT}
stage() { echo "[$(date "+%F %T")] $*"; }

case "$PHASE" in
shard)
    K=$(printf "%03d" "$TASK")
    LIST=$OUT/shards/shard_$K.txt
    stage "task2 shard $K start ($(wc -l < "$LIST") .ll)"
    $PY Scripts/shaheen/task2_bc.py "$LL_ROOT" "$OUT/bc" --opt-level "$OPT" --list "$LIST" \
        --manifest-tag "_$K" --workers "$WORKERS" --timeout "$CMD_TIMEOUT"
    stage "fused task3 shard $K start"
    $PY Scripts/shaheen/fused_task3.py --bc-root "$OUT/bc" --list "$LIST" --out "$OUT" \
        --workers "$WORKERS" --tag "$K" $SELECT
    ;;
recompile)
    K=$(printf "%03d" "$TASK")
    stage "task4 recompile shard $K start"
    $PY Scripts/shaheen/task4_re.py "$OUT/bc" "$REPO/Binaries/Dataset-1-$OPT-re" --list "$OUT/shards/shard_$K.txt" \
        --manifest-tag "_$K" --workers "$WORKERS" --timeout "$CMD_TIMEOUT"
    ;;
retok)
    # step 4 again with the current tokenizer, from existing .bc ($BC_ROOT) into a fresh $OUT
    K=$(printf "%03d" "$TASK")
    stage "retok shard $K start (bc from $BC_ROOT)"
    $PY Scripts/shaheen/fused_task3.py --bc-root "$BC_ROOT" --list "$OUT/shards/shard_$K.txt" --out "$OUT" \
        --workers "$WORKERS" --tag "$K" $SELECT
    ;;
select)
    stage "select start"
    $PY Scripts/shaheen/select_level.py --out "$OUT" --naming "$NAMING" --workers "$WORKERS"
    ;;
corpus)
    K=$(printf "%03d" "$TASK")
    stage "bpe corpus part $K start"
    $PY Scripts/shaheen/corpus_text.py --bc-root "$OUT/bc" --list "$CORPUS_ROOT/lists/$OPT/part_$K.txt" \
        --out "$CORPUS_ROOT/$OPT" --workers "$WORKERS" --tag "$K"
    ;;
final)
    split=$(echo train validation test | cut -d" " -f$((TASK + 1)))
    stage "final $split start"
    KEEP=""
    [ -d "$OUT/selection" ] && KEEP="--keep $OUT/selection/$split.txt"
    $PY Scripts/shaheen/build_split.py --out "$OUT" --split "$split" --local "$LOCAL/final" --workers "$WORKERS" \
        --naming "$NAMING" $KEEP
    ;;
*) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
stage "$PHASE done"
'
