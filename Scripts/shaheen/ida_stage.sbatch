#!/bin/bash
# IDA 9.4 stages on Shaheen III, one exclusive node per array task:
#   i64  : binary -> <out>/<rel>.i64   (task1 step 1, via idalib instead of idat)
#   lift : input  -> <out>/<rel>.ll    (ida2llvm.py; input may be a binary or a .i64)
#
#   .venv/bin/python Scripts/shaheen/make_lift_list.py Binaries/Dataset-1 > i64_inputs.txt
#   sbatch --array=0-7 Scripts/shaheen/ida_stage.sbatch i64 i64_inputs.txt Binaries/Dataset-1 $DATA/Dataset-1-i64
# Site settings (repository, SIF, IDA install, IDAUSR template): Scripts/shaheen/cluster_env.sh.
# Resubmit the same command to resume. Per-file results: <out>/_<stage>_status/shard_*.jsonl
#SBATCH --job-name=regraph-ida
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
USAGE="usage: sbatch --array=0-N ida_stage.sbatch {i64|lift} inputs.txt in_root out_root"
STAGE_NAME=${1:?$USAGE}
LIST=$(realpath "${2:?$USAGE}")
IN_ROOT=$(realpath "${3:?$USAGE}")
OUT_ROOT=$(realpath -m "${4:?$USAGE}")
WORKERS=${WORKERS:-160}
TIMEOUT=${TIMEOUT:-14400}

source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"
REPO=$REGRAPH_REPO
DRIVER=$REGRAPH_DRIVER
IDA_TAR=${IDA_TAR:?set IDA_TAR to the IDA 9.4 .tar.zst}

case "$STAGE_NAME" in
    i64)  ANALYZE=$REPO/Scripts/shaheen/make_i64.py; export I64_IN_ROOT=$IN_ROOT I64_OUT_ROOT=$OUT_ROOT ;;
    lift) ANALYZE=$REPO/Scripts/shaheen/lift_one.py; export LIFT_IN_ROOT=$IN_ROOT LIFT_OUT_ROOT=$OUT_ROOT LIFT_TARGET=${LIFT_TARGET:-} ;;
    bbcount) ANALYZE=$REPO/Scripts/shaheen/bb_count.py; export BB_IN_ROOT=$IN_ROOT BB_OUT_ROOT=$OUT_ROOT BB_LIFT_ROOT=${BB_LIFT_ROOT:?set BB_LIFT_ROOT to the .ll tree} ;;
    funcflags) ANALYZE=$REPO/Scripts/shaheen/func_flags.py; export FF_IN_ROOT=$IN_ROOT FF_OUT_ROOT=$OUT_ROOT ;;
    *) echo "$USAGE" >&2; exit 2 ;;
esac

STAGE=/tmp/regraph-ida.${SLURM_JOB_ID}
mkdir -p "$STAGE/work" "$STAGE/logs"
# ida2llvm appends DEBUG logs to <repo>/logs/ida2llvm.log; one shared Lustre file for every process on
# every node serializes them on Lustre range locks. Each node logs to its own /tmp (bind-mounted over
# <repo>/logs below) and saves it compressed under <out>/_<stage>_logs/ when the task ends.
save_logs() {
    mkdir -p "$OUT_ROOT/_${STAGE_NAME}_logs"
    for f in "$STAGE"/logs/*; do
        [ -s "$f" ] && gzip -c "$f" > "$OUT_ROOT/_${STAGE_NAME}_logs/${SLURM_JOB_ID}_$(basename "$f").gz"
    done
    rm -rf "$STAGE"
}
trap save_logs EXIT
zstd -dc -T0 "$IDA_TAR" | tar -C "$STAGE" -xf -

export IDADIR=$STAGE/ida-pro-9.4
export PYTHONPATH=$IDADIR/idalib/python  # idapro is not in the uv lock; load it from the IDA install
# license, EULA, offline TlmHost (IDAUSR_TEMPLATE). Every IDA process locks IDAUSR/ida.reg; with
# many nodes on one Lustre copy the locks fail (EAGAIN, "ida.reg: Resource temporarily unavailable"), so each
# node works on its own copy.
cp -r "${IDAUSR_TEMPLATE:?set IDAUSR_TEMPLATE}" "$STAGE/idausr"
export IDAUSR=$STAGE/idausr
export TMPDIR=$STAGE/work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "node=$(hostname) stage=$STAGE_NAME array=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1} workers=$WORKERS timeout=$TIMEOUT"
# ida2llvm verifies its output with llvm-as, so run inside the LLVM 18 SIF (env and /tmp carry over)
# shellcheck disable=SC2086
singularity exec $SINGULARITY_BINDS --bind "$STAGE/logs:$REPO/logs" \
    "$REGRAPH_SIF" \
    "$REPO/.venv/bin/python" "$DRIVER" --list "$LIST" --out "$OUT_ROOT/_${STAGE_NAME}_status" \
    --analyze "$ANALYZE" --workers "$WORKERS" --timeout "$TIMEOUT"
