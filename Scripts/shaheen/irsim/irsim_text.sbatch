#!/bin/bash
# step I/G input: normalized IR texts of every function for one level (irsim_text.py, the coreutils IR-similarity
# code), one shard per array task.   sbatch --array=0-15 irsim_text.sbatch BC_ROOT LIST_DIR OUT_DIR
#SBATCH --exclusive -N1 --cpus-per-task=384 --mem=0 -t 06:00:00 -J text -o logs/%x-%A_%a.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
BC=$1 LISTS=$2 OUT=$3 K=$(printf %03d $SLURM_ARRAY_TASK_ID)
LOCAL=/tmp/r2text.$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID; mkdir -p $LOCAL; trap 'rm -rf $LOCAL' EXIT
singularity exec $SINGULARITY_BINDS $REGRAPH_SIF \
    env TMPDIR=$LOCAL OMP_NUM_THREADS=1 USE_TORCH=0 $REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/irsim/irsim_text.py \
    --bc-root $BC --list $LISTS/shard_$K.txt --out $OUT --workers 384 --tag $K
