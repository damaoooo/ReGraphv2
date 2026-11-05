#!/bin/bash
# coreutils step F final: build_split.py on the fused parquet of one level (one split named coreutils, wash at 2048,
# split_train_validation dataset1-xm) -> OUT/coreutils_final_set.   sbatch cu_final.sbatch OUT
#SBATCH --exclusive -N1 --cpus-per-task=384 --mem=0 -t 04:00:00 -J cu-final -o logs/%x-%j.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
OUT=$1
L=/tmp/r2cufinal.$SLURM_JOB_ID; mkdir -p $L; trap 'rm -rf $L' EXIT
singularity exec $SINGULARITY_BINDS $REGRAPH_SIF \
  env TMPDIR=$L HF_DATASETS_CACHE=$L/hf OMP_NUM_THREADS=1 USE_TORCH=0 bash -c "cd $REGRAPH_REPO && .venv/bin/python Scripts/shaheen/build_split.py --out $OUT --split coreutils --local \$TMPDIR/final --workers 192 --naming dataset1-xm"
