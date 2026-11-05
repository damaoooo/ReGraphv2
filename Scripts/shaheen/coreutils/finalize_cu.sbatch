#!/bin/bash
# coreutils step F finalize for one level: coreutils_final_set -> test_final_set (+binary_name, fva), finalize_stats.json
#   sbatch finalize_cu.sbatch OPT
#SBATCH --exclusive -N1 --mem=0 -t 02:00:00 -J cu-finalize -o logs/%x-%j.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
L=/tmp/r2cufin.$SLURM_JOB_ID; mkdir -p $L; trap 'rm -rf $L' EXIT
export TMPDIR=$L HF_DATASETS_CACHE=$L/hf OMP_NUM_THREADS=1
$REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/coreutils/finalize_cu.py $1
