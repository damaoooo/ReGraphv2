#!/bin/bash
# coreutils step I (method of the 10-31 coreutils irsim, code Scripts/shaheen/irsim unchanged):
#   sbatch irsim_cu.sbatch pairs | sbatch --array=0-6 irsim_cu.sbatch dist | sbatch irsim_cu.sbatch summary
# Universe = functions with text at the base levels O0 and Oc (as 10-31; the 12 mips32 clang-crash modules have no
# O1/O2/O3/Os/Oz .bc); the other 5 levels are extra levels. 1M pairs per kind, kinds XA,XO,XC,XM.
#SBATCH --exclusive -N1 --cpus-per-task=384 --mem=0 -t 06:00:00 -J irsim-cu -o logs/%x-%j-%a.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
export OMP_NUM_THREADS=1 IRSIM_TEXT=$REGRAPH_DATA/irsim/coreutils/text IRSIM_FLOW=$REGRAPH_DATA/Coreutils-flowchart IRSIM_KINDS=XA,XO,XC,XM IRSIM_ARCH_VIEW=1
P=$REGRAPH_DATA/irsim/coreutils/pairs LEVELS=(O0 Oc O3 O1 O2 Os Oz)
case $1 in
pairs)   ${IRSIM_PY:?set IRSIM_PY to the python of the irsim venv} $REGRAPH_REPO/Scripts/shaheen/irsim/build_pairs.py --out $P --no-split-level --kinds XA,XO,XC,XM --base-levels O0,Oc --extra-levels O1,O2,O3,Os,Oz --n-hash 1000000 --n-dist 1000000 --workers 128 ;;
dist)    ${IRSIM_PY:?set IRSIM_PY to the python of the irsim venv} $REGRAPH_REPO/Scripts/shaheen/irsim/compute_dist.py --pairs-dir $P --level ${LEVELS[$SLURM_ARRAY_TASK_ID]} --workers 384 ;;
summary) ${IRSIM_PY:?set IRSIM_PY to the python of the irsim venv} $REGRAPH_REPO/Scripts/shaheen/irsim/summarize.py --pairs-dir $P --levels O0,Oc,O3,O1,O2,Os,Oz ;;
esac
