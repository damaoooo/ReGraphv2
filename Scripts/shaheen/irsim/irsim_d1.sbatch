#!/bin/bash
# Dataset-1 step I (method of the 10-27 Dataset-1 irsim: kinds XA,XO,XC,XM,XV, base levels O0,Oc,O3, 1M pairs per
# kind; code Scripts/shaheen/irsim, which equals the 10-27 code with these defaults plus env-configurable paths):
#   sbatch irsim_d1.sbatch pairs | sbatch --array=0-6 irsim_d1.sbatch dist | sbatch irsim_d1.sbatch summary
#SBATCH --exclusive -N1 --cpus-per-task=384 --mem=0 -t 12:00:00 -J irsim-d1 -o logs/%x-%j-%a.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
export OMP_NUM_THREADS=1 IRSIM_TEXT=$REGRAPH_DATA/irsim/Dataset-1/text IRSIM_FLOW=$REGRAPH_DATA/Dataset-1-flowchart
P=$REGRAPH_DATA/irsim/Dataset-1/pairs LEVELS=(O0 Oc O3 O1 O2 Os Oz)
case $1 in
pairs)   ${IRSIM_PY:?set IRSIM_PY to the python of the irsim venv} $REGRAPH_REPO/Scripts/shaheen/irsim/build_pairs.py --out $P --n-hash 1000000 --n-dist 1000000 --workers 128 ;;
dist)    ${IRSIM_PY:?set IRSIM_PY to the python of the irsim venv} $REGRAPH_REPO/Scripts/shaheen/irsim/compute_dist.py --pairs-dir $P --level ${LEVELS[$SLURM_ARRAY_TASK_ID]} --workers 384 ;;
summary) ${IRSIM_PY:?set IRSIM_PY to the python of the irsim venv} $REGRAPH_REPO/Scripts/shaheen/irsim/summarize.py --pairs-dir $P --levels O0,Oc,O3,O1,O2,Os,Oz ;;
esac
