#!/bin/bash
# coreutils step G: one row per function, untruncated, with token_len (format of the 10-31 coreutils-Oc-rell).
#   sbatch cu_rell.sbatch OPT   -> $REGRAPH_DATA/coreutils-<OPT>-rell
#SBATCH --exclusive -N1 --mem=0 -t 02:00:00 -J cu-rell -o logs/%x-%j.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
$REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/coreutils/build_cu_rell.py $REGRAPH_DATA/coreutils-$1-rell 192 $REGRAPH_DATA/irsim/coreutils/text/$1
