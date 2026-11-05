#!/bin/bash
# coreutils-EIR-rell rule set (b): build_eir_rell_b.py on the eir_text_b.py texts
#SBATCH --exclusive -N1 --mem=0 -t 02:00:00 -J s11-eir-rell-b -o logs/%x-%j.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
W=$REGRAPH_DATA/S11A
$REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/eir/build_eir_rell_b.py $REGRAPH_DATA/coreutils-EIR-rell-b 48 $W/out_tb/text $W/out_tb/status
