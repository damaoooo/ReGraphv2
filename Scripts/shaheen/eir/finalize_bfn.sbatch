#!/bin/bash
# bodyfirst noisa fused: finalize (test_final_set, o0_row_map, dropped) + gate numbers vs O0 + noisa checks
#SBATCH --exclusive -N1 --mem=0 -t 02:00:00 -J s11-bfn-finalize -o logs/%x-%j.out
set -euo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
L=/tmp/r2s11bfnfin.$SLURM_JOB_ID; mkdir -p $L; trap 'rm -rf $L' EXIT
export TMPDIR=$L HF_DATASETS_CACHE=$L/hf OMP_NUM_THREADS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
F=$REGRAPH_DATA/Coreutils-EIR-bodyfirst-noisa-fused; PY=$REGRAPH_REPO/.venv/bin/python; W=$REGRAPH_DATA/S11A
$PY $REGRAPH_REPO/Scripts/shaheen/eir/finalize_eir.py $F $W/out_bfn/status
$PY $REGRAPH_REPO/Scripts/shaheen/eir/gate_bf.py $F
$PY $REGRAPH_REPO/Scripts/shaheen/eir/gate_bfn.py $F $REGRAPH_DATA/Coreutils-EIR-bodyfirst-fused $W/out_bfn/status
