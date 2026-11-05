#!/bin/bash
#SBATCH --exclusive -N1 --mem=0 -t 02:00:00 -J s11-o0idx -o logs/%x-%j.out
: "${REGRAPH_REPO:?set REGRAPH_REPO (Scripts/shaheen/cluster_env.sh)}" "${REGRAPH_DATA:?set REGRAPH_DATA}"
export HF_DATASETS_CACHE=/tmp/hfc.$SLURM_JOB_ID OMP_NUM_THREADS=8
$REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/eir/o0_index.py
