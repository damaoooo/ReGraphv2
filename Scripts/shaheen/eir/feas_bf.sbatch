#!/bin/bash
#SBATCH --exclusive -N1 -t 00:30:00 -J s11-feasbf -o logs/%x-%j.out
set -uo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
ST=/tmp/feas.$SLURM_JOB_ID; mkdir -p $ST; trap 'rm -rf $ST' EXIT
zstd -dc -T0 ${REVNG_TAR:?set REVNG_TAR to the rev.ng c865bc3d1d83 .tar.zst} | tar -C $ST -xf -
tar -C $ST -xf $REGRAPH_DATA/S11/out/ref/ref_ncat.tar
DIS=$(find $ST/revng/root -path '*llvm/bin/llvm-dis' | head -1); echo "dis=$DIS"
singularity exec $SINGULARITY_BINDS $REGRAPH_SIF \
  env TMPDIR=$ST USE_TORCH=0 $REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/eir/feas_bf.py $ST/dataset1__nmap__x64-gcc-7-O2_ncat.isolate.tar $DIS
