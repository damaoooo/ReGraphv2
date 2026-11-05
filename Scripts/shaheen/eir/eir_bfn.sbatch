#!/bin/bash
# S11 part B bodyfirst noisa fused for one lift shard (eir_bf.py) in the LLVM 18 SIF; rev.ng's llvm-dis from node-local /tmp.
#   sbatch eir_bf.sbatch <lift shard .tar.zst> [workers]
#SBATCH --exclusive -N1 -t 12:00:00 -J s11-bfn -o logs/%x-%j.out
set -uo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
IN=$1; TAG=$(basename $(dirname $IN))_$(basename $IN .tar.zst); W=$REGRAPH_DATA/S11A
ST=/tmp/r2s11bf.$SLURM_JOB_ID; mkdir -p $ST; trap 'rm -rf $ST' EXIT
zstd -dc -T0 ${REVNG_TAR:?set REVNG_TAR to the rev.ng c865bc3d1d83 .tar.zst} | tar -C $ST -xf - revng/root/lib64 revng/root/lib 2>/dev/null
zstd -dc -T0 $IN > $ST/shard.tar
echo "node=$(hostname) shard=$IN tag=$TAG"
singularity exec $SINGULARITY_BINDS $REGRAPH_SIF \
  env TMPDIR=$ST USE_TORCH=0 OMP_NUM_THREADS=1 $REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/eir/eir_bf.py $ST/shard.tar $TAG \
  $REGRAPH_DATA/Coreutils-EIR-bodyfirst-noisa-fused $W/out_bfn/status $W/work/ida_names.json \
  $REGRAPH_DATA/S11/o0_index/coreutils_test.parquet $ST/revng/root/lib64/llvm/llvm/bin/llvm-dis --workers ${2:-96} --tmp $ST --noisa
