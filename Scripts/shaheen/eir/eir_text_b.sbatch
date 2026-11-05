#!/bin/bash
# S11 part B text rule set (b) for one lift shard (eir_text_b.py) in the LLVM 18 SIF; rev.ng's llvm-dis from node-local /tmp.
#   sbatch eir_text_b.sbatch <lift shard .tar.zst> [workers]
#SBATCH --exclusive -N1 -t 12:00:00 -J s11-textb -o logs/%x-%j.out
set -uo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
IN=$1; TAG=$(basename $(dirname $IN))_$(basename $IN .tar.zst); W=$REGRAPH_DATA/S11A
ST=/tmp/r2s11tb.$SLURM_JOB_ID; mkdir -p $ST; trap 'rm -rf $ST' EXIT
zstd -dc -T0 ${REVNG_TAR:?set REVNG_TAR to the rev.ng c865bc3d1d83 .tar.zst} | tar -C $ST -xf - revng/root/lib64 revng/root/lib 2>/dev/null
zstd -dc -T0 $IN > $ST/shard.tar
echo "node=$(hostname) shard=$IN tag=$TAG"
singularity exec $SINGULARITY_BINDS $REGRAPH_SIF \
  env TMPDIR=$ST USE_TORCH=0 OMP_NUM_THREADS=1 $REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/eir/eir_text_b.py $ST/shard.tar $TAG \
  $W/out_tb/text $W/out_tb/status $W/work/ida_names.json $ST/revng/root/lib64/llvm/llvm/bin/llvm-dis --workers ${2:-96} --tmp $ST
