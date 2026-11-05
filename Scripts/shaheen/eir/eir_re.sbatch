#!/bin/bash
# S11-A step 2: EIR .re for one lift shard (eir_re.py) in the LLVM 18 SIF; rev.ng's llvm-dis from node-local /tmp.
#   sbatch [--array=..] eir_re.sbatch <lift shard .tar.zst | dir of shards> <out dir> [workers]
#   -> <out>/re/<group>_<shard>.tar.zst, <out>/status/<group>_<shard>.jsonl
#SBATCH --exclusive -N1 -t 12:00:00 -J s11a-re -o logs/%x-%A_%a.out
set -uo pipefail
source "${REGRAPH_REPO:?set REGRAPH_REPO}/Scripts/shaheen/cluster_env.sh"  # site settings
IN=$1; [ -d "$IN" ] && IN=$(ls $IN/*.tar.zst | sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p")
OUT=$2; TAG=$(basename $(dirname $IN))_$(basename $IN .tar.zst)
ST=/tmp/r2s11a.$SLURM_JOB_ID.${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $ST/re $OUT/re $OUT/status; trap 'rm -rf $ST' EXIT
zstd -dc -T0 ${REVNG_TAR:?set REVNG_TAR to the rev.ng c865bc3d1d83 .tar.zst} | tar -C $ST -xf - revng/root/lib64 revng/root/lib 2>/dev/null
zstd -dc -T0 $IN > $ST/shard.tar
DIS=$ST/revng/root/lib64/llvm/llvm/bin/llvm-dis
echo "node=$(hostname) shard=$IN tag=$TAG"
singularity exec $SINGULARITY_BINDS $REGRAPH_SIF \
  env TMPDIR=$ST $REGRAPH_REPO/.venv/bin/python $REGRAPH_REPO/Scripts/shaheen/eir/eir_re.py $ST/shard.tar $TAG $ST/re $OUT/status/$TAG.jsonl \
  $REGRAPH_DATA/S11A/work/ida_names.json $DIS --workers ${3:-64} --tmp $ST
tar -C $ST/re -cf - . | zstd -q -T8 -3 -o $OUT/re/$TAG.tar.zst.part && mv $OUT/re/$TAG.tar.zst.part $OUT/re/$TAG.tar.zst
