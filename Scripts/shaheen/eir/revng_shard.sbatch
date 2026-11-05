#!/bin/bash
# S11: rev.ng isolate lift (EIR) for one list of binaries (one per line: <input path>\t<name>), NPAR single-thread
# processes per node; rev.ng c865bc3d1d83 unpacked to node-local /tmp. Per binary: wall, max RSS, rc, output size;
# outputs packed into OUT/<list>.tar.zst (the .isolate.tar files, zstd -3) + OUT/<list>.status.tsv + OUT/<list>.fail_logs/.
#   sbatch [--array=..] revng_shard.sbatch LIST_OR_DIR OUT NPAR   (with --array: LIST_OR_DIR/<idx>.tsv)
#SBATCH --exclusive -N1 -t 24:00:00 -J s11-revng -o logs/%x-%A_%a.out
set -uo pipefail
L=$1; [ -d "$L" ] && L=$L/$(printf %03d ${SLURM_ARRAY_TASK_ID:-0}).tsv
OUT=$2 NPAR=${3:-1}; S=$(basename $L .tsv)
ST=/tmp/r2s11.$SLURM_JOB_ID.${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $ST/out $ST/logs; trap 'rm -rf $ST' EXIT
zstd -dc -T0 ${REVNG_TAR:?set REVNG_TAR to the rev.ng c865bc3d1d83 .tar.zst} | tar -C $ST -xf -
export REVNG=$ST/revng/revng ST TO=${TO:-82000} STRIP=${STRIP:-0} OBJCOPY=$ST/revng/root/lib64/llvm/llvm/bin/llvm-objcopy
echo "node=$(hostname) list=$L n=$(wc -l < $L) npar=$NPAR start=$(date '+%F %T')"
one() {
  b=$1 n=$2; s=$(date +%s)
  if [ "$STRIP" = 1 ]; then $OBJCOPY --strip-debug "$b" $ST/$n.nodebug && b=$ST/$n.nodebug; fi
  timeout $TO /usr/bin/time -v $REVNG quick artifact isolate --analyses parse-binary,detect-abi "$b" -o $ST/out/$n.isolate.tar --tar > $ST/logs/$n.log 2>&1
  rc=$?; w=$(( $(date +%s) - s )); m=$(grep -m1 'Maximum resident' $ST/logs/$n.log | awk '{print $6}')
  sz=$(stat -c %s $ST/out/$n.isolate.tar 2>/dev/null || echo 0)
  printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$rc" "$w" "${m:-NA}" "$sz" >> $ST/status.tsv
  echo "$n rc=$rc wall=${w}s maxrss_kb=${m:-NA} out=$sz"
}
export -f one
tr '\t' '\n' < $L | xargs -d '\n' -n 2 -P $NPAR bash -c 'one "$0" "$1"'
mkdir -p $OUT/$S.fail_logs
awk -F'\t' '$2!=0{print $1}' $ST/status.tsv | while read n; do grep -v -E "^\s+(Command being timed|User time|System time|Percent of CPU|Elapsed|Average|Maximum resident|Major|Minor|Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Exit status)" $ST/logs/$n.log | head -12 > $OUT/$S.fail_logs/$n.head.log; tail -n 30 $ST/logs/$n.log > $OUT/$S.fail_logs/$n.tail30.log; rm -f $ST/out/$n.isolate.tar; done
rmdir $OUT/$S.fail_logs 2>/dev/null || true
(echo -e "name\trc\twall_s\tmaxrss_kb\tout_bytes"; sort $ST/status.tsv) > $OUT/$S.status.tsv
tar -C $ST/out -cf - . | zstd -q -T0 -3 -o $OUT/$S.tar.zst.part && mv $OUT/$S.tar.zst.part $OUT/$S.tar.zst
echo "end=$(date '+%F %T') ok=$(awk -F'\t' 'NR>1&&$2==0' $OUT/$S.status.tsv | wc -l)/$(wc -l < $L)"
