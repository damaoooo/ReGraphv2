#!/bin/bash
# Pack one opt level's recompiled objects: Binaries/Dataset-1-<OPT>-re/ -> Binaries/Dataset-1-<OPT>-re.tar.zst
# Runs on the host (the SIF has no zstd). Submit after that level's `opt_stage.sbatch recompile` shards.
#   sbatch --dependency=afterok:<recompile job> Scripts/shaheen/re_pack.sbatch O2
#SBATCH --job-name=re-pack
#SBATCH --partition=workq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zhoul0e/iops/projects/ida-batch/logs/%x-%j.out

set -euo pipefail
OPT=${1:?usage: sbatch re_pack.sbatch OPT_LEVEL}
BIN=/scratch/zhoul0e/ReGraphv2/Binaries
DIR=Dataset-1-$OPT-re
n=$(find "$BIN/$DIR" -name '*.re' | wc -l)
echo "$(date '+%F %T') packing $DIR ($n .re files)"
# only the .re files: the per-shard _task4_*.jsonl manifests stay out of the archive
(cd "$BIN" && { echo "$DIR/NOTES.txt"; find "$DIR" -name '*.re' | sort; }) | tar -C "$BIN" -cf - -T - | zstd -q -T0 -10 -o "$BIN/$DIR.tar.zst.part"
mv "$BIN/$DIR.tar.zst.part" "$BIN/$DIR.tar.zst"
echo "$(date '+%F %T') done: $(ls -la "$BIN/$DIR.tar.zst" | awk '{print $5}') bytes, $(zstd -dc "$BIN/$DIR.tar.zst" | tar -tf - | grep -c '\.re$') .re in archive"
