# Task1 Lift Multi-node on Shaheen

This directory contains a separate multi-node runner for `task1_lift`:

- `task1_lift_multinode.py`: runs one deterministic shard of the input tree.
- `submit_task1_lift_multinode.sh`: submits a Slurm array, one array task per node.

Shaheen `/scratch/zhoul0e` is shared across nodes, so there is no sync step. Every shard writes directly to the same output tree:

```text
/scratch/zhoul0e/Dataset-1-lift/Dataset-1/<relative-input-path>.ll
```

## Submit

Run from `/scratch/zhoul0e/ReGraphv2`:

```bash
Scripts/ray_opt_ablation/submit_task1_lift_multinode.sh 2
```

The argument is the number of shards/nodes. For example, `4` submits array tasks `0-3`, each with one exclusive node and `384` workers.

Useful overrides:

```bash
SHARDS=4 TIMEOUT_SECONDS=300 WORKERS=384 Scripts/ray_opt_ablation/submit_task1_lift_multinode.sh
INPUT=/scratch/zhoul0e/Dataset-1 OUTPUT=/scratch/zhoul0e/Dataset-1-lift Scripts/ray_opt_ablation/submit_task1_lift_multinode.sh 4
```

The submit script refuses to overlap with an existing `regraph_task1_*` job unless `FORCE=1` is set. This avoids two jobs writing the same missing `.ll` at the same time.

## Monitor

```bash
squeue -u $USER -o '%.12i %.24j %.2t %.10M %.6D %R'
find /scratch/zhoul0e/Dataset-1-lift/Dataset-1 -type f -name '*.ll' | wc -l
tail -f Scripts/ray_opt_ablation/slurm_logs/task1_multinode/task1_mn-<array_jobid>_0.out
```

Per-shard detailed failure logs are written next to the Slurm logs:

```text
Scripts/ray_opt_ablation/slurm_logs/task1_multinode/task1_multinode_<job>_<task>.log
Scripts/ray_opt_ablation/slurm_logs/task1_multinode/task1_multinode_failed_<job>_<task>.txt
```

## Notes

The runner partitions files by sorted relative path: `index % num_shards == shard_index`. This makes shards deterministic and non-overlapping.

Some binaries make IDA/idalib crash with signal 11 even when run alone. The runner records those failures and continues, because otherwise one bad binary wastes the whole node allocation. Existing non-empty `.ll` files are skipped with `--resume`.
