# Ray Opt Ablation

该目录包含 `Scripts/opt_ablation.sh` 的 Ray 多节点版本。

## 提交任务

```bash
sbatch /ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

冒烟测试：

```bash
SMOKE=1 FORCE_CLEAN=1 sbatch /ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

续跑：

```bash
RESUME=1 sbatch /ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

覆盖路径：

```bash
DATASET_PATH=/ibex/tmp/zhoul0e/Dataset-1 \
OUTPUT_PATH=/ibex/tmp/zhoul0e/Dataset-1-O0 \
sbatch /ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

传递 driver 参数：

```bash
DRIVER_EXTRA_ARGS="--progress-summary-interval-s 30 --graph-chunk-size 100" \
sbatch /ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

默认 cache 路径：

```bash
/ibex/tmp/zhoul0e/regraph_cache/<slurm-job-id>
```

如需覆盖：

```bash
CACHE_ROOT=/ibex/tmp/zhoul0e/my_regraph_cache \
sbatch /ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

## 输出

源数据集为只读。所有输出都写入输出根目录：

- `train/`、`validation/`、`test/`：镜像源数据的 split 目录树，包含 `.bc`、`*_functions` 和 `results.db`。
- `train_raw_dataset`, `validation_raw_dataset`, `test_raw_dataset`.
- `train_wash_dataset`, `validation_wash_dataset`, `test_wash_dataset`.
- `train_final_set`, `validation_final_set`, `test_final_set`.
- `logs/run.log`, `logs/events.jsonl`, `logs/stage_failures/*.txt`.
- `manifests/*_{success,failed,skipped}.jsonl`.
- Slurm 的 stdout/stderr 会写入 `Scripts/ray_opt_ablation/slurm_logs/slurm-<jobid>.out` 和 `.err`。
- HuggingFace/datasets cache 会写入 `CACHE_ROOT` 下的 `huggingface/`，不会使用 `/home/zhoul0e/.cache/huggingface`。

## 共享文件系统

launcher 假定 `/ibex/tmp/zhoul0e` 在所有分配节点上共享，并被 bind 到 Singularity 容器里。driver 通过绝对路径提交，不使用 Ray `--working-dir`，因此 Ray 不会在启动前打包或上传 repo/dataset。

## 扩展与伸缩

任务以跨全部 split 的全局文件队列方式调度。若要扩展到更多节点，可修改 `slurm_ray_opt_ablation.sbatch` 中的 `#SBATCH --nodes`、`#SBATCH --cpus-per-task` 和 `#SBATCH --time`。

默认使用自适应 chunk 大小。仅在必要时通过 `ray_opt_ablation.py` 的 driver 参数进行覆盖。

sbatch 文件默认请求 `--mem-per-cpu=8G`，以避免 LLVM 提取与图生成受限于 Slurm 较小的每 CPU 默认内存。对于特别大的二进制可适当增大；若分区策略要求更小请求，可相应调低。
