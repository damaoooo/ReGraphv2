# Ray Pipelines on Shaheen

## 新版 fused `.ll -> final_set` 流程

新流程入口是：

```text
Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch
Scripts/ray_opt_ablation/ray_fused_pipeline.py
```

它替代旧的 `function_map.csv + results.db + wash` 链路，执行：

```text
.ll -> Task2 .bc -> fused Task3 parquet -> HuggingFace dataset -> final_set
```

### 提交 fused 流程

正式跑一个 opt level：

```bash
cd /scratch/zhoul0e/ReGraphv2
DATASET_PATH=/scratch/zhoul0e/Dataset-1-lift/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O3-fused \
FORCE_CLEAN=1 \
sbatch Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O3
```

冒烟测试：

```bash
cd /scratch/zhoul0e/ReGraphv2
SMOKE=1 FORCE_CLEAN=1 \
sbatch --time=00:30:00 Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O3
```

续跑：

```bash
cd /scratch/zhoul0e/ReGraphv2
DATASET_PATH=/scratch/zhoul0e/Dataset-1-lift/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O3-fused \
RESUME=1 \
sbatch Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O3
```

覆盖节点数：

```bash
cd /scratch/zhoul0e/ReGraphv2
DATASET_PATH=/scratch/zhoul0e/Dataset-1-lift/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O3-fused \
FORCE_CLEAN=1 \
sbatch --nodes=3 Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O3
```

传递 driver 参数：

```bash
cd /scratch/zhoul0e/ReGraphv2
DRIVER_EXTRA_ARGS="--task2-chunk-size 100 --task3-chunk-size 200 --max-parquet-files 5000 --command-timeout-seconds 600" \
DATASET_PATH=/scratch/zhoul0e/Dataset-1-lift/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O3-fused \
sbatch Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O3
```

### Fused 输出

所有输出写入 `OUTPUT_PATH`：

- `bc/`：Task2 reoptimized `.bc`，按输入 split/相对路径镜像。
- `task3_fused/parquet/`：最终 parquet，通常包含 `train/`、`validation/`、`test/`。
- `task3_fused/manifests/`：fused Task3 的 `.bc` 级 success/failed/no-function manifest。
- `hf/`：由 parquet 保存出的 HuggingFace dataset，例如 `train_dataset`。
- `train_final_set`、`validation_final_set`、`test_final_set`：最终训练/验证任务数据。
- `logs/run.log`、`logs/events.jsonl`、`logs/stage_failures/*.txt`。
- `manifests/task2_{success,failed,skipped}.jsonl`。

成功标准：

1. `sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed,NNodes,AllocCPUS -P` 显示主作业 `COMPLETED|0:0`。
2. `OUTPUT_PATH/logs/run.log` 里出现 `pipeline completed successfully`。
3. `ray_cluster_cpus` 等于 `节点数 * 384`，例如 2 节点是 `768`。
4. 需要的 split 下存在 `<split>_final_set/train_dataset_pool`。
5. `squeue -u $USER` 没有遗留测试作业。

### Fused 流程与 Shaheen Ray 标准

`slurm_ray_fused_pipeline.sbatch` 继承当前已验证的 Shaheen 启动方式：

- 不使用 `ray job submit`；在 head node 上直接执行 driver。
- Ray head/worker 都通过 `srun --overlap` 启动。
- 每个 Ray 节点显式设置 control ports 和 worker port range。
- 每个 worker node 使用独立端口段。
- cache 默认写到 `/tmp/regraph_<slurm-job-id>`，包含 HuggingFace/datasets、Ray temp、XDG cache。
- cleanup 先 `ray stop --force`，再终止后台 Ray `srun` step。
- driver 通过绝对路径提交，不使用 Ray `--working-dir` 打包 repo/dataset。

`Scripts/task3_extract.py --backend ray` 只是一个 Ray driver，假设 Ray 集群已经由 sbatch 按上述标准启动；不要单独把它当成 Shaheen launcher。

## Legacy Ray Opt Ablation

该目录包含 `Scripts/opt_ablation.sh` 的 Ray 多节点版本。

注意：下面的 `slurm_ray_opt_ablation.sbatch` / `ray_opt_ablation.py` 是旧流程，仍描述 `function_map.csv`、`results.db`、raw/wash/final 数据集链路。新 fused 需求应使用上一节的 `slurm_ray_fused_pipeline.sbatch`。

## 提交任务

```bash
cd /scratch/zhoul0e/ReGraphv2
sbatch Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

冒烟测试：

```bash
cd /scratch/zhoul0e/ReGraphv2
SMOKE=1 FORCE_CLEAN=1 sbatch Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

续跑：

```bash
cd /scratch/zhoul0e/ReGraphv2
RESUME=1 sbatch Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

覆盖路径：

```bash
cd /scratch/zhoul0e/ReGraphv2
DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O0 \
sbatch Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

传递 driver 参数：

```bash
cd /scratch/zhoul0e/ReGraphv2
DRIVER_EXTRA_ARGS="--progress-summary-interval-s 30 --graph-chunk-size 100" \
sbatch Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

默认 cache 路径：

```bash
/tmp/regraph_<slurm-job-id>
```

如需覆盖：

```bash
cd /scratch/zhoul0e/ReGraphv2
CACHE_ROOT=/scratch/zhoul0e/my_regraph_cache \
sbatch Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

## 输出

源数据集为只读。所有输出都写入输出根目录：

- `train/`、`validation/`、`test/`：镜像源数据的 split 目录树，包含 `.bc`、`*_functions` 和 `results.db`。
- `train_raw_dataset`, `validation_raw_dataset`, `test_raw_dataset`.
- `train_wash_dataset`, `validation_wash_dataset`, `test_wash_dataset`.
- `train_final_set`, `validation_final_set`, `test_final_set`.
- `logs/run.log`, `logs/events.jsonl`, `logs/stage_failures/*.txt`.
- `manifests/*_{success,failed,skipped}.jsonl`.
- Slurm 的 stdout/stderr 默认写入 `Scripts/ray_opt_ablation/slurm_logs/regraph_ray_opt-<jobid>.out` 和 `.err`。
- HuggingFace/datasets cache 会写入 `CACHE_ROOT` 下的 `huggingface/`，不会使用 `/home/zhoul0e/.cache/huggingface`。

## 共享文件系统

launcher 假定 `/scratch/zhoul0e` 在所有分配节点上共享，并被 bind 到 Singularity 容器里。driver 通过绝对路径提交，不使用 Ray `--working-dir`，因此 Ray 不会在启动前打包或上传 repo/dataset。

## 扩展与伸缩

任务以跨全部 split 的全局文件队列方式调度。若要扩展到更多节点，可修改 `slurm_ray_opt_ablation.sbatch` 中的 `#SBATCH --nodes`、`#SBATCH --cpus-per-task` 和 `#SBATCH --time`。

默认使用自适应 chunk 大小。仅在必要时通过 `ray_opt_ablation.py` 的 driver 参数进行覆盖。

Shaheen 版本默认请求 `--exclusive`、`--cpus-per-task=384` 和 `--mem=0`，让每个分配节点的 CPU 和内存都交给 Ray 使用。

## Shaheen 成功运行记录（2026-04-26）

这次在 KAUST Shaheen 上跑通的关键点如下。后续如果换一个 LLM 接手，优先按这一节判断，不要从通用 Ray 集群教程重新猜。

### 已验证的提交方式

当前 sbatch 脚本默认适配 Shaheen：`workq`、独占节点、每节点 384 CPU、Singularity 镜像 `/scratch/zhoul0e/regraph-data-env-llvm18.1.3.sif`。

冒烟测试命令：

```bash
cd /scratch/zhoul0e/ReGraphv2
SMOKE=1 FORCE_CLEAN=1 sbatch --time=00:30:00 Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

正式跑默认数据集：

```bash
cd /scratch/zhoul0e/ReGraphv2
FORCE_CLEAN=1 sbatch Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

如果要临时改成 3 节点，可以让 sbatch 命令行覆盖脚本里的 `#SBATCH --nodes=2`：

```bash
cd /scratch/zhoul0e/ReGraphv2
FORCE_CLEAN=1 sbatch --nodes=3 Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch O0
```

### 为什么这个脚本能在 Shaheen 上跑

- 不使用 `ray job submit`。Shaheen 上 Ray Jobs API 曾出现 504/404 或长时间无响应；现在是在 head node 上直接执行 driver：`RAY_ADDRESS=<head-ip>:<port> python3 ray_opt_ablation.py ...`。
- Ray head/worker 都通过 `srun --overlap` 启动。因为 `ray start --block` 会长期占住 step，后续 `ray status`、driver 和 cleanup 仍然需要新的 `srun` step。
- 每个 Ray 节点显式设置 control ports 和 worker port range。Shaheen 每节点 384 CPU，`WORKER_PORT_SPAN` 默认是 `900`，否则 Ray worker 数量可能超过可用 worker port，导致节点注册或 worker 启动失败。
- 每个 worker node 用独立端口段。脚本按 Slurm job id 计算 `base_port`，head 使用 `base_port+100` 开始的 worker range，worker 节点使用 `base_port+1000`、`base_port+2000` 等独立 range，避免多节点端口冲突。
- cache 默认放在 `/tmp/regraph_<jobid>`。HuggingFace/datasets cache、Ray temp 和 XDG cache 都跟着 job 隔离，避免写 login home 或复用坏 cache。
- cleanup 会先 `ray stop --force`，再等待/终止后台 Ray `srun` step。日志里 `[ray-step] ... exited after cleanup with status=1` 是 Ray 被 stop 后的正常现象，不代表 job 失败。

### 本次成功证据

2 节点 smoke job `11689167` 已在 Shaheen 上完成：

```text
sacct: 11689167|regraph_ray_opt|COMPLETED|0:0|00:12:40|2|768
ray_cluster_cpus=768
stage=task2 complete success=20 failed=0 skipped=0
stage=task3 complete success=20 failed=0 skipped=0
stage=graph complete success=27095 failed=0 skipped=0
stage=dataprocess complete success=27095 failed=0 skipped=0
stage=wash complete success=3013 failed=0 skipped=0
stage=final complete success=3 failed=0 skipped=0
pipeline completed successfully
```

之前独立 RayTest 也验证过 3 节点 Ray 集群能起来并执行任务：3 节点共 1152 CPU，所有简单 Ray task 均成功返回数字和 IP。

### 成功标准

至少同时满足下面几点，才算真的跑通：

1. `sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed,NNodes,AllocCPUS -P` 显示主作业 `COMPLETED|0:0`。
2. `logs/run.log` 里出现 `pipeline completed successfully`。
3. `task2`、`task3`、`graph`、`dataprocess`、`wash`、`final` 都是 `failed=0`。
4. `ray_cluster_cpus` 等于 `节点数 * 384`，例如 2 节点是 `768`，3 节点是 `1152`。
5. `squeue -u $USER` 没有遗留测试作业。`squeue` 清空后，`sacct` 有时会短暂仍显示 RUNNING，等 10 秒再查通常会更新成 COMPLETED。

### 常用排障命令

```bash
cd /scratch/zhoul0e/ReGraphv2
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed,NNodes,AllocCPUS -P
tail -n 220 /scratch/zhoul0e/Dataset-smoketest-O0/logs/run.log
tail -n 160 Scripts/ray_opt_ablation/slurm_logs/regraph_ray_opt-<jobid>.out
tail -n 260 Scripts/ray_opt_ablation/slurm_logs/regraph_ray_opt-<jobid>.err
```

如果 job 已经确认失败或不再需要，先取消 allocation，避免独占节点浪费：

```bash
scancel <jobid>
squeue -u $USER
```

### 常见误判

- `FutureWarning: Ray will no longer override accelerator visible devices env var` 是 Ray 自身提示，不是失败。
- cleanup 后看到 Ray background step `status=1` 可以是正常的，因为 `ray stop --force` 会让 `ray start --block` 退出。
- `ray status` 里 CPU 使用为 `0.0/768.0 CPU` 只表示当前没有 task 正在跑，不表示 Ray 没看到 CPU；关键看分母是否是预期总 CPU。
- `llvm-nm` 找不到 `T` 函数的 `.bc` 不应该让 task3 失败。`split_llvm_ir.sh` 会写 `.no_functions` marker，`ray_opt_ablation.py` 会把 `function_map.csv + .no_functions` 视为成功处理过的样本。

### 修改前先做的轻量检查

```bash
cd /scratch/zhoul0e/ReGraphv2
bash -n Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch
bash -n Scripts/split_llvm_ir.sh
singularity exec --bind /scratch/zhoul0e:/scratch/zhoul0e \
  /scratch/zhoul0e/regraph-data-env-llvm18.1.3.sif \
  python3 -m py_compile Scripts/ray_opt_ablation/ray_opt_ablation.py
git diff --check -- \
  Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch \
  Scripts/ray_opt_ablation/ray_opt_ablation.py \
  Scripts/split_llvm_ir.sh
```
