# Ray Opt Ablation 需求

## 目标

实现一个基于 Ray 的 `Scripts/opt_ablation.sh` 多节点版本，用于 Slurm + Singularity。流水线必须通过“全局文件队列”方式扩展到多节点，而不是按 `train` / `validation` / `test` split 粒度分配任务。

## 环境

- Slurm 资源分配中，每个节点运行一个 Ray 进程。
- Singularity 镜像：`/scratch/zhoul0e/regraph-data-env-llvm18.1.3.sif`。
- 默认 Slurm 资源从 3 节点 x 2 CPU 起步，但实现必须支持通过 sbatch 头部增加节点数和 CPU 数。
- Slurm 作业应为大规模 LLVM 提取与图生成请求足够内存。默认 sbatch 使用 `--mem-per-cpu=16G`，因此在增大 `--cpus-per-task` 时内存会随之扩展。
- Ray 启动模式：
  - `srun --overlap`
  - 在 Singularity 内启动 Ray head/worker
  - `ray job submit`
  - `ray stop --force` 清理
- `/scratch/zhoul0e` 是所有节点共享的文件系统；launcher 不应通过 Ray `--working-dir` 打包/上传 repo 或 Dataset-1，而应直接使用共享路径。
- 代码仓库位于 `/home/zhoul0e/ReGraphv2`，需在容器中 bind `/home/zhoul0e` 以访问仓库。
- Ray driver entrypoint 应固定运行在 head node，避免 Ray Jobs supervisor 被调度到启动后退出的 worker 上。

## 数据与输出

- 源数据集为只读。
- 默认全量数据集：`/scratch/zhoul0e/Dataset-1`。
- 冒烟测试数据集：`/scratch/zhoul0e/Dataset-smoketest`。
- 默认输出：`/scratch/zhoul0e/Dataset-1-${OPT_LEVEL}`。
- 冒烟输出：`/scratch/zhoul0e/Dataset-smoketest-${OPT_LEVEL}`。
- 输出需镜像源数据 split 结构：
  - `train/...`
  - `validation/...`
  - `test/...`
- 所有 `.bc`、`*_functions`、`results.db`、raw/wash/final 数据集都必须写入输出根目录。

## 调度

- 将所有 split 文件合并为一个全局 Ray 工作队列。
- 保留 split 作为元数据，用于输出路径和报告。
- 保持全局阶段屏障：
  1. Task2 重优化
  2. Task3 函数拆分
  3. GraphBuilder
  4. DataProcess 原始数据集生成
  5. 数据集清洗
  6. 最终 split 生成
- 基于 Ray 集群 CPU 数和文件数量使用自适应 chunk 大小。
- 允许通过 CLI 覆盖 chunk 大小。

## 复用

- 尽可能复用现有单文件处理逻辑：
  - `Scripts/task2_reoptimize.py`
  - `Scripts/task3_extract.py`
  - `GraphBuilder/graph_generator.py`
  - `DataProcess.parallel_processor.process_chunk_standalone`
  - `DataProcess.dataset_wash` 辅助逻辑
  - `Pretrain.split_train_validation`
- 现有代码仅可为“路径可配置化”和“可复用 helper”目的进行调整。

## LLVM Pass 插件

- 在 SIF 运行时环境内编译 GraphBuilder `.so` 插件。
- 不得写入 SIF 文件。
- 不得将构建产物写入共享仓库目录。
- 每个节点仅编译一次，并写入该节点本地临时存储：
  - `$TMPDIR/regraph_ray_plugins_${SLURM_JOB_ID}`
  - 备用路径：`/tmp/regraph_ray_plugins_${SLURM_JOB_ID}`
- 使用锁文件避免同一节点重复构建。

## 失败处理、续跑与日志

- 如果输出已存在，默认行为应为失败退出。
- `--resume` 从已有输出继续执行。
- `--force-clean` 删除输出后重新运行。
- 单文件失败不应中断当前阶段。
- 下一阶段只接收上一阶段成功产物。
- 每个 chunk 任务自动重试一次。
- 若 chunk 仍失败，则拆分成单文件任务并重试。
- 失败信息必须在屏幕可见，并写入：
  - `logs/run.log`
  - `logs/events.jsonl`
  - `logs/stage_failures/*.txt`
- sbatch 必须显式检查 Ray job status；如果 Ray job 为 `FAILED` / `STOPPED` 或 status 查询失败，Slurm 作业必须非零退出。
- 对于长时间待处理的任务，日志必须可诊断，至少包含 chunk id、chunk 挂起时长、chunk 大小，以及该待处理 chunk 的文件路径预览。
- 进度应由 driver 统一管理并展示：
  - 总体进度
  - 当前阶段进度
  - 已完成/总数
  - 百分比
  - 处理速率
  - ETA
  - 已耗时
  - split 级 success/failure/skip 汇总

## 验收标准

- 静态检查：
  - `python3 -m py_compile Scripts/ray_opt_ablation/ray_opt_ablation.py`
  - `bash -n Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch`
  - `sbatch --test-only Scripts/ray_opt_ablation/slurm_ray_opt_ablation.sbatch`
- 冒烟运行：
  - 数据集：`/scratch/zhoul0e/Dataset-smoketest`
  - 先跑一个优化等级，例如 `O0`
  - Slurm 作业以 `COMPLETED 0:0` 结束
  - Ray 能识别所有已分配节点和 CPU
  - 输出包含 `.bc`、`*_functions/function_map.csv`、`results.db`、raw/wash/final 数据集
  - HF 数据集可通过 `datasets.load_from_disk` 加载
