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

`OPT_LEVEL` 可传 `O0`、`O1`、`O2`、`O3`、`Os`、`Og` 等 clang 风格优化等级。`O0` 的 Task2 只用 `llvm-as` 把 lifted `.ll` assemble 成 `.bc`，不再额外跑一次 `clang -O0`；其他优化等级继续用 `clang -c -emit-llvm` 重新生成 `.bc`。

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
DRIVER_EXTRA_ARGS="--task3-chunk-size 500 --max-parquet-files 100000 --command-timeout-seconds 28800 --final-output-root /scratch/zhoul0e/bandwidth/Dataset-1-O1-fused" \
DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O1-fused \
sbatch --nodes=3 Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O1
```

`task2` 默认 `--task2-chunk-size 1`，也就是每个 `.ll` 文件一个 Ray task。`.ll` 输入队列会按相对路径做稳定 hash shuffle，这样慢文件只拖住自己的 task，不会把同一个目录里的 z3/openssl 文件集中压在少数节点或少数时间段里。只有在 Ray task 数过多、调度开销明显时，才建议手动调大。

`--command-timeout-seconds 28800` 表示单条 `llvm-as` / `clang` / `llvm-nm` / `llvm-extract` / `opt` 命令最多运行 8 小时。设为 `0` 表示不限制单命令超时，但 Slurm 作业本身仍受 `#SBATCH --time` 或提交时 `--time` 限制。

`task2` 失败文件会写入 `manifests/task2_failed.jsonl` 和 `logs/stage_failures/task2.txt`，并在 `logs/run.log` / Slurm stdout 中列出。fused driver 会继续把已成功生成的 `.bc` 交给 task3 和后续阶段；只有 task2 一个可用 `.bc` 都没生成时才会失败退出。

`task3` 的 `--task3-chunk-size` 控制每个 Ray task 串行处理多少个函数。task3 的 `llvm-nm` 函数枚举阶段也会通过 Ray 按 `.bc` 文件并行运行；枚举完成后，会先按 `.bc` 的相对路径做稳定 hash shuffle，再按 `.bc` 文件 round-robin 打散函数，然后切 chunk，避免 z3 这类大程序的函数集中堵在少数 chunk 里。Dataset-1 O1 当前约 1600 万函数，推荐 `--task3-chunk-size 500 --max-parquet-files 100000`：raw parquet 峰值约 3.3 万个，chunk manifest 默认按 worker 聚合并在成功 compact 后删除。不要再用 `--task3-chunk-size 20/50` 跑完整 Dataset-1，文件数峰值太高。

task3 默认使用 `--chunk-manifest-mode worker`，即每个 Ray worker 进程追加自己的 success/failed manifest，而不是每个 chunk 单独生成两个 manifest 文件。成功 compact 后默认执行 `--cleanup-chunk-manifests` 删除 chunk 级 manifest；如果要保留调试文件，可在 driver 上加 `--keep-task3-chunk-manifests`。

task3 的 `llvm-extract` / `opt` 中间 IR、dot 文件默认写到节点本地 `$TMPDIR/task3_<output-name>`，不写到 scratch 的 `task3_fused/.task3_fused_state/tmp`。这是为了避免函数级临时文件触发 scratch 文件数配额；每个 chunk 结束后会清理自己的临时目录。

如果只想生成 Dataset-1 CSV 清单里的函数，在 driver 参数里加 `--task3-csv-filter-dir <csv-dir>`。该目录需要包含 `training_Dataset-1.csv`、`validation_Dataset-1.csv`、`testing_Dataset-1.csv`。过滤发生在 `llvm-nm` 枚举之后、`llvm-extract`/建图之前，所以 CSV 外的函数不会进入 graph/tokenizer 阶段。例如：

```bash
DRIVER_EXTRA_ARGS="--task3-csv-filter-dir /scratch/zhoul0e/ReGraphv2/IR/csv_list --task3-chunk-size 500 --max-parquet-files 100000 --command-timeout-seconds 28800" \
DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O1-fused-csv \
sbatch --nodes=3 Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O1
```

如果给已有输出目录加 CSV 过滤，必须使用新的 `OUTPUT_PATH`，或者在 `DRIVER_EXTRA_ARGS` 里加 `--force-task3-rebuild`。否则旧的未过滤 parquet 会污染过滤结果；driver 会检测 `task3_fused/manifests/csv_filter_summary.json`，不匹配时直接退出。

`RESUME=1` 时，如果 `task3_fused/parquet/{train,validation,test}` 都已经存在且完整，driver 会跳过 Task3，日志里会出现 `stage=task3_fused skipped existing final parquet splits=...`。如果需要强制重建 Task3，必须在 `DRIVER_EXTRA_ARGS` 加 `--force-task3-rebuild`；这会删除旧的 `task3_fused` 后重新跑 Task3。若 Task3 没有重跑且 `hf/{split}_dataset` 已完整，`dataprocess_hf` 也会跳过。final 阶段按 `<split>_final_set/train_dataset_pool` 是否完整判断是否跳过。

Task3 函数级坏样本不会阻断整个 pipeline。失败函数会写入 worker failed manifest，例如 `task3_fused/.task3_fused_state/chunk_manifests/*_failed.jsonl`，后续 parquet/HF/final 会继续使用已成功的函数记录。常见失败包括个别函数的 Graphviz DOT 解析错误，例如 `pygraphviz.agraph.DotError: Invalid Input`。

### final_set 写入 bandwidth 与本地 staging

如果设置 `--final-output-root /scratch/zhoul0e/bandwidth/<output-name>`，最终的 `train_final_set`、`validation_final_set`、`test_final_set` 会真实写入 bandwidth 文件系统；`OUTPUT_PATH` 下保留同名 symlink，训练和评估脚本仍然可以用 `OUTPUT_PATH/<split>_final_set/...` 路径。

final 阶段会先把对应的 HF dataset 从 scratch 复制到节点本地 `$TMPDIR/final_sets/<output-name>/_hf_inputs/<split>_dataset`，再把 `Pretrain.split_train_validation` 的输出写到 `$TMPDIR/final_sets/<output-name>/<split>_final_set`，成功后复制到 `--final-output-root`，最后删除本地 input 和 output。除非传 `--keep-final-local`，本地 `/tmp` 不会同时保留已经完成的 split。这个设计是为了避免 HuggingFace `save_to_disk` 从 scratch 读写时掉到几十 examples/s。

相关参数和环境变量：

- `--final-output-root`：final_set 的真实落盘根目录；默认是 `OUTPUT_PATH`，也可用 `REGRAPH_FINAL_OUTPUT_ROOT` 覆盖。
- `--final-local-root`：final 阶段本地工作目录；默认在设置了独立 `--final-output-root` 时使用 `$TMPDIR/final_sets/<output-name>`，也可用 `REGRAPH_FINAL_LOCAL_ROOT` 覆盖。
- `--keep-final-local`：调试用，保留本地 final input/output；正常 Dataset-1 不建议打开。
- `--final-filter-reference` / `FINAL_FILTER_REFERENCE`：final_set 生成完成后，用 CSV 文件/目录、单个 final_set、`train_dataset_pool`，或包含 `train_final_set` / `validation_final_set` / `test_final_set` 的 root 作为 whitelist，in-place 过滤 `validation_final_set` 和 `test_final_set`；`train_final_set` 不过滤，过滤前 final_set 不会保留为额外输出。旧的 `--final-csv-filter-dir` / `FINAL_CSV_FILTER_DIR` 仍作为兼容别名。
- `--final-filter-reference-kind` / `FINAL_FILTER_REFERENCE_KIND`：`auto`、`csv` 或 `final-set`，默认 `auto`。
- `--final-filter-match-mode` / `FINAL_FILTER_MATCH_MODE`：`exact` 或 `origin`，默认 `exact`。如果 reference final_set 与目标 final_set 来自不同 opt level，通常用 `origin`。

final 的 `train`、`validation`、`test` 三个 split 会作为独立 Ray task 并行处理。driver 会把这些 task 轮流 pin 到 Ray 节点资源上；申请 3 个节点时，三个 split 会分别跑在 3 个节点的本地 `$TMPDIR` staging 目录上。若只申请 2 个节点，则第三个 split 会轮转复用其中一个节点。

用 CSV 过滤 validation/test final_set 的提交示例：

```bash
cd /scratch/zhoul0e/ReGraphv2
FINAL_FILTER_REFERENCE=/scratch/zhoul0e/ReGraphv2/IR/csv_list \
DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O3-fused \
DRIVER_EXTRA_ARGS="--task3-chunk-size 500 --max-parquet-files 100000 --command-timeout-seconds 28800 --final-output-root /scratch/zhoul0e/bandwidth/Dataset-1-O3-fused" \
sbatch --nodes=3 Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O3
```

用另一个 final_set root 过滤 validation/test：

```bash
FINAL_FILTER_REFERENCE=/scratch/zhoul0e/Dataset-1-O0-fused \
FINAL_FILTER_REFERENCE_KIND=final-set \
FINAL_FILTER_MATCH_MODE=origin \
DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O3-fused \
sbatch --nodes=3 Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O3
```

Dataset-1 的 final_set 使用约定：每个外层 split 目录都用内部 `train_*` 文件作为该 split 的有效数据，因为 driver 调 `Pretrain.split_train_validation` 时固定 `--train-ratio 1.0`。例如 evaluation validation 应传：

```text
/scratch/zhoul0e/Dataset-1-O1-fused/validation_final_set/train_dataset_pool
/scratch/zhoul0e/Dataset-1-O1-fused/validation_final_set/train_positive_map.pkl
```

不要传 `validation_final_set/validation_dataset_pool`；该内部目录是空副产物。

### Dataset-1 O0/O1/O2 已验证提交命令

O0 已经有完整 Task3 parquet 时，直接 resume：

```bash
cd /scratch/zhoul0e/ReGraphv2
DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O0-fused \
RESUME=1 \
DRIVER_EXTRA_ARGS="--task3-chunk-size 500 --max-parquet-files 100000 --command-timeout-seconds 28800 --final-output-root /scratch/zhoul0e/bandwidth/Dataset-1-O0-fused" \
sbatch --nodes=3 --exclude=nid00018,nid00025,nid00026 --job-name=regraph_O0_fused \
  Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O0
```

O1/O2 如果需要重建 Task3，使用 `--force-task3-rebuild`：

```bash
cd /scratch/zhoul0e/ReGraphv2
DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O1-fused \
RESUME=1 \
DRIVER_EXTRA_ARGS="--task3-chunk-size 500 --max-parquet-files 100000 --command-timeout-seconds 28800 --final-output-root /scratch/zhoul0e/bandwidth/Dataset-1-O1-fused --force-task3-rebuild" \
sbatch --nodes=3 --job-name=regraph_O1_fused \
  Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O1

DATASET_PATH=/scratch/zhoul0e/Dataset-1 \
OUTPUT_PATH=/scratch/zhoul0e/Dataset-1-O2-fused \
RESUME=1 \
DRIVER_EXTRA_ARGS="--task3-chunk-size 500 --max-parquet-files 100000 --command-timeout-seconds 28800 --final-output-root /scratch/zhoul0e/bandwidth/Dataset-1-O2-fused --force-task3-rebuild" \
sbatch --nodes=3 --job-name=regraph_O2_fused \
  Scripts/ray_opt_ablation/slurm_ray_fused_pipeline.sbatch O2
```

这三条在 2026-04-28 已完成：O0 job `11754418`、O1 job `11754415`、O2 job `11754416` 均为 `COMPLETED|0:0`，并在 run log 中出现 `pipeline completed successfully`。

### 2026-04-28 Dataset-1 成功经验复盘

这一节给后续接手的 LLM/工程师看。不要把下面这些现象当成新问题重新排查，除非日志和这里描述的不一致。

本次最终跑通的是 Dataset-1 的 O0/O1/O2 fused pipeline，均使用 3 个 Shaheen 独占节点，即 `1152` Ray CPU。最终状态：

```text
11754418 regraph_O0_fused COMPLETED 0:0 00:19:40
11754415 regraph_O1_fused COMPLETED 0:0 02:25:07
11754416 regraph_O2_fused COMPLETED 0:0 02:20:10
```

最终数据位置：

```text
/scratch/zhoul0e/Dataset-1-O0-fused/*_final_set -> /scratch/zhoul0e/bandwidth/Dataset-1-O0-fused/*_final_set
/scratch/zhoul0e/Dataset-1-O1-fused/*_final_set -> /scratch/zhoul0e/bandwidth/Dataset-1-O1-fused/*_final_set
/scratch/zhoul0e/Dataset-1-O2-fused/*_final_set -> /scratch/zhoul0e/bandwidth/Dataset-1-O2-fused/*_final_set
```

#### 成功原因

- Shaheen 单节点 384 CPU，但只支持独占节点；无论申请多少核心都会拿到整个节点。因此多节点跑 Ray 时按 `nodes * 384` 理解资源，不要纠结 `cpus-per-task` 之外的核心数。
- Task2 默认 `chunk_size=1` 是正确选择。慢 `.ll` 文件只拖住一个 Ray task，不会让大 chunk 堵住单核。
- Task3 不要用 `chunk_size=20/50` 跑完整 Dataset-1。函数量约 1600 万，chunk 太小会制造几十万到八十万级 raw parquet/manifest 文件，逼近或超过文件数 quota。当前成功配置是 `--task3-chunk-size 500 --max-parquet-files 100000`，raw parquet 峰值约 3.2 万。
- Task3 的 `.bc` 和函数列表已经做稳定 hash shuffle，并按 `.bc` round-robin 打散。这样 z3/openssl 等大程序不会集中在一个 chunk 或一个核心上。
- Task3 函数级失败可以继续跑。O2 里出现过少量 `pygraphviz.agraph.DotError: Invalid Input`，失败记录写入 `task3_fused/.task3_fused_state/chunk_manifests/*_failed.jsonl`，pipeline 仍然成功。
- final 阶段真正的 I/O 瓶颈不是只写 output；`save_to_disk` 也会读 HF input。如果 output 写 `/tmp` 但 input 仍从 scratch 读，速度会掉到几十 examples/s。成功方案是 input 和 output 都先放节点本地 `/tmp`，完成后再复制到 bandwidth。
- bandwidth 适合放最终大文件。O0 的 train final set 约 21GB，从本地 `/tmp` 复制到 bandwidth 只用了约 11 秒；test HF input 约 42.6GB，从 scratch 复制到 `/tmp` 约 20 秒。
- final 阶段现在按 split 并行提交到 Ray；申请 3 个节点时，`train/validation/test` 会分别占用一个节点做本地 staging 和 `split_train_validation`。

#### 曾经踩过的坑

- O0 第一次 Ray head 曾在 `nid00018,nid00025,nid00026` 上遇到 GCS/port 启动失败。成功重提时使用了 `--exclude=nid00018,nid00025,nid00026`。
- 中间有一次 resume 曾在 Task3 raw shards 被清理后误把剩余少量 `.bc` 重新 compact，导致 O1/O2 的 train parquet 被小数据覆盖。现在 driver 已加保护：`RESUME=1` 且完整 final parquet 存在时跳过 Task3；需要重建必须显式传 `--force-task3-rebuild`。
- 不要用 host `/usr/bin/python3` 去检查 `ray_fused_pipeline.py` 语法。登录环境的 Python 太旧，会报 `future feature annotations is not defined`。用 SIF：

```bash
module load singularity >/dev/null 2>&1
singularity exec --bind /scratch/zhoul0e:/scratch/zhoul0e \
  /scratch/zhoul0e/regraph-data-env-llvm18.1.3.sif \
  python3 -m py_compile /scratch/zhoul0e/ReGraphv2/Scripts/ray_opt_ablation/ray_fused_pipeline.py
```

- 这个 SIF 有 `datasets`，可以检查 final_set 格式；但没有 `torch`，不能在里面完整跑训练 collator。若要跑训练 smoke test，需要使用训练实际环境或含 PyTorch 的 SIF。

#### 接手时优先看的日志

检查 Slurm 状态：

```bash
sacct -j <jobid> --format=JobID,JobName%24,State,ExitCode,Elapsed,NodeList%40 -P
```

检查 pipeline 是否真正完成：

```bash
grep -E "ray_cluster_cpus|stage=task3_fused skipped|stage=dataprocess_hf skipped|final_set_link|pipeline completed successfully" \
  /scratch/zhoul0e/Dataset-1-O1-fused/logs/run.log
```

检查 Task3 进度：

```bash
grep "Task3 Ray progress" /scratch/zhoul0e/Dataset-1-O1-fused/logs/run.log | tail -10
```

检查 final_set symlink：

```bash
find /scratch/zhoul0e/Dataset-1-O1-fused -maxdepth 1 -type l -name '*final_set' -printf '%p -> %l\n' | sort
```

检查失败函数：

```bash
find /scratch/zhoul0e/Dataset-1-O1-fused/task3_fused -type f -name '*failed.jsonl' -printf '%p %s\n' 2>/dev/null | sort
```

#### final_set 格式验证

本次用 SIF 验证过 O0/O1/O2 的 `train_final_set`、`validation_final_set`、`test_final_set`。每个外层 split 都应该使用内部 `train_dataset_pool` 和 `train_positive_map.pkl`：

```text
<split>_final_set/train_dataset_pool
<split>_final_set/train_task_dataset
<split>_final_set/train_positive_map.pkl
```

不要把内部 `validation_dataset_pool` 传给 evaluation。因为 driver 使用 `--train-ratio 1.0`，内部 validation 是空副产物，`datasets.load_from_disk` 可能对 0-shard dataset 报 `IndexError`。这是预期现象，不是最终数据损坏。

O0/O1/O2 的验证结果摘要：

```text
每个 opt 的 train_final_set:      train_pool=5,859,275  train_task=5,607,071
每个 opt 的 validation_final_set: train_pool=103,254    train_task=100,651
每个 opt 的 test_final_set:       train_pool=10,105,358 train_task=9,103,676
```

### Fused 输出

所有输出写入 `OUTPUT_PATH`：

- `bc/`：Task2 生成的 `.bc`，按输入 split/相对路径镜像；`O0` 由 `llvm-as` assemble，其他优化等级由 `clang` reoptimize。
- `task3_fused/parquet/`：最终 parquet，通常包含 `train/`、`validation/`、`test/`。
- `task3_fused/manifests/`：fused Task3 的 `.bc` 级 success/failed/no-function manifest。
- `hf/`：由 parquet 保存出的 HuggingFace dataset，例如 `train_dataset`。
- `train_final_set`、`validation_final_set`、`test_final_set`：最终训练/验证/测试任务数据。如果设置了 `--final-output-root`，这里是指向 bandwidth 真实目录的 symlink。
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
- fused 脚本默认申请 3 个节点，方便 final 阶段把 `train`、`validation`、`test` 分散到不同机器；提交时仍可用 `sbatch --nodes=<N>` 覆盖。
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
