# Scripts

所有命令都从仓库根目录执行。训练/评估使用 `uv run --locked --extra fa2 python ...`；数据管线使用 `uv run --locked --extra pipeline python ...`。环境配置见 `../docs/TRAINING_STEPS_GUIDE.md`。原 Conda `ReLL` 环境保留作回退，IDA 本地 SDK 仍需单独安装。

```text
Scripts/
├── ida2llvm.py            # IDA (idalib) microcode -> LLVM IR
├── pcode2llvm.py          # Ghidra P-Code -> LLVM IR（配合 ghidra/PcodeDump.java）
├── ghidra/
├── pipeline.py            # task1~4 的统一入口（inference.py 也通过它做 lift）
├── task1_lift.py          # 二进制 -> .i64 -> .ll（IDA）或 二进制 -> .ll（Ghidra）
├── task2_reoptimize.py    # .ll -> .bc，按 --opt-level 重新优化
├── task3_extract.py       # .bc -> <name>_functions/ 函数级 IR（调用 split_llvm_ir.sh）
├── task4_recompile.py     # 可选：.bc -> .re
├── split_llvm_ir.sh
├── utils.py               # task1~4 共用
├── build_db2.sh           # Dataset-2 构建流程（graph -> HF dataset -> wash -> final_set）
├── baselines/
│   └── ida2asm.py         # 汇编基线：把每个 IDA 函数导出为 .asm
├── train/                 # 训练 + 评估包装脚本
│   ├── train_test_fused_opt.sh
│   ├── train_test_oc_graph_ablation.sh
│   └── run_modernbert_comparison.sh
├── benchmarks/
│   └── speedtest.py       # 模型吞吐/显存测速
├── shaheen/               # Slurm 多机数据管线（Shaheen III），见下文“Slurm 多机数据管线”
│   ├── coreutils/         # coreutils 测试集 finalize 与 ReLL 文本
│   ├── irsim/             # IR 编辑距离（配对、文本、距离、汇总）
│   └── eir/               # rev.ng EIR 消融：lift、.re、rell 文本、bodyfirst/noisa fused、门槛统计
└── lifter_regress/        # 修改 lifter 后的回归与差分测试工具
```

## 数据管线（task1 ~ task4）

```bash
# 完整流水线：lift -> reopt -> 函数提取（task4 默认关闭）
uv run --locked --extra pipeline python Scripts/pipeline.py pipeline --input-path Binaries/Dataset-1 --output IR/Dataset-1-new --workers 32 --opt-level Oc

# 断点续传 / 从某个任务开始
python Scripts/pipeline.py pipeline --input-path ... --output ... --resume
python Scripts/pipeline.py pipeline --input-path ... --output ... --start-from 2 --opt-level Oc

# 已有 .i64，跳过 IDA 建库直接 lift
python Scripts/pipeline.py pipeline --input-path ... --output ... --task1-start-from-step2

# 用 Ghidra 后端
python Scripts/pipeline.py pipeline --input-path ... --output ... --lifter-backend ghidra --ghidra-home <GHIDRA_DIR>

# 单独跑某个任务
python Scripts/pipeline.py task2 --input-path IR/xxx --opt-level O0 --resume
python Scripts/pipeline.py task3 --input-path IR/xxx --resume
```

- task1 在 `--output` 下镜像输入目录结构；失败记录写到 `logs/lift_task1_log.txt`，`ida2llvm.py` 的详细日志写到 `logs/ida2llvm.log`。
- **论文（SEC'27）用的是 `Oc`**（保守 canonicalization），生成训练/评估数据一律用 `--opt-level Oc`。
- task2 在原目录把 `x.ll` 写成 `x.bc`；`--opt-level` 支持 `O0/O1/O2/O3/Os/Og/Oz/Oc`，`--arch auto` 按文件名推断 `-m32/-m64`。
- task3 为每个 `x.bc` 生成 `x_functions/`，不提取 IDA 段名在 `EXCLUDED_SEGMENTS`（`Scripts/split_functions.py`）里的函数：PLT 跳板段（`.plt`、`.plt.got`、`.plt.sec`、`.iplt`、`.MIPS.stubs`）和编译器生成的 `.init`/`.fini`。这是显式名单，不是通用 thunk 识别。段名来自 lifter 写在 `x.ll.lift.json` 里的 `segment`；task3 校验该状态文件描述的是旁边的 `x.ll`（lift 成功、记录的输出校验值等于 `x.ll` 的内容，`x.ll` 必须存在），且所记函数都在 `x.bc` 里有定义、每个函数都有段名，否则报错；`x.bc` 是否由该 `x.ll` 优化而来由 task2 自己的完成记录和 resume 负责，task3 不另行证明。resume 与新提取使用同一套校验，`x.ll` 改动后两者都会拒绝旧结果。这些函数只在样本里去掉，`x.ll`/`x.bc` 保持 lift 出来的完整模块，其他函数仍调用它们。task1 完成记录（且其输出校验值等于当前 `x.ll`）标明 Ghidra 后端时，没有段信息，按不过滤提取；其他情况缺少状态文件时报错（需用当前 lifter 重跑 task1）。每个 `x_functions/.complete.json` 的 `segment_filter` 记录实际采用的策略。单独运行 `split_llvm_ir.sh x.bc out/ --lift-state x.ll.lift.json`（或显式 `--no-segment-filter`），输出目录须为空。
- 所有任务都支持 `--resume` 和 `--workers`。

### 重跑与完整性

- Task 1 的 lifting、Task 2/4 的 resume 会校验输入内容、配置和输出 SHA256；旧版无校验记录的产物会重新生成。修改 lifter 后，Task 1 不会仅因为 `.ll` 存在而跳过。
- 已有 `.i64` 仍视为输入数据库复用，不会自动验证它与原始 binary 的对应关系。更新 binary 或 IDA 分析版本后，应使用重新生成的数据库。Task 1 不再自动删除输入目录中的 `.asm/.idb` 等文件。
- Task 3 在临时目录完成提取后替换旧目录，`.complete.json` 记录输入与每个产物的校验值；只有 CSV 或部分函数文件不算完成。旧目录中的过期函数不会混入新结果。
- Task 2/3/4 任一文件失败都会返回非零退出码，完整 pipeline 随即停止。Task 2/4 的产物先写临时文件，成功后再发布。
- 不要让多个 pipeline 同时写同一个输出目录。不同优化等级建议使用不同目录；同目录切换等级会替换 `.bc`，不保留多份结果。
- 严格 resume 会读取文件计算校验值，NAS 上可能有明显 I/O 成本。Task 2–4 限制待执行 Future 数量。
- Oc 保留 IR 的 target；其他 clang 优化模式及 Task 4 目前仍按宿主工具链和 `-m32/-m64` 工作，不能把它们当作保持源架构 ABI 的交叉编译器。
- 历史非法函数 bitcast 清理仍保留兼容行为；触发时会警告，并在 Task 2 完成记录的 `metadata.sanitized_removed_lines` 中记录删行数。这不是语义等价保证，正式实验应检查并修复产生非法 IR 的源头。

小样本回归：`uv run --locked --extra pipeline python -m pytest tests/test_pipeline_regressions.py -q`（需要 LLVM 工具和 pytest）。

### 批量函数提取

Task 3 先用 `llvm-extract` 按 128 个函数分块，再用 `split_funcs` 在每块内复用进程和 bitcode 缓冲区。它仍为每个函数建立独立 lazy module，不是修改原始模块后重复打印。

```bash
cmake -S LLVMTools -B LLVMTools/build -DCMAKE_BUILD_TYPE=Release
cmake --build LLVMTools/build -j 4
uv run --locked --extra pipeline python -m pytest tests/test_function_extraction.py -q
```

需要 LLVM 18.1 和 OpenSSL 开发包。这一次构建同时生成 CFG/DDG、MetaRemover 插件（`LLVMTools/build/lib/`）和 `split_funcs`（`LLVMTools/build/bin/`）。源码分别放在 `LLVMTools/cfg_exporter/`、`ddg_exporter/`、`meta_remover/`、`split_functions/`；Python 图解析仍在 `GraphBuilder/`，提取驱动在 `Scripts/split_functions.py`。

请在实际运行环境（例如 Shaheen 的 LLVM 容器）内构建，二进制不会提交到 Git。`split_funcs` 与 PATH 中 `llvm-extract` 的版本必须一致；缺失时自动警告并回退到纯 CLI 两级分块，已有工具损坏或版本不一致则报错，不静默降级。

可用 `REGRAPH_SPLIT_FUNCS=/absolute/path/to/split_funcs` 指定工具；指向不存在的路径可测试 CLI 回退。独立运行 shell 脚本默认使用 `python3`（仅标准库），Task 3 会显式传入当前 Python。

两种后端均保持 `T/t/W/w` 符号筛选、SHA1 文件名和 `function_map.csv` 格式；每个输出的 ModuleID 固定为 `regraph-function`。含空白或引号的函数名显式拒绝，不静默截断。缓存使用 LLVM 版本、可执行文件路径/大小/修改时间及分块配置，不重复哈希工具和源码。输入和输出内容仍做校验。修改提取逻辑后请不带 `--resume` 重跑；工具大小和时间戳不是内容完整性保证。

单个二进制的 lift 也可以直接调用：

```bash
python Scripts/ida2llvm.py -f <file.i64> -o <out.ll>
```

`ida2llvm.py` 关闭数据库时不保存，输入的 `.i64` 不会被修改，同一个 `.i64` 重复运行结果一致；结束时在 stderr 打印 `lift stats: defined=… failed=… no_decompile=…`。

## Slurm 多机数据管线（`Scripts/shaheen/`）

在 Slurm 集群上按节点分片跑 task1~4 与 fused/final_set 构建，一个作业占一个独占节点。所有作业脚本先 source `Scripts/shaheen/cluster_env.sh`，站点相关的设置都用环境变量给出，脚本里不写死路径、账户或分区：

| 变量 | 含义 |
|---|---|
| `REGRAPH_REPO` | 本仓库的 checkout（sbatch 运行的是脚本的副本，脚本自己找不到仓库） |
| `REGRAPH_SIF` | 管线运行用的 LLVM 18 Singularity 镜像 |
| `REGRAPH_BIND` | 绑定进镜像的目录，逗号分隔（至少包含数据根目录和仓库） |
| `REGRAPH_DATA` | 数据与各阶段工作目录的根（`eir/`、`coreutils/`、`irsim/` 下的脚本用它拼默认路径） |
| `IDA_TAR`、`IDAUSR_TEMPLATE` | IDA Pro 9.4 安装包（.tar.zst，每个作业解到节点本地 /tmp）和 IDAUSR 模板（许可、EULA、关闭 Lumina 的 ida.reg），IDA 阶段需要 |
| `REVNG_TAR` | rev.ng c865bc3d1d83 安装包（.tar.zst），EIR 阶段需要 |
| `IRSIM_PY` | irsim 用的 Python（需要 pandas、rapidfuzz） |

作业日志写到提交目录的 `logs/`（先建好）；分区、账户等用 `sbatch -p … -A …` 在提交时给。

- `ida_stage.sbatch {i64|lift|bbcount|funcflags}`：IDA 阶段（idalib 建库、`ida2llvm.py` lift、基本块数、函数属性），由 `ida_driver.py` 在每个节点上并行处理一个输入分片，可续跑；输入清单用 `make_lift_list.py` 生成。
- `submit_opt.sh` → `opt_stage.sbatch {shard|select|final|recompile|corpus|retok}`：每个优化级别的 task2（`task2_bc.py`）、fused 提取与建图（`fused_task3.py`）、样本选择（`select_level.py`）、`build_split.py` 生成 final_set；`task4_re.py` + `re_pack.sbatch` 生成并打包 `.re`。
- `build_rell_text.py`：与 final_set 逐行对齐的 ReLL（Qwen）文本。
- `coreutils/`：coreutils 测试集的 `build_split`（`cu_final.sbatch`）、finalize（加 binary_name/fva，`finalize_cu.py`）、不截断的 ReLL 文本（`build_cu_rell.py`）。
- `irsim/`：IR 编辑距离研究：`irsim_text.py`（规范化文本）、`build_pairs.py`、`compute_dist.py`、`summarize.py`，作业脚本 `irsim_text/irsim_cu/irsim_d1.sbatch`。
- `eir/`：rev.ng EIR 消融（只做 coreutils）：
  - `revng_shard.sbatch`：rev.ng isolate lift；
  - `eir_transform.py`：全部形式转换规则（块标签、辅助函数改 declare、删元数据、text_main 删除规则、bodyfirst、noisa、按地址改回 IDA 名）；
  - `eir_re.py`：每个构建合成一个 EIR `.re`；
  - `eir_text_b.py` + `build_eir_rell_b.py`：rell 文本 `text_main` / `text_b` / `text_b_bodyfirst`；
  - `eir_bf.py`（`--noisa`）+ `finalize_eir.py`：函数体在前的 fused 测试集；
  - `gate_bf.py`、`gate_bfn.py`、`token_lens.py`：与 O0 同行比较的门槛统计；
  - `feas_bf.py`：30 个函数的快速检查；
  - `make_manifests_eir.py`、`make_plans_eir.py`：IDA 特征和 CLAP/DEJINA 导出计划；
  - `make_pairs_eir.py`、`summarize_eir.py`、`text_b_view.py`：EIR 档的规范化 IR 距离；
  - `o0_index.py`、`ida_names.py`：O0 行索引和 IDA 函数名。

## 修改 lifter 后的回归（`Scripts/lifter_regress/`）

`runlift.py` 用指定 lifter 批量 lift 一组样本（`sample_builds.txt` 列出 111 个构建），`compare.py` / `ircmp.py` 比较两次 lift 的结果与 IR，`difftest.py` / `dt_all.py` 做执行差分，`order_lift.py` / `order_cmp.py` 检查函数顺序无关性，`rr_*.py` 汇总回归分类。需要 `IDADIR`（IDA 安装目录），`LIFT_PY` 默认用当前 Python。

## 训练和评估

```bash
bash Scripts/train/train_test_fused_opt.sh Oc              # 论文主设置：Oc fused 数据训练 + 测试
bash Scripts/train/train_test_fused_opt.sh O0              # 其他优化级别对比
bash Scripts/train/train_test_fused_opt.sh O2 --skip-train
bash Scripts/train/train_test_oc_graph_ablation.sh ddg     # 图分支消融：cfg / ddg / none / all
bash Scripts/train/run_modernbert_comparison.sh            # ModernBERT 与归档 RoFormer 结果对比
```

脚本会按自身位置推断仓库根目录，也可以用 `REPO_ROOT=...` 覆盖；各脚本的数据/输出路径参数见 `--help`。

## 测速

```bash
python Scripts/benchmarks/speedtest.py --help
```
