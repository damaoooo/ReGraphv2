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
└── benchmarks/
    └── speedtest.py       # 模型吞吐/显存测速
```

大规模 fused 数据（Ray / Slurm 多机预处理）不在本分支，见 `rope-gnn-cluster` 分支。

## 数据管线（task1 ~ task4）

```bash
# 完整流水线：lift -> reopt -> 函数提取（task4 默认关闭）
python Scripts/pipeline.py pipeline --input-path Binaries/Dataset-1 --output IR/Dataset-1-new --workers 32

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
- task3 为每个 `x.bc` 生成 `x_functions/`。
- 所有任务都支持 `--resume` 和 `--workers`。

单个二进制的 lift 也可以直接调用：

```bash
python Scripts/ida2llvm.py -f <file.i64> -o <out.ll>
```

`ida2llvm.py` 关闭数据库时不保存，输入的 `.i64` 不会被修改，同一个 `.i64` 重复运行结果一致；结束时在 stderr 打印 `lift stats: defined=… failed=… no_decompile=…`。

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
