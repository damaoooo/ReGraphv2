# ReGraphV2 项目上下文

## 项目定位

这是一个面向 binary function similarity / retrieval 的研究代码库。核心思路是把二进制函数 lift 到 LLVM IR，经过可选 re-optimization/canonicalization 后，提取函数级 IR、CFG、DDG，再用 RoFormer + MoCo 对比学习训练函数嵌入。当前代码重点在 CFG/DDG 图分支、优化级别消融、Dataset-1/Dataset-2 的检索评估。

当前主要 Python 环境是 conda 的 `ReLL`。运行长任务、训练、评估、Ray pipeline 前优先使用：

```bash
conda run -n ReLL python ...
```

或者在 shell 中 `conda activate ReLL` 后再运行。不要假设系统 Python 依赖完整。

## 先读这些

优先使用 LSP 读取代码结构。Python 和 Bash LSP 已配置；在新会话里进入本 repo 后，可以先启动：

```text
lsp_start_server(server_id="python", workspace_root="/home/damaoooo/Downloads/regraphv2")
lsp_start_server(server_id="bash", workspace_root="/home/damaoooo/Downloads/regraphv2")
```

建议先用 `lsp_document_symbols` / `lsp_goto_definition` / `lsp_find_references` 看关键文件。`lsp_file_exports(include_signatures=true)` 在本项目的一些重型模块上可能超时，必要时改用 `lsp_document_symbols` 加定向读取。

不要全量扫生成数据目录。`IR/`、`Binaries/`、`IDBs/`、`runs/`、`data/models--*/` 都可能很大；只在任务明确需要时读取具体文件。

## 目录地图

- `paper/`: 论文 PDF、`analysis.ipynb` 历史分析和绘图 notebook、`figures/` 图片；先运行 notebook 的路径初始化单元。
- `docs/`: 使用说明，包括 `TRAINING_STEPS_GUIDE.md`。
- `Scripts/benchmarks/speedtest.py`: 模型测速入口，命令从仓库根目录执行。
- `envs/containers/`: 本地 Apptainer/Singularity 镜像；`logs/`: 数据生成和 lifting 日志。
- `Scripts/`: 数据生成、lift、reopt 脚本；`Scripts/train/` 是训练/评估包装脚本，`Scripts/baselines/` 放基线数据生成（如 `ida2asm.py` 汇编基线）。
- `Scripts/pipeline.py`: 传统 Typer pipeline，串起 task1/task2/task3/task4。
- `Scripts/task1_lift.py`: 用 IDA 生成 `.i64`，再调用 `Scripts/ida2llvm.py` lift 到 LLVM IR。IDA 路径硬编码为 `/home/damaoooo/ida-pro-9.3`。
- `Scripts/task2_reoptimize.py`: `.ll -> .bc`，支持 `O0/O1/O2/O3/Os/Og/Oz/Oc/Oc2`，其中 `Oc/Oc2` 是保守 canonicalization pass 组合。
- `Scripts/task3_extract.py`: 对 `.bc` 调 `Scripts/split_llvm_ir.sh` 提取函数级 IR。
- `Scripts/task4_recompile.py`: `.bc -> .re`，用于把优化后的 bitcode 重新编译成可执行/对象侧产物。
- `DataProcess/`: 从函数级 LLVM IR 构建 HuggingFace dataset。`cli.py` 是 Typer CLI，`dataset_builder_new.py` 串起 `FileProcessor` 和 `ParallelProcessor`，`file_processor.py` 生成 token、CFG、DDG。
- `GraphBuilder/`: Python 图解析和 LLVM pass 源码。`cfg_graph_builder.py` 输出 `(src_start, src_end, dst_start, dst_end, probability)`，`ddg_graph_builder.py` 输出 `(src_start, src_end, dst_start, dst_end)`。
- `Tokenizer/`: LLVM IR tokenizer、normalizer 和 tokenizer 文件。
- `Model/`: 模型骨架。`model_backbone.py` 的 `RoFormerGraph` 是 RoFormer 编码器 + CFG/DDG GATv2 图分支，`graph_utils.py` 把 span edge list batch 成 torch geometric 输入。
- `Pretrain/`: 训练逻辑。`pretrain_config.py` 集中默认配置，`pretrain_dataset.py` 读取 final_set，`pretrain_model.py` 是 MoCo wrapper，`run_pretrain.py` 是 Typer 训练入口。
- `evaluation.py`: 检索评估入口，生成/复用 embedding cache，输出 Recall@K、MRR、Markdown。
- `inference.py`: 单目录/单二进制推理流水线，封装 lift、graph generation、embedding、verbose attention/graph payload。
- `api_server.py`: FastAPI 包装 `/embed` 和 `/health`，用 `REGRAPH_MODEL_PATH` 指定模型。

## 核心数据格式

训练/验证/测试通常使用 `*_final_set` 目录，里面关键文件是：

```text
train_dataset_pool/
train_task_dataset/
train_positive_map.pkl
```

`train_dataset_pool` 是全池 HuggingFace dataset；`train_task_dataset` 是训练或评估 anchor/task dataset；`train_positive_map.pkl` 是 anchor index 到 positive indices 的映射。评估脚本现在的主要签名是：

```bash
python evaluation.py <train_positive_map.pkl> \
  --dataset-path <train_dataset_pool> \
  --model-path <model_dir> \
  --embeddings-path <cache.pth-or-npy> \
  --markdown-output <result.md> \
  --cfg --ddg --bf16
```

如果 embedding cache 已经存在并且全量评估，可以只用 positive map + embedding cache，不必重新传模型和 dataset。`.pth/.pt` cache 用 torch，`.npy` cache 用 numpy。

## 训练入口

主入口：

```bash
python -m Pretrain.run_pretrain train \
  --dataset-dir IR/Dataset-1-Oc-fused/train_final_set \
  --validation-dataset-dir IR/Dataset-1-Oc-fused/validation_final_set \
  --cfg --ddg \
  --set max_seq_length=2048 \
  --set save_steps=10000 \
  --set output_dir=runs/dataset1_oc_fused/checkpoints \
  --set logging_dir=runs/dataset1_oc_fused/tensorboard \
  --set final_model_dir=runs/dataset1_oc_fused/model
```

`run_pretrain.py` 会按图分支自动给 `output_dir`、`logging_dir`、`final_model_dir` 加后缀：`cfg_ddg`、`cfg`、`ddg` 或 `plain`。例如 `final_model_dir=runs/x/model` 且 `--cfg --ddg` 时，实际模型目录是 `runs/x/model_cfg_ddg`。

常用包装脚本：

```bash
bash Scripts/train/train_test_fused_opt.sh Oc   # 论文主设置
bash Scripts/train/train_test_fused_opt.sh O0
bash Scripts/train/train_test_fused_opt.sh O2 --max-length 4096
bash Scripts/train/train_test_fused_opt.sh Os --skip-train
bash Scripts/train/train_test_oc_graph_ablation.sh ddg
```

`docs/TRAINING_STEPS_GUIDE.md` 说明了当前默认更偏向按 epoch 训练：`num_train_epochs=1`、`max_steps=-1`、`save_steps=10000`。短 smoke test 才临时设置 `max_steps`。

## 大规模数据生成

大规模 Dataset-1 fused 数据（Ray / Slurm，多机）在 `rope-gnn-cluster` 分支上维护和运行（Shaheen 集群），本分支不再保留 `Scripts/ray_opt_ablation/`。需要查看或修改 fused 流水线时切到该分支。

论文（SEC'27 artifact）使用的是 `Oc`：`sroa, mem2reg, simplifycfg, early-cse, sccp, correlated-propagation, jump-threading, simplifycfg, reassociate, gvn, dce, bdce, adce, simplifycfg`，定义在 `Scripts/task2_reoptimize.py`，与 `sec27-artifact` / `rope-gnn-cluster` 分支一致。生成训练/评估数据一律用 `--opt-level Oc`；`Oc2` / `Oc3` 是效果不好的探索设置，不要用。在 `rope-gnn-cluster` 上生成 `Oc` 数据时确保没有 export `REGRAPH_IR_CANONICALIZATION`（Oc3 的位宽规整会通过它作用到 Oc）。这些都不等同于 clang `-O?`。

## 推理和 API

`inference.py` 可从 executable dir 开始跑 lift、graph generation、embedding，并可导出 verbose payload。verbose 结果对可视化很重要，函数值通常包含：

```text
embedding, use_cfg, use_ddg, ir, tokens,
cfg_graph, cfg_model_input, ddg, ddg_model_input, attention_weights
```

API 服务由 `api_server.py` 提供。启动前设置：

```bash
export REGRAPH_MODEL_PATH=/path/to/model_cfg_ddg
export REGRAPH_USE_CFG=1
export REGRAPH_USE_DDG=1
python api_server.py
```

`REGRAPH_USE_CFG` / `REGRAPH_USE_DDG` 也可以在请求里覆盖。`inference.resolve_graph_flags` 会尝试从模型目录里的 metadata/config 推断图分支。

## 常见坑

- 本项目很多默认路径是本机绝对路径，尤其 `Pretrain/pretrain_config.py`、`Scripts/task1_lift.py` 和若干 shell 脚本。运行前先确认路径，不要盲目搬命令。
- 训练和评估默认会很久、吃 GPU、写大量 checkpoint/embedding。除非用户明确要跑，先只做静态分析或给命令。
- 旧 `Scripts/ablation.sh` 已归档并移除，其历史调用混用了训练池和测试映射。新任务参考 `Scripts/train/train_test_oc_graph_ablation.sh`、`Scripts/train/run_modernbert_comparison.sh` 和当前 `evaluation.py`；旧产物恢复说明见 NAS 上的 `/mnt/nas/ReLL-archives/2026-10-21-roformer-retirement/README.md`。
- `DataProcess` 依赖 `results.db`，数据库中 `results` 表记录 `input_path` 等 graph generation 结果。
- CFG 图边有第 5 个 probability/edge_attr，DDG 图边没有 edge_attr。`Model/graph_utils.py` 的 `build_batched_span_graph_tensors(..., include_edge_attr=True/False)` 是训练和评估 collator 的共同关键路径。
- `PretrainConfig.__setattr__` 会同步 `max_seq_length` 和 `max_position_embeddings`，改序列长度时不要只改其中一个。
- `run_pretrain.py` 的 `SafeTrainer` 会把最终模型保存为 torch bin，避免 safetensors shared tensor 问题，并维护 `checkpoint-last`、`checkpoint-best-validation-loss`、`checkpoint-best-train-loss`。
- 修改模型输入、collator 或 graph tensor shape 后，至少跑一个小 batch/smoke test；图分支错误容易到训练中后段才暴露。

## 代码阅读建议

遇到训练/模型问题，优先读：

```text
Pretrain/run_pretrain.py
Pretrain/pretrain_config.py
Pretrain/pretrain_dataset.py
Pretrain/pretrain_model.py
Model/model_backbone.py
Model/graph_utils.py
evaluation.py
```

遇到数据生成/reoptimization 问题，优先读：

```text
Scripts/task1_lift.py
Scripts/task2_reoptimize.py
Scripts/task3_extract.py
Scripts/task4_recompile.py
DataProcess/cli.py
DataProcess/dataset_builder_new.py
DataProcess/file_processor.py
GraphBuilder/cfg_graph_builder.py
GraphBuilder/ddg_graph_builder.py
```

遇到在线推理/API/可视化数据格式问题，优先读：

```text
inference.py
api_server.py
GraphBuilder/*
Model/graph_utils.py
```
