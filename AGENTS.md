# ReLL 项目上下文（modernBERT-gnn 分支）

## 接手必读（先看这里）

本仓库是**公开**的。实验结果、论文口径、审稿意见只放在本地、不进 git 的文件里。接手实验或改论文时，按顺序读：

1. **`workspace/AGENTS.md`**（本地，被 `.gitignore` 和 `.git/info/exclude` 忽略）：实验设计、全部结果的出处、论文写作口径、已知例外和局限、操作规矩。**改论文前必须读完。**
2. `workspace/docs/PAPER_BRIEF_2026-11-04.md`：给改论文用的完整结果和论证，数字由脚本从打分文件生成。
3. `workspace/docs/REPORT_2026-11-05_{zh,en}.pdf`：同一批结果的 LaTeX 报告（图和表），生成脚本 `workspace/exp/code/qwen_eval/make_report.py`。
4. 本文件余下部分：代码结构、环境、训练和评估命令。

如果 `workspace/` 不存在，说明你不在作者的工作站上：只能做代码层面的工作，不要猜测或编造实验数字。

## 项目定位

面向 binary function similarity / retrieval 的研究代码库，模型名 **ReLL**（旧名 ReGraphv2，代码和路径里仍可见，对外一律写 ReLL）。流程分四步：
1. 用基于 IDA Pro / Hex-Rays 的 lifter 把二进制函数 lift 成 LLVM IR；
2. 用保守的重优化 pass 组合 `-Oc` 规范化；
3. 提取函数级 IR、CFG、DDG；
4. 用 ModernBERT 编码器加 CFG / DDG 图分支，以 MoCo 对比学习训练函数嵌入。

lift + `-Oc` 这一层与模型无关：IR 可以直接给其他文本模型，也可以重编译成 x86 目标文件（`.re`），交给只吃二进制的工具。论文的主张是这层预处理本身的价值，而不是 ReLL 是最好的模型。

## 实验流水线概览

只描述流程，不含结果；结果见 `workspace/AGENTS.md`。

1. **数据**：Talos Dataset-1（关闭内联重新编译）和 coreutils 9.5，6 种架构（arm / mips / x86，各 32 / 64 位），gcc 和 clang 多个版本与优化级别。
2. **lift**：`Scripts/ida2llvm.py`（IDA 9.4，新建 IDB，关闭 Lumina）。数据生成在 Slurm 集群上做（`modernBERT-cluster` 分支）。
3. **重优化**：7 个级别 O0（不做 LLVM 重优化）、O1、O2、O3、Os、Oz、Oc；`task4_recompile.py` 把 Oc IR 编回 x86 / x86-64 的 `.re`。
4. **建数据集**：`DataProcess/` 生成 `*_final_set`（token、CFG、DDG、positive map）。
5. **训练和嵌入**：ReLL 每个级别一个模型（`Scripts/train/train_test_fused_opt.sh`）；baseline 在 `baseline/`（本地 checkout，不进 git）。
6. **打分**：所有方法共用一份冻结的候选列表，用 `Scripts/scoring/` 打分（R@1 和 MRR，Pool100 / Pool10000）。
7. **报告**：`make_report.py` 从打分文件生成中英文报告。

## 分支

| 分支 | 用途 |
|---|---|
| `modernBERT-gnn`（本分支） | 单机工作站：lift、建数据集、训练、评估 |
| `modernBERT-cluster` | Slurm 集群上的大规模数据生成（分片的 IDA / LLVM 阶段）、IR 相似度研究、EIR 消融脚本 |

两条分支长期并存，**不合并**。共用模块（lifter、Model、DataProcess、tests 等）改动时，按需要把同样的修改移植到另一条分支，避免无谓的分叉。

## 环境

用 uv 管理独立 `.venv`，依赖以 `pyproject.toml` 和 `uv.lock` 为准；`requirements.txt` 只是历史清单。

```bash
uv sync --locked                       # 核心
uv sync --locked --extra pipeline      # + 数据生成
uv sync --locked --extra fa2           # + FlashAttention（需要与 Torch cu130 匹配的 CUDA Toolkit，用 CUDA_HOME 指定）
uv run --locked --extra fa2 python ...
```

API 使用 `--extra api`，同时用到多个功能时把对应的 extra 一起传入。以下需要单独安装，不在锁文件里：
- LLVM 18.1：`opt`、`clang`、`llvm-extract`；
- Graphviz；
- IDA Pro 9.4 / idalib，需要许可证，只有 lift 时才用到。

详见 `docs/TRAINING_STEPS_GUIDE.md`。

## 先读这些

优先用 LSP 读代码结构，例如 `lsp_document_symbols`、`lsp_goto_definition`、`lsp_find_references`。

```text
lsp_start_server(server_id="python", workspace_root="<repo root>")
lsp_start_server(server_id="bash", workspace_root="<repo root>")
```

`lsp_file_exports(include_signatures=true)` 在一些重型模块上可能超时，必要时改用 `lsp_document_symbols` 加定向读取。

不要全量扫描数据目录（`workspace/` 的布局见 `workspace/AGENTS.md`）。`workspace/`、`IR/`、`Binaries/`、`runs/`、`logs/` 都可能很大，而且都被 git 忽略，只在任务明确需要时读取具体文件。本地大数据（数据集、模型运行、分数）统一放在 `workspace/` 下，任何时候都不要提交。

## 目录地图

- `Scripts/ida2llvm.py`：IDA 9 microcode → LLVM IR 的 lifter。
  - 输出头部记录 `lifter_id`，即文件 sha256 的前 16 位，用来追溯数据由哪个版本生成。
  - 遇到无法保证语义的函数时显式拒绝，原因写进 `.ll.lift.json`，而不是输出可能错误的 IR。
  - 单元测试 `tests/test_ida2llvm_*.py` 会 stub IDA 模块（见 `test_ida2llvm_microcode.py` 里的 `IDA_MODULES`）。lifter 新增 IDA import 时，要同步更新这个列表。
- `Scripts/pipeline.py`：Typer pipeline，串起 task1 到 task4。
  - `task1_lift.py`：用 IDA 生成 `.i64`，再调用 lifter。IDA 路径按 `--ida-path` > `IDADIR` > `~/.idapro/ida-config.json` 的顺序解析。`.venv` 里没有装 `idapro`，lift 时要加 `PYTHONPATH=$IDADIR/idalib/python`。
  - `task2_reoptimize.py`：`.ll → .bc`，支持 `O0/O1/O2/O3/Os/Og/Oz/Oc`。
  - `task3_extract.py`：调用 `split_llvm_ir.sh` 和 `split_functions.py` 提取函数级 IR。
  - `task4_recompile.py`：`.bc → .re`（clang O0 后端编成 x86 / x86-64 目标文件）。
- `Scripts/scoring/`：所有对比方法共用的正式工具。
  - `sample_candidates.py`：冻结候选列表的抽样器，PCG64，种子 42；
  - `test_candidates.py`：自检，包括完整性、合法性，以及按种子重新生成后逐字节一致；
  - `score_candidates.py`：打分器，输出 R@1 / MRR，Pool100 和 Pool10000。

  已发布的分数都记录了打分器的 sha256，改它就会影响可比性，修改之前先确认。
- `Scripts/tokenizer/`：`corpus_text.py` 构建 BPE 语料，`train_bpe.py` 重训分词器。正式分词器在 `Tokenizer/output_tokenizer/`。
- `Scripts/train/`：训练和评估的包装脚本。`train_test_fused_opt.sh` 是主入口，`train_test_oc_graph_ablation.sh` 用于图分支消融。
- `Scripts/baselines/ida2asm.py`：导出汇编基线。
- `Scripts/benchmarks/speedtest.py`：吞吐和显存测速。
- `Scripts/pcode2llvm.py` + `Scripts/ghidra/`：Ghidra 后端（`pipeline.py --lifter-backend ghidra` / `task1_lift.py --backend ghidra`），可选。
- `LLVMTools/`：CFG / DDG 导出插件、MetaRemover、`split_funcs` 的 C++ 源码，统一用 CMake 构建：

  ```bash
  cmake -S LLVMTools -B LLVMTools/build -DCMAKE_BUILD_TYPE=Release
  cmake --build LLVMTools/build -j
  ```

  插件生成在 `build/lib/`，可执行文件在 `build/bin/`。
- `DataProcess/`：从函数级 IR 构建 HuggingFace 数据集。
  - `cli.py`：Typer CLI；
  - `builder.py`：数据格式、worker 调度、发布；
  - `file_processor.py`：生成 token、CFG、DDG；
  - `wash.py`：清洗；
  - `select_samples.py`：抽样。
- `GraphBuilder/`：图构建，边与 token span 对齐。CFG 的边是 `(src_start, src_end, dst_start, dst_end, probability)`，DDG 的边没有 probability。
- `Tokenizer/`：IR normalizer 和 BPE tokenizer。
- `Model/`：
  - `model_backbone.py` 里的 `ModernBertGraph` = ModernBERT 编码器 + CFG / DDG 图分支（`SpanGraphBranch`）；
  - `graph_utils.py` 负责把 span 边表 batch 成 PyG 输入。
- `Pretrain/`：
  - `pretrain_config.py`：默认配置；
  - `pretrain_dataset.py`：读取 final_set；
  - `pretrain_model.py`：`ModernBertPretrainModel`，以及 MoCo wrapper `MoCoPretrainModel`；
  - `run_pretrain.py`：训练入口。
- `evaluation.py`：检索评估，生成或复用 embedding cache，输出 Recall@K、MRR 和 Markdown 报告。
  - 数据加载用多进程（`--num-workers`），并且默认按长度分批（`--length-bucketing`）。结果按原行号写回，行序与数据集一致。
- `inference.py`：单目录 / 单二进制的推理流水线，覆盖 lift、建图、embedding，以及 verbose payload。
- `api_server.py`：FastAPI 封装，提供 `/embed` 和 `/health`。

## 核心数据格式

训练、验证、测试都用 `*_final_set` 目录：

```text
train_dataset_pool/      # 全池 HuggingFace dataset
train_task_dataset/      # anchor / task dataset
train_positive_map.pkl   # anchor index -> positive indices
```

```bash
python evaluation.py <train_positive_map.pkl> \
  --dataset-path <train_dataset_pool> --model-path <model_dir> \
  --embeddings-path <cache.npy> --markdown-output <result.md> --cfg --ddg --bf16
```

embedding cache 已存在、并且做全量评估时，只需要 positive map 和 cache。

## 训练

```bash
bash Scripts/train/train_test_fused_opt.sh Oc                      # 主设置
bash Scripts/train/train_test_fused_opt.sh O0
bash Scripts/train/train_test_fused_opt.sh Oc --split-suffix _4096 # 使用 *_final_set_4096
bash Scripts/train/train_test_fused_opt.sh Os --skip-train
bash Scripts/train/train_test_oc_graph_ablation.sh ddg
```

- `run_pretrain.py` 会按图分支，给 `output_dir`、`logging_dir`、`final_model_dir` 加后缀 `cfg_ddg` / `cfg` / `ddg` / `plain`。
- 默认按 epoch 训练：`num_train_epochs=1`、`max_steps=-1`。只有短的冒烟测试才临时设置 `max_steps`。

## Oc

`Oc` 由 14 个固定 pass 组成：

```text
sroa, mem2reg, simplifycfg, early-cse, sccp, correlated-propagation, jump-threading,
simplifycfg, reassociate, gvn, dce, bdce, adce, simplifycfg
```

- 定义在 `Scripts/task2_reoptimize.py`，与 `sec27-artifact` 分支一致；
- 生成训练和评估数据一律用 `--opt-level Oc`，它不等同于 clang 的 `-O?`；
- 实验里的对照级别 `O0` 指"不做 LLVM 重优化"。Hex-Rays 在 lift 之前已经做过自己的化简，所以它不等于"只 lift"。

## 推理和 API

`inference.py` 的 verbose 结果包含：

```text
embedding, use_cfg, use_ddg, ir, tokens, cfg_graph, cfg_model_input, ddg, ddg_model_input, attention_weights
```

启动 API：

```bash
export REGRAPH_MODEL_PATH=/path/to/model_cfg_ddg REGRAPH_USE_CFG=1 REGRAPH_USE_DDG=1
python api_server.py
```

## 常见坑

- **耗时：** 训练和评估很慢、很吃 GPU，还会写出大量 checkpoint 和 embedding。除非用户明确要跑，否则只做静态分析，或者只给出命令。
- **批量任务要并行：** 机器有多核，例如校验、扫描、打分、特征提取，先评估能否多进程加速，再决定是否串行跑。
- **数据库依赖：** `DataProcess` 依赖 `results.db`，其中 `results` 表记录 graph generation 的结果。
- **CFG 与 DDG 的区别：** CFG 边有第 5 个 probability / edge_attr，DDG 没有。`Model/graph_utils.build_batched_span_graph_tensors(..., include_edge_attr=...)` 是训练和评估 collator 共用的关键路径。
- **序列长度：** `PretrainConfig.__setattr__` 会同步 `max_seq_length` 和 `max_position_embeddings`，改序列长度时不要只改其中一个。
- **模型保存：** `SafeTrainer` 把最终模型存为 torch bin，并维护 `checkpoint-last`、`checkpoint-best-validation-loss`、`checkpoint-best-train-loss`。
- **改动后要冒烟测试：** 修改模型输入、collator 或图张量形状后，至少跑一个小 batch；图分支的错误往往到训练中后段才暴露。
- **测试：** 全套测试用 `uv run python -m pytest tests -q`，可以按文件并行跑。

## 代码阅读建议

- **训练 / 模型：**
  - `Pretrain/run_pretrain.py`、`pretrain_config.py`、`pretrain_dataset.py`、`pretrain_model.py`；
  - `Model/model_backbone.py`、`graph_utils.py`；
  - `evaluation.py`。
- **数据生成 / 重优化：**
  - `Scripts/task1_lift.py` 到 `task4_recompile.py`；
  - `Scripts/ida2llvm.py`；
  - `DataProcess/cli.py`、`builder.py`、`file_processor.py`；
  - `GraphBuilder/*`。
- **评测协议：** `Scripts/scoring/*`。
- **推理 / API：** `inference.py`、`api_server.py`、`GraphBuilder/*`、`Model/graph_utils.py`。
