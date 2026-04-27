# ReGraph v2 Fused Task3 数据生成重构需求文档

## 1. 背景

当前数据生成流程大致为：

1. Task2 reoptimization：`.ll -> .bc`
2. Task3 function-extract：从 `.bc` 拆出大量函数级 `.ll` 小文件
3. graph-extract：扫描函数级 `.ll`，生成 purified IR、instrumented IR、CFG dot、DDG dot，并写入 `results.db`
4. `DataProcess.cli`：读取 `results.db`，解析 CFG/DDG，tokenize，写 parquet/HF dataset

当前问题：

- function-extract 会生成海量函数级小文件。
- graph-extract 又会生成更多中间小文件。
- Shaheen 对文件数量有限制，总文件数不能超过 `1M`，最好低于 `10K`。
- 后续计划在 Shaheen 上使用 Ray 多节点并行，可能申请多台机器；Shaheen 每台机器为 384 core exclusive。
- 旧流程依赖 `function_map.csv`、hashed function file name、`results.db`，这些都是为绕过 Unix 文件名长度和小文件组织问题服务的，新流程不再需要。

本次需求是在 `rope-gnn-cluster` 分支上重构 Task3 及 DataProcess，使 Task3 直接完成：

```text
function-extract + CFG/DDG graph build + tokenize + truncate + parquet write
```

并让 `DataProcess.cli` 只负责读取 parquet 并保存 HuggingFace dataset。

## 2. 总目标

实现一个 fused Task3 阶段，输入 Task2 生成的 `.bc` 文件，输出训练可直接使用的数据。

每条函数记录字段为：

```text
binary_name
function_name
file_path
input_ids
cfg_graph
ddg_graph
```

字段定义：

- `binary_name`：从输入根目录到 `.bc` 文件的相对路径，去掉 `.bc` 后缀。
- `function_name`：直接来自 `llvm-nm` 的 defined text symbol。
- `file_path`：保留用于调试和兼容下游工具，指向来源 `.bc` 或可识别该函数来源的逻辑路径。
- `input_ids`：purified IR tokenize 后的 token ids，在 Task3 阶段直接截断。
- `cfg_graph`：parsed CFG edge list，不保存 dot。
- `ddg_graph`：parsed DDG edge list，不保存 dot。

## 3. 非目标

本次不需要：

- 保留函数级 `.ll` 小文件，除非显式开启 `--debug`。
- 生成或保留 `function_map.csv`。
- 生成或依赖 `results.db`。
- 保留旧 DataProcess SQLite 读取路径。
- 单独运行 wash 阶段。
- 向前兼容旧分支数据格式。
- 硬编码 Shaheen 路径、conda 环境路径、Ray 集群参数。

## 4. Task3 功能需求

### 4.1 输入

Task3 输入为 Task2 reoptimization 后的目录，内部包含 `.bc` 文件。

Task3 需要递归扫描输入目录下的 `.bc` 文件。

如果输入目录下存在：

```text
train/
validation/
test/
```

则 Task3 需要保持 split 结构，分别处理并输出 split 对应数据。

如果不存在 split 目录，则按单一 dataset 处理。

### 4.2 函数发现

对每个 `.bc` 文件执行 `llvm-nm`，提取 defined text symbol：

```bash
llvm-nm file.bc | awk '$2 == "T" {print $3}'
```

无函数的 `.bc` 不应视为致命失败，应记录为 no-function/skipped 状态。

### 4.3 函数抽取

对每个 function name 使用 `llvm-extract` 抽取临时函数 IR。

临时文件必须放在受控 temp/debug 目录中。

默认行为下，以下中间产物全部在处理完成后清理：

- 函数 `.ll`
- purified `.ll`
- instrumented `.ll`
- CFG dot
- DDG dot

开启 `--debug` 时保留这些中间文件，用于排查问题。

### 4.4 Graph 生成

对临时函数 IR 执行现有 opt pass：

1. purify metadata
2. generate DDG
3. generate CFG

继续复用当前已有逻辑和 `.so` 路径配置：

- `DEFAULT_PURIFY_SO_PATH`
- `DEFAULT_DDG_SO_PATH`
- `DEFAULT_CFG_SO_PATH`

最终不保留 graph dot 文件，只把 dot 解析为 edge list。

### 4.5 Tokenize

使用现有 tokenizer：

```text
Tokenizer.ir_tokenizer.load_tokenizer
```

默认 tokenizer path 使用当前项目默认值，但 CLI 必须允许通过 `--tokenizer-path` 覆盖。

tokenize 的输入应为 purified IR。

### 4.6 截断

Task3 阶段直接执行截断，默认：

```text
max_seq_length = 2048
```

规则：

- 如果 `input_ids` 长度超过 `max_seq_length`：
  - 保留前 `max_seq_length - 1` 个 token。
  - 末尾追加 tokenizer 的 `eos_token_id`。
- `cfg_graph` 和 `ddg_graph` 中超出 token 范围的边必须删除。
- 判断边是否超界时，沿用当前 `dataset_wash.truncate_example` 的语义：边中最大 index/span 不得超过截断边界。
- 生成后的记录必须保证图边不会引用被截断掉的 token 范围。

### 4.7 成功/失败规则

函数级成功条件：

```text
binary_name 存在
function_name 存在
input_ids 非空
cfg_graph 存在
ddg_graph 存在
```

任何一个缺失，该函数视为失败，不写入最终 parquet。

失败函数应记录日志和 manifest，但不能中断整个 `.bc` 或整个任务。

`.bc` 级成功条件：

- 该 `.bc` 已被完整扫描。
- 所有函数均已尝试处理。
- 成功函数记录已持久化到 parquet 临时 shard 或最终 shard。
- 失败函数已写入 failure manifest。

`.bc` 级 resume 只以 `.bc` 为单位，不需要函数级 resume。

## 5. Parquet 输出需求

### 5.1 Schema

最终 parquet schema：

```text
binary_name: string
function_name: string
file_path: string
input_ids: sequence<int32>
cfg_graph: sequence<sequence<float32 or compatible numeric>>
ddg_graph: sequence<sequence<int32>>
```

说明：

- `cfg_graph` 当前边格式含 float edge attr，因此可保持 float-compatible schema。
- `ddg_graph` 为 int edge list。
- 不保存 IR 文本。
- 不保存 dot。
- 不保存 purified/instrumented IR。

### 5.2 分片目标

Shaheen 文件数限制：

- 硬限制：总文件数不能超过 `1M`
- 项目目标：最好不超过 `10K`
- 默认设计目标：临时 parquet + 最终 parquet 总数不超过 `5000`

最终 parquet 目标大小：

```text
1 GiB per shard
```

但如果严格 1 GiB 会导致文件数超过 `--max-parquet-files`，则文件数量上限优先。

### 5.3 推荐写入方案

采用两阶段受控分片：

1. Ray/local worker 处理 `.bc` chunk，写有限数量的临时 parquet shard。
2. driver 执行 compaction，把临时 parquet 合并成约 1 GiB 的最终 parquet。
3. compaction 成功后默认删除临时 parquet，降低文件数。
4. 若失败，保留 manifest 和必要中间信息用于 resume/debug。

不采用：

- 每个函数一个文件。
- 每个 Ray task 随意写最终 parquet。
- 单一集中 writer actor 接收全部 records。

原因：

- 避免小文件爆炸。
- 避免集中 actor 成为吞吐瓶颈。
- 更容易控制总文件数和 resume。

## 6. Ray / Local Backend 需求

### 6.1 Backend 参数

Task3 需要支持：

```text
--backend local|ray
```

默认可为 `local`。

### 6.2 Local Backend

Local backend 用于本地开发和验证。

要求：

- 支持多进程或线程并行。
- 可在当前非 Shaheen 环境运行。
- 与 Ray backend 输出同样 schema 和 manifest。

### 6.3 Ray Backend

Ray backend 用于 Shaheen 多节点。

要求：

- 不硬编码 Ray 地址。
- 通过环境变量或 Ray 默认机制连接集群。
- 支持多台 Shaheen 节点，每台 384 cores exclusive。
- 每个 Ray task 处理 `.bc` chunk。
- chunk 大小可配置。
- 输出详细日志，方便后期在 Shaheen debug。

建议 CLI 参数：

```text
--backend ray
--chunk-size
--target-functions-per-task
--max-parquet-files
--target-shard-size-bytes
--resume
--debug
```

## 7. Resume 需求

Resume 粒度：

```text
.bc 文件级别
```

需要记录：

- 已完成 `.bc`
- no-function `.bc`
- 失败 `.bc`
- 每个 `.bc` 的函数数量
- 成功函数数量
- 失败函数数量
- 输出 shard 路径
- 失败原因摘要

`--resume` 时：

- 已成功完成的 `.bc` 跳过。
- no-function `.bc` 跳过。
- 未完成或失败 `.bc` 重新处理。
- 不需要函数级断点续跑。

## 8. Debug / Logging 需求

### 8.1 默认日志

需要至少记录：

- backend 类型
- 输入路径
- 输出路径
- split
- tokenizer path
- max_seq_length
- Ray cluster CPU 数，如果是 Ray backend
- chunk id
- chunk size
- `.bc` path
- function count
- success count
- failure count
- no-function count
- parquet shard path
- compaction 输入/输出文件数量
- stderr 摘要，尤其是 opt/llvm-extract 失败信息

### 8.2 Manifest

建议输出：

```text
manifests/task3_success.jsonl
manifests/task3_failed.jsonl
manifests/task3_skipped.jsonl
manifests/task3_no_functions.jsonl
manifests/compaction_success.jsonl
manifests/compaction_failed.jsonl
```

### 8.3 Debug 模式

开启：

```text
--debug
```

行为：

- 保留临时函数 IR。
- 保留 purified IR。
- 保留 instrumented IR。
- 保留 CFG/DDG dot。
- 日志中记录这些文件路径。
- Debug 文件应集中放入 debug 目录，不能散落到输入目录中。

## 9. DataProcess.cli 需求

旧 CLI 删除，不再支持：

- 读取 `results.db`
- 根据 `results.db` 重新解析 CFG/DDG
- 旧 `directory input_dir output_file` 语义
- `DatasetBuilder`
- `ParallelProcessor` 中 SQLite 相关路径

新 CLI 只负责：

1. 读取 Task3 输出的 parquet。
2. 按 split 保存 HuggingFace dataset。
3. 保持 schema。
4. 输出可由 `datasets.load_from_disk` 加载的目录。

建议命令语义：

```bash
python -m DataProcess.cli parquet \
  --input-parquet-dir <task3_output/parquet> \
  --output-dir <hf_dataset_output>
```

如果输入包含 split：

```text
train/
validation/
test/
```

则输出：

```text
train_dataset/
validation_dataset/
test_dataset/
```

或沿用项目现有命名约定。

## 10. 下游适配需求

### 10.1 Dataset Features

`DataProcess/dataset_features.py` 需要更新，包含：

```text
binary_name
function_name
file_path
input_ids
cfg_graph
ddg_graph
```

### 10.2 Positive Pair / Split 逻辑

旧逻辑依赖：

- `file_path`
- `function_map.csv`
- hashed function filename
- 旧目录名推导 `origin_binary_name`

新逻辑应直接使用：

```text
binary_name
function_name
```

正样本分组规则：

```text
group by (binary_name, function_name)
```

### 10.3 Training

训练侧主要消费：

```text
input_ids
cfg_graph
ddg_graph
```

应尽量保持训练 collator 不变。

如果训练侧或工具侧读取额外字段，不应受影响。

## 11. 删除/废弃范围

可以删除或重写以下旧路径中不再需要的逻辑：

- `results.db` 初始化、写入、读取逻辑
- `function_map.csv` 生成和读取逻辑
- 基于 hashed function `.ll` 文件名的映射逻辑
- GraphBuilder 独立数据库 pipeline
- DataProcess 旧 SQLite builder 流程
- 单独 wash CLI 中已被 Task3 覆盖的过滤/截断流程

注意：删除前需要确认没有训练主路径仍直接 import 被删除对象。

## 12. 验收标准

### 12.1 本地验收

使用小样本 `.bc` 运行：

```bash
python Scripts/task3_extract.py \
  --input-path <sample_input> \
  --output <sample_output> \
  --backend local \
  --workers 4
```

应满足：

- 成功生成 parquet。
- parquet schema 正确。
- 没有遗留临时文件，除非开启 `--debug`。
- manifest 存在。
- 失败函数不会中断整个任务。
- `DataProcess.cli` 可读取 parquet 并保存 HF dataset。
- `datasets.load_from_disk` 能加载结果。

### 12.2 Debug 验收

运行：

```bash
python Scripts/task3_extract.py ... --debug
```

应满足：

- debug 目录存在。
- 中间 `.ll`、purified IR、instrumented IR、dot 文件可查。
- 日志能追踪 function name 到 debug 文件。

### 12.3 Resume 验收

流程：

1. 第一次运行 Task3。
2. 第二次使用 `--resume`。
3. 验证已成功 `.bc` 被跳过。
4. 删除某个 `.bc` 的 success manifest 或模拟失败。
5. 验证只重跑对应 `.bc`。

### 12.4 文件数量验收

对于大规模任务，必须在日志中输出：

```text
temporary parquet count
final parquet count
total parquet count
max_parquet_files
```

并保证默认配置下目标为：

```text
temporary + final parquet <= 5000
```

### 12.5 Shaheen/Ray 验收

由于当前本地无法完整验证 Shaheen，代码需满足：

- Ray backend import/compile 通过。
- 不硬编码 Shaheen 路径。
- 通过环境变量获取 Ray 地址和缓存路径。
- 日志包含 Ray cluster CPU 数。
- Ray task failure 有 chunk id、`.bc` path、stderr 摘要。
- 支持后续在 Shaheen 直接 debug。

## 13. 默认参数

建议默认值：

```text
backend = local
max_seq_length = 2048
target_shard_size_bytes = 1073741824
max_parquet_files = 5000
resume = false
debug = false
cleanup = true
```

Ray/local worker 数：

- local 默认使用 `os.cpu_count()`
- Ray 默认根据 Ray cluster available CPUs 自适应

## 14. 关键设计决策总结

- 小文件默认不保留。
- IR 不进入最终 parquet。
- parquet 存 parsed CFG/DDG edge list，不存 dot。
- 任意字段缺失即函数失败。
- 分片以文件数限制优先，目标 1 GiB。
- 临时 parquet 也计入文件数限制。
- 默认临时 + 最终 parquet 不超过 5000。
- Resume 以 `.bc` 为单位。
- DataProcess.cli 只做 parquet 到 HF dataset。
- 不需要旧 SQLite/function_map 兼容。
- Ray 适配不硬编码，重点增加日志和 manifest。
