# DataProcess

生产流程只保留一条：读取 GraphBuilder 的 `results.db`，生成 token/CFG/DDG，再保存 Hugging Face dataset。

```bash
uv run --locked --extra pipeline python -m DataProcess \
  IR/Dataset-2-IR IR/Dataset-2/db2_raw_dataset \
  --num-processes 32 \
  --tokenizer-path Tokenizer/output_tokenizer/llvm_ir_bpe.json
```

## 参数与输出

- 输入目录必须包含 `results.db`，其中记录的图与 IR 文件必须可读取。
- 每条记录必须有与当前输入、LLVM pass 和 `opt` 版本一致的 GraphBuilder signature；旧数据库、缺失签名或不匹配会使整个构建失败，不发布部分数据。先运行 `python -m GraphBuilder.graph_generator <输入目录>` 更新产物。构建期间不要同时重新生成输入或图；签名并不验证图文件本身是否被手工改动。
- 重建 LLVMTools 后需重跑 graph_generator：即使源码相同，重新编译也可能改变插件内容，使旧签名失效。构建数据集时需能访问生成图时对应的插件，并在 PATH 中提供 `opt`。
- 输出目录必须不存在或为空，输出使用 `datasets.load_from_disk()` 读取。
- `--num-processes/-p` 控制 worker 数，默认取本进程可用的 CPU（CPU 亲和性，设置了 `SLURM_CPUS_PER_TASK` 时以它为上限），不按整台节点的核数；`-p 1` 使用相同处理函数，可用于定位失败样本。
- 两个构建 CLI 入口默认设置 `USE_TORCH=0`，避免 tokenizer-only worker 加载 PyTorch；可显式设置 `USE_TORCH=1` 覆盖。普通 import 不修改此设置；自定义构建脚本需要在导入相关模块之前设置，并用 `if __name__ == '__main__'` 保护多进程入口。
- worker 并非越多越快，建议对比物理核数与逻辑核数；在途任务按块数有界，不代表内存字节数有固定上限。临时 Arrow 缓存与最终保存会同时占用磁盘，预留约两份数据的空间。
- `--tokenizer-path/-t` 指定 tokenizer。
- 已删除 `--use-hf`、`--parallel`、`--batch-size`、`--no-cache`、`--no-cleanup`。不再支持 JSON 摘要输出、线程批处理或 PDWriter。
- 不删除 GraphBuilder 的图与 IR 产物，不再使用 `progress.txt` 或文件列表缓存。
- 当前不支持数据构建断点续跑；失败时不发布最终目录。修复问题后重跑，已有成功数据集需另选输出目录。
- 单样本处理返回失败时会跳过并报告数量；数据库、worker、保存异常向上传递，全部样本无效也会失败。

## Dataset-2 完整流程（`Scripts/build_db2.sh`）

```bash
Scripts/build_db2.sh <task3 输出目录> <输出目录> [workers]
# 例：REGRAPH_PYTHON_CMD=.venv/bin/python Scripts/build_db2.sh IR/Dataset-2-IR IR/Dataset-2 32
```

- 前置条件：带 `pipeline` extra 的 Python 环境（默认 `uv run --project <repo> --locked --extra pipeline python`，可用 `REGRAPH_PYTHON_CMD` 换成其他解释器）、PATH 中的 LLVM 18 `opt`、已编译的 LLVMTools（`cmake -S LLVMTools -B LLVMTools/build -DCMAKE_BUILD_TYPE=Release && cmake --build LLVMTools/build -j 4`）。不依赖本机绝对路径；`TOKENIZER_PATH` 默认是仓库内的 tokenizer，并同时传给 DataProcess 与 wash。
- `workers` 省略时各阶段按本进程可用 CPU 自行决定（不使用 `nproc`，它受 `OMP_NUM_THREADS` 影响）；给出时传给 graph_generator 的 `--jobs`（含预筛选）、DataProcess 的 `--num-processes` 和 wash 的 `--num-proc`。
- graph_generator 失败策略：运行前用一个已知正确的函数检查 `opt` 与三个插件，失败时在改动任何结果之前退出并给出 opt 的报错；输出目录不可写、worker 异常、或本次尝试的函数全部失败且没有已完成的结果时，退出码为 1。其余被 pass 拒绝的单个函数会被跳过并列出（每次重跑都会重试，不算失败）。
- final_set（`Pretrain.split_train_validation`）：必须给出 `--naming`；`function_map.csv` 默认在每个函数文件所在目录查找，数据搬家后才需要 `--base-path`。符合命名规则的二进制缺少映射、或有函数不在映射中，会直接报错而不是丢弃样本；没有任何正样本对也会报错。不符合命名规则的二进制按策略丢弃并报告数量。
- 正样本是同一 origin 下的同名函数：dataset1 的 origin 保留架构和编译器，dataset2 保留架构，因此不会跨架构配对。

## 实现

- `__main__.py`：包入口，直接运行 `python -m DataProcess`。
- `cli.py`：命令行参数定义。
- `builder.py`：HF schema、按路径排序读取数据库、worker 初始化与有界调度、完整保存和最终发布。
- `file_processor.py`：单函数 token 与图生成，合法空图不等于失败。
- `processing_result.py`：供单文件处理器与构建器交换数据的结果类型。
- `dataset_utils.py`：仍供 GraphBuilder 使用，不属于已删除的输出模式。
- `wash.py`：独立的下游清洗与截断入口，运行 `python -m DataProcess.wash`；本次只改名，不改算法。
- 已删除无调用方的 `hf_dataset_player.py`；交互式查看直接使用 `datasets.load_from_disk()`。

新数据按输入路径确定行序，不再随 worker 完成顺序变化。重新生成后应重新生成 final_set、positive map 和 embedding，不能复用依赖旧行号的缓存。

## 回归

```bash
uv run --locked --extra pipeline python -m pytest tests/test_dataprocess_hf.py -q
```
