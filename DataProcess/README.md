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
- 输出目录必须不存在或为空，输出使用 `datasets.load_from_disk()` 读取。
- `--num-processes/-p` 控制 worker 数，默认 CPU 核数；`-p 1` 使用相同处理函数，可用于定位失败样本。
- `--tokenizer-path/-t` 指定 tokenizer。
- 已删除 `--use-hf`、`--parallel`、`--batch-size`、`--no-cache`、`--no-cleanup`。不再支持 JSON 摘要输出、线程批处理或 PDWriter。
- 不删除 GraphBuilder 的图与 IR 产物，不再使用 `progress.txt` 或文件列表缓存。
- 当前不支持数据构建断点续跑；失败时不发布最终目录。修复问题后重跑，已有成功数据集需另选输出目录。
- 单样本处理返回失败时会跳过并报告数量；数据库、worker、保存异常向上传递，全部样本无效也会失败。

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
