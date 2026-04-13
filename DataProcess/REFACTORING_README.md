# ASM Dataset Builder Refactoring

`DataProcess` 现在面向 ASM 训练数据，不再依赖 LLVM IR 预处理、DDG、CFG 或 `results.db`。

## 文件结构

1. `dataset_features.py`
Defines the Hugging Face dataset schema:
`file_path` + `input_ids`.

2. `processing_result.py`
Single-file processing result container.

3. `file_processor.py`
Reads a single `.asm` file and tokenizes it directly.

4. `parallel_processor.py`
Keeps the original processing modes:
sequential, batched, and streaming parallel parquet writing.

5. `dataset_utils.py`
Recursive ASM file discovery with `find_asm_files()`.

6. `dataset_builder_new.py`
Main entry point for scanning directories, resume support, caching, and saving outputs.

7. `cli.py`
Typer CLI for directory-based ASM dataset building.

## Current behavior

- Input directory: recursively scans `**/*.asm`
- No dependency on `results.db`
- No DDG or CFG generation
- Keeps:
  - tokenizer loading flow
  - batch / parallel processing modes
  - parquet bin writing style
  - `progress.txt` resume behavior
  - optional Hugging Face dataset export

## Example

```bash
python -m DataProcess.cli directory /path/to/asm_dir /path/to/output --parallel
```

## Notes

- `--no-cleanup` is retained for CLI compatibility, but ASM processing no longer creates temp files.
- `results.json` remains a summary output for non-streaming modes.
