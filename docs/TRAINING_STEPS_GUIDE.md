# 预训练轮数与 checkpoint 配置

## uv 环境

从仓库根目录执行。项目使用独立 `.venv`，不会修改原 Conda `ReLL` 环境。
当前锁定目标是 Linux x86_64、Python 3.13.11、Torch 2.10.0 cu130。
依赖以 `pyproject.toml` 和 `uv.lock` 为准；根目录 `requirements.txt` 是历史清单，不用于重建本次验证环境。

```bash
# 本机的 /usr/local/cuda 默认是 12.9，不能用于编译 Torch cu130 的扩展。
# 其他机器改为实际安装的兼容 CUDA 13.x Toolkit 路径。
CUDA_HOME=/usr/local/cuda-13.1 MAX_JOBS=8 uv sync --locked --extra fa2
uv run --locked --extra fa2 python -m Pretrain.run_pretrain train --help
uv run --locked --extra fa2 python -m pytest tests -q
```

训练和评估都保留 `--extra fa2`，否则默认同步可能移除这个可选依赖。
不使用 FlashAttention 时可以仅 `uv sync --locked`，并在训练中设置
`--set attention_backend=sdpa`。

FlashAttention 没有匹配 wheel 时需要系统 CUDA Toolkit、兼容的 C++ 编译器，
以及足够的编译内存；可以设置 `MAX_JOBS=4` 限制编译并发。
uv 不安装 NVIDIA 驱动。PyG 扩展固定从对应 Torch/CUDA 的官方 wheel 页面获取。
本机 RTX 5090 可以在构建时设置 `FLASH_ATTN_CUDA_ARCHS=120` 缩短编译时间，
但这样得到的 wheel 不应拿去给 RTX 4090 使用。多 GPU 架构部署保持上面的默认构建方式。
锁文件锁定 Python 包，不锁定系统驱动、编译器和 GPU 架构。

数据生成依赖使用 `--extra pipeline`；API 使用 `--extra api`；notebook 使用
`--extra notebook`。同时使用多个功能时同时传入相应 extra。
LLVM 18.1、CFG/DDG 插件、Graphviz、Ghidra/Java 和 IDA 安装及许可证仍单独管理。

CFG/DDG、MetaRemover 和函数提取工具现统一从 `LLVMTools/` 构建（需 LLVM 18.1、OpenSSL 开发包）：

```bash
cmake -S LLVMTools -B LLVMTools/build -DCMAKE_BUILD_TYPE=Release
cmake --build LLVMTools/build -j 4
```

Python 默认加载 `LLVMTools/build/lib/` 下的插件和 `LLVMTools/build/bin/split_funcs`。构建时的 LLVM 应与运行时 PATH 中的 `opt`、`llvm-extract` 一致。
IDA 的本地 Python wheel 不包含在公开锁文件中。

已有 shell 包装脚本默认使用 uv；支持 `PYTHON_CMD` 的脚本仍可显式指定其他解释器。
下文的裸 `python` 命令在激活 `.venv` 后执行，或加上
`uv run --locked --extra fa2` 前缀。

## 当前默认配置

- `num_train_epochs = 1`：默认训练 1 个 epoch。
- `max_steps = -1`：不再用固定 step 截断训练。
- `save_steps = 10000`：每 10K step 保存一次常规 checkpoint。
- `eval_steps = None`：有验证集时默认跟随 `save_steps`，也就是每 10K step 验证一次。
- `save_total_limit = None`：默认不删除历史 checkpoint。

训练结束时会额外维护 3 个命名 checkpoint：

- `checkpoint-last`：最后一次训练状态。
- `checkpoint-best-validation-loss`：validation loss 最低的 checkpoint。
- `checkpoint-best-train-loss`：train loss 最低的 checkpoint。

这些目录里会写入 `named_checkpoint_info.json`，记录来源 checkpoint、step 和对应 loss。

## 常用覆盖方式

通常不需要再设置 `max_steps`。如果想多训几轮，优先改 epoch：

```bash
python -m Pretrain.run_pretrain train --set num_train_epochs=2
```

如果只是想调整保存频率：

```bash
python -m Pretrain.run_pretrain train --set save_steps=20000
```

只有在做很短的 smoke test 时，才建议临时设置 `max_steps`：

```bash
python -m Pretrain.run_pretrain train --set max_steps=1000
```
