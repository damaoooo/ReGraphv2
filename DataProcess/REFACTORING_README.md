# Dataset Builder Code Refactoring

这个文件夹包含了重构后的 LLVM IR 数据集构建器代码。原来的 `dataset_builder.py` 文件超过1000行，现在被拆分成了更易管理的模块。

## 文件结构

### 核心模块

1. **`dataset_features.py`** - 数据集特征定义
   - 定义 HuggingFace 数据集的特征结构
   - 包含 `get_dataset_features()` 函数

2. **`processing_result.py`** - 处理结果数据类
   - 定义 `ProcessingResult` 数据类
   - 包含处理单个文件的结果信息

3. **`file_processor.py`** - 单文件处理逻辑
   - `FileProcessor` 类：处理单个 LLVM IR 文件
   - 包含 DDG、CFG 生成和标记化逻辑
   - 处理 opt 工具调用和临时文件清理

4. **`parallel_processor.py`** - 并行处理逻辑
   - `ParallelProcessor` 类：管理多进程处理
   - 包含顺序、批量和并行处理方法
   - HuggingFace 数据集创建的独立函数

5. **`dataset_utils.py`** - 工具函数
   - 文件搜索：`find_llvm_files()`
   - 临时文件清理：`cleanup_residual_temp_files()`
   - `FileFilter` 类：并行文件过滤

6. **`dataset_builder_new.py`** - 主数据集构建器
   - 重构后的 `DatasetBuilder` 类
   - 整合所有组件的主要接口
   - 向后兼容函数

7. **`cli.py`** - 命令行界面
   - Typer 基础的 CLI 实现
   - 所有命令行参数和选项

8. **`dataset_builder.py`** - 向后兼容模块
   - 提供向后兼容性
   - 重新导出所有重要组件
   - 遗留函数包装器

## 使用方式

### 使用新的模块化代码

```python
from dataset_builder_new import DatasetBuilder
from dataset_features import get_dataset_features
from file_processor import FileProcessor

# 创建数据集构建器
builder = DatasetBuilder(
    tokenizer=tokenizer,
    tokenizer_path=tokenizer_path,
    file_list=file_paths,
    # ... 其他参数
)

# 处理数据集
results = builder.process_dataset(output_path)
```

### 使用单个组件

```python
from file_processor import FileProcessor

# 只处理单个文件
processor = FileProcessor(
    tokenizer=tokenizer,
    ddg_so_path=ddg_so_path,
    purify_so_path=purify_so_path,
    cfg_so_path=cfg_so_path
)
result = processor.process_single_file(file_path)
```

### 命令行使用

```bash
# 使用新的 CLI
python cli.py directory /path/to/input /path/to/output.json

# 或者使用原来的方式（向后兼容）
python dataset_builder.py directory /path/to/input /path/to/output.json
```

## 重构的好处

1. **模块化**：每个文件都有明确的职责
2. **可维护性**：更小的文件更容易理解和修改
3. **可重用性**：组件可以独立使用
4. **可测试性**：更容易对单个组件进行单元测试
5. **向后兼容**：现有代码无需修改

## 迁移指南

现有的代码可以继续使用 `dataset_builder.py`，不需要立即迁移。如果要使用新的模块化代码：

1. 直接从相应模块导入需要的类
2. 使用 `dataset_builder_new.py` 中的 `DatasetBuilder` 类
3. 使用 `cli.py` 进行命令行操作

## 开发建议

- 新功能应该添加到相应的模块中
- 保持每个模块的单一职责
- 添加新功能时考虑向后兼容性
- 在 `dataset_builder.py` 中添加必要的导入和包装器
