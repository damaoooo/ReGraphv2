# Pipeline Scripts Usage Guide

原来的 `lift_dataset1.py` 脚本已经被拆分成了4个独立的Python文件，每个都支持断点续传功能。

## 文件说明

- `utils.py` - 通用工具函数
- `task1_lift.py` - 任务1：二进制文件提升到LLVM IR  
- `task2_reoptimize.py` - 任务2：重新优化LLVM IR文件
- `task3_extract.py` - 任务3：提取单个函数
- `pipeline.py` - 主控制脚本，可以运行单个任务或完整流水线

## 使用方法

### 1. 运行完整流水线（推荐）

```bash
# 运行完整流水线
python3 pipeline.py pipeline --output /path/to/output

# 使用Dataset-1作为输入
python3 pipeline.py pipeline --db1 --output /path/to/output

# 自定义输入路径
python3 pipeline.py pipeline --input-path /path/to/input --output /path/to/output

# 断点续传
python3 pipeline.py pipeline --output /path/to/output --resume

# 从第2个任务开始运行
python3 pipeline.py pipeline --output /path/to/output --start-from 2

# 自定义工作进程数
python3 pipeline.py pipeline --output /path/to/output --workers 8
```

### 2. 运行单个任务

#### 任务1：二进制文件提升到LLVM IR
```bash
# 基本用法
python3 pipeline.py task1 --output /path/to/output

# 自定义输入路径
python3 pipeline.py task1 --input-path /path/to/binaries --output /path/to/output

# 断点续传
python3 pipeline.py task1 --output /path/to/output --resume

# 直接运行任务脚本
python3 task1_lift.py --output /path/to/output
```

#### 任务2：重新优化LLVM IR文件  
```bash
# 通过pipeline运行
python3 pipeline.py task2 --input-path /path/to/ll/files

# 直接运行任务脚本
python3 task2_reoptimize.py --input-path /path/to/ll/files --resume
```

#### 任务3：提取单个函数
```bash
# 通过pipeline运行
python3 pipeline.py task3 --input-path /path/to/bc/files

# 直接运行任务脚本
python3 task3_extract.py --input-path /path/to/bc/files --resume
```

## 主要改进

1. **模块化设计**：每个任务都是独立的脚本，可以单独运行
2. **断点续传**：所有任务都支持 `--resume` 参数，跳过已经处理的文件
3. **更好的错误处理**：每个任务都有独立的错误处理和统计
4. **灵活的控制**：可以从任意任务开始运行流水线
5. **清晰的进度显示**：每个任务都有独立的进度条
6. **更好的日志**：任务1有独立的日志文件 `lift_task1_log.txt`

## 典型使用场景

### 场景1：首次运行完整流水线
```bash
python3 pipeline.py pipeline --db1 --output /path/to/output --workers 16
```

### 场景2：中途中断后继续运行
```bash
python3 pipeline.py pipeline --db1 --output /path/to/output --resume
```

### 场景3：只想重新优化已有的.ll文件
```bash
python3 pipeline.py task2 --input-path /path/to/existing/ll/files --resume
```

### 场景4：从任务2开始运行（任务1已完成）
```bash
python3 pipeline.py pipeline --db1 --output /path/to/output --start-from 2
```

这样的设计让你可以更灵活地控制处理流程，并且在遇到问题时可以轻松地从中断点继续。
