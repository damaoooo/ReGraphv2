# 预训练配置重构说明

## 概述

已经成功将预训练过程中的所有硬编码配置项提取到配置文件中，提高了代码的可维护性和灵活性。

## 修改的文件

### 1. 新增文件

#### `Pretrain/pretrain_config.py`
- 包含 `PretrainConfig` 数据类，定义了所有预训练相关的配置项
- 包含 `DEFAULT_CONFIG` 默认配置实例
- 配置项分类：
  - 序列长度配置
  - 路径配置  
  - 模型配置
  - 训练配置
  - 保存配置
  - 日志配置
  - 数据处理配置
  - MLM配置

#### `Pretrain/config_examples.py`
- 提供了多种配置示例
- 包含调试配置、大模型配置、自定义路径配置等

### 2. 修改的文件

#### `Pretrain/pretrain_dataset.py`
- `MyFinalDataCollator` 现在接受 `PretrainConfig` 参数
- 从配置中获取 MLM 相关参数和填充设置
- 向后兼容：如果不提供配置，使用默认值

#### `Pretrain/run_pretrain.py`
- 所有函数 (`debug_cpu`, `debug_gpu`, `main`) 现在接受 `PretrainConfig` 参数
- 使用配置中的路径而不是硬编码路径
- 使用 `create_deberta_v3_config_from_pretrain_config` 函数创建模型配置
- 训练参数全部从配置文件读取

#### `Model/model_backbone.py`
- 扩展了 `create_deberta_v3_config` 函数，支持更多参数
- 新增 `create_deberta_v3_config_from_pretrain_config` 函数
- 从 `PretrainConfig` 直接创建模型配置

#### `DataProcess/dataset_wash.py`
- 尝试导入配置，如果失败则使用硬编码的默认值
- 向后兼容现有的数据处理流程

## 使用方法

### 1. 使用默认配置
```python
# 使用默认配置运行
python run_pretrain.py train
```

### 2. 使用自定义配置
```python
from Pretrain.pretrain_config import PretrainConfig

# 创建自定义配置
my_config = PretrainConfig(
    max_seq_length=2048,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
)

# 在代码中使用
main(my_config)
```

### 3. 修改特定参数
```python
from Pretrain.pretrain_config import DEFAULT_CONFIG

# 复制默认配置并修改特定参数
config = DEFAULT_CONFIG
config.max_seq_length = 2048
config.per_device_train_batch_size = 4

debug_gpu(config)
```

## 主要改进

1. **集中管理**：所有配置项现在集中在一个文件中
2. **类型安全**：使用数据类提供类型注解
3. **向后兼容**：现有代码仍然可以工作
4. **灵活性**：可以轻松创建不同场景的配置
5. **可维护性**：修改配置不需要在多个文件中搜索

## 配置项说明

### 关键配置项
- `max_seq_length`: 最大序列长度（默认4096）
- `per_device_train_batch_size`: 每设备批次大小（默认1）
- `learning_rate`: 学习率（默认5e-5）
- `num_train_epochs`: 训练轮数（默认3）
- `save_steps`: 保存间隔步数（默认500000）

### 路径配置
所有路径都可以在配置文件中修改，包括：
- Tokenizer路径
- 数据集路径
- 输出目录
- 日志目录

### 模型配置
模型结构参数现在也可配置：
- `hidden_size`: 隐藏层大小
- `num_hidden_layers`: 隐藏层数量
- `num_attention_heads`: 注意力头数量
- `intermediate_size`: 中间层大小

这样的重构使得整个预训练流程更加灵活和易于维护！
