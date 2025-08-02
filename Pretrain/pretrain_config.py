"""
预训练配置文件
包含所有预训练相关的硬编码配置项
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class PretrainConfig:
    # === 序列长度配置 ===
    max_seq_length: int = 2048
    
    # === 路径配置 ===
    tokenizer_path: str = "/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
    train_dataset_pool_path: str = "/home/damaoooo/Downloads/regraphv2/IR/train_dataset_pool"
    train_dataset_idx_path: str = "/home/damaoooo/Downloads/regraphv2/IR/train_task_dataset"
    train_dataset_map_path: str = "/home/damaoooo/Downloads/regraphv2/IR/train_positive_map.pkl"
    output_dir: str = "./output"
    final_model_dir: str = "./final_model"
    logging_dir: str = "./logs"
    
    # === 模型配置 ===
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    relative_attention: bool = True
    pos_att_type: str = "p2c|c2p"
    torch_dtype: str = "bfloat16"
    use_flash_attn: bool = True
    
    # === 训练配置 ===
    num_train_epochs: int = 3  # 保留以防兼容性需要，但将被 max_steps 覆盖
    max_steps: int = 20000  # 最大训练步数，设置后将忽略 num_train_epochs
    per_device_train_batch_size: int = 2
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    optim: str = "paged_adamw_8bit"
    
    # === 保存配置 ===
    save_strategy: str = "steps"
    save_steps: int = 1000  # 每1000步保存一次检查点
    save_total_limit: int = 3
    
    # === 日志配置 ===
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: str = "tensorboard"
    
    # === 数据处理配置 ===
    dataloader_num_workers: Optional[int] = None  # 将在运行时根据CPU核心数自动设置
    remove_unused_columns: bool = False
    torch_compile: bool = False
    
    # === MLM配置 ===
    mlm: bool = True
    mlm_probability: float = 0.15
    edge_pad_value: int = -1
    pad_to_multiple_of: int = 8
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.dataloader_num_workers is None:
            import os
            self.dataloader_num_workers = max(4, os.cpu_count() // 2)

# 默认配置实例
DEFAULT_CONFIG = PretrainConfig()
