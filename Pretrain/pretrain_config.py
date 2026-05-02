"""
预训练配置文件
包含所有预训练相关的硬编码配置项
"""

from typing import Optional, Any, Dict
from transformers import RoFormerConfig


class PretrainConfig(RoFormerConfig):
    def __setattr__(self, name: str, value: Any):
        """Keep sequence length fields synchronized.

        The codebase uses `max_seq_length` for data truncation/padding, while
        RoFormer relies on `max_position_embeddings` internally. Divergence
        between them can cause hard-to-debug runtime issues.
        """
        if name in {"max_seq_length", "max_position_embeddings"}:
            synced_value = int(value)
            super().__setattr__("max_seq_length", synced_value)
            super().__setattr__("max_position_embeddings", synced_value)
            return
        super().__setattr__(name, value)

    def __init__(
        self,
        # === 序列长度配置 ===
        max_seq_length: int = 2048,
        # === 路径配置 ===
        tokenizer_path: str = "/home/damaoooo/Downloads/regraphv2/IR/dataset-1/train_corpus_tokenizer/llvm_ir_bpe.json",
        train_dataset_dir: str = "/home/damaoooo/Downloads/regraphv2/IR/dataset-1/train_final_set",
        train_dataset_pool_path: Optional[str] = None,
        train_dataset_idx_path: Optional[str] = None,
        train_dataset_map_path: Optional[str] = None,
        validation_dataset_dir: Optional[str] = None,
        validation_dataset_pool_path: Optional[str] = None,
        validation_dataset_idx_path: Optional[str] = None,
        validation_dataset_map_path: Optional[str] = None,
        output_dir: str = "./output",
        final_model_dir: str = "./db1_model",
        logging_dir: str = "./logs",
        # === 模型扩展配置 ===
        use_sdpa_attention: bool = True,
        use_cfg: bool = True,
        use_ddg: bool = True,
        embedding_size: int = 768,
        graph_hidden_size: Optional[int] = None,
        graph_layers: int = 1,
        graph_attention_heads: int = 4,
        graph_dropout: float = 0.1,
        # === 训练配置 ===
        num_train_epochs: int = 1,
        max_steps: int = -1,
        per_device_train_batch_size: int = 4,
        fp16: bool = False,
        bf16: bool = True,
        gradient_checkpointing: bool = True,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        optim: str = "paged_adamw_8bit",
        moco_buffer_size: int = 65536,
        moco_momentum: float = 0.999,
        moco_temperature: float = 0.07,
        # === 保存配置 ===
        save_strategy: str = "steps",
        save_steps: int = 10000,
        save_total_limit: Optional[int] = None,
        # === 日志配置 ===
        logging_strategy: str = "steps",
        logging_steps: int = 100,
        report_to: str = "tensorboard",
        # === 验证配置 ===
        do_eval: bool = True,
        eval_strategy: str = "steps",
        eval_steps: Optional[int] = None,
        per_device_eval_batch_size: Optional[int] = None,
        # === 断点续训配置 ===
        resume_from_checkpoint: bool = True,
        resume_checkpoint_path: Optional[str] = None,
        # === 异常样本跳过配置 ===
        skip_bad_batch: bool = True,
        bad_batch_log_path: Optional[str] = None,
        # === 数据处理配置 ===
        dataloader_num_workers: Optional[int] = None,
        remove_unused_columns: bool = False,
        torch_compile: bool = False,
        # === MLM配置 ===
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: int = 8,
        **kwargs: Any,
    ):
        model_defaults: Dict[str, Any] = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "relative_attention": True,
            "pos_att_type": "p2c|c2p",
            "torch_dtype": "bfloat16",
            "max_position_embeddings": max_seq_length,
        }
        model_defaults.update(kwargs)
        super().__init__(**model_defaults)

        # === 序列长度配置 ===
        self.max_seq_length = max_seq_length

        # === 路径配置 ===
        self.tokenizer_path = tokenizer_path
        self.train_dataset_dir = train_dataset_dir
        self.train_dataset_pool_path = train_dataset_pool_path
        self.train_dataset_idx_path = train_dataset_idx_path
        self.train_dataset_map_path = train_dataset_map_path
        self.validation_dataset_dir = validation_dataset_dir
        self.validation_dataset_pool_path = validation_dataset_pool_path
        self.validation_dataset_idx_path = validation_dataset_idx_path
        self.validation_dataset_map_path = validation_dataset_map_path
        self.output_dir = output_dir
        self.final_model_dir = final_model_dir
        self.logging_dir = logging_dir

        # === 模型扩展配置 ===
        self.use_sdpa_attention = use_sdpa_attention
        self.use_cfg = use_cfg
        self.use_ddg = use_ddg
        self.embedding_size = embedding_size
        self.graph_hidden_size = graph_hidden_size or self.hidden_size
        self.graph_layers = graph_layers
        self.graph_attention_heads = graph_attention_heads
        self.graph_dropout = graph_dropout

        # === 训练配置 ===
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.fp16 = fp16
        self.bf16 = bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optim = optim

        # === MoCo 配置 ===
        self.moco_buffer_size = moco_buffer_size
        self.moco_momentum = moco_momentum
        self.moco_temperature = moco_temperature

        # === 保存配置 ===
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit

        # === 日志配置 ===
        self.logging_strategy = logging_strategy
        self.logging_steps = logging_steps
        self.report_to = report_to

        # === 验证配置 ===
        self.do_eval = do_eval
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.per_device_eval_batch_size = per_device_eval_batch_size

        # === 断点续训配置 ===
        self.resume_from_checkpoint = resume_from_checkpoint
        self.resume_checkpoint_path = resume_checkpoint_path

        # === 异常样本跳过配置 ===
        self.skip_bad_batch = skip_bad_batch
        self.bad_batch_log_path = bad_batch_log_path

        # === 数据处理配置 ===
        if dataloader_num_workers is None:
            import os
            dataloader_num_workers = max(4, os.cpu_count() // 2)
        self.dataloader_num_workers = dataloader_num_workers
        self.remove_unused_columns = remove_unused_columns
        self.torch_compile = torch_compile

        # === MLM配置 ===
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of


# 默认配置实例
DEFAULT_CONFIG = PretrainConfig()
