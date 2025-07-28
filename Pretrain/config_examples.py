"""
示例：如何自定义预训练配置

你可以复制这个文件并修改参数来适应你的需求
"""

from Pretrain.pretrain_config import PretrainConfig

# 示例1：修改序列长度和批次大小
custom_config = PretrainConfig(
    max_seq_length=2048,  # 减少序列长度以节省内存
    per_device_train_batch_size=2,  # 增加批次大小
    learning_rate=1e-4,  # 调整学习率
    warmup_steps=500,  # 减少预热步数
)

# 示例2：修改模型结构
large_model_config = PretrainConfig(
    max_seq_length=4096,
    hidden_size=1024,  # 更大的隐藏层
    num_hidden_layers=24,  # 更多层数
    num_attention_heads=16,  # 更多注意力头
    intermediate_size=4096,  # 更大的中间层
    per_device_train_batch_size=1,  # 大模型需要更小的批次
)

# 示例3：修改路径配置
custom_path_config = PretrainConfig(
    tokenizer_path="/path/to/your/tokenizer.json",
    train_dataset_pool_path="/path/to/your/dataset_pool",
    train_dataset_idx_path="/path/to/your/task_dataset", 
    train_dataset_map_path="/path/to/your/positive_map.pkl",
    output_dir="/path/to/your/output",
    final_model_dir="/path/to/your/final_model",
)

# 示例4：调试配置（快速测试）
debug_config = PretrainConfig(
    max_seq_length=1024,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=100,  # 更频繁地保存
    logging_steps=10,  # 更频繁地记录日志
    warmup_steps=50,  # 较少的预热步数
)

# 在run_pretrain.py中使用自定义配置：
# if __name__ == "__main__":
#     import sys
#     
#     if len(sys.argv) > 1:
#         mode = sys.argv[1].lower()
#         if mode == "debug_cpu":
#             debug_cpu(debug_config)  # 使用调试配置
#         elif mode == "debug_gpu":
#             debug_gpu(debug_config)
#         elif mode == "train":
#             main(custom_config)  # 使用自定义配置
#     else:
#         debug_gpu(debug_config)
