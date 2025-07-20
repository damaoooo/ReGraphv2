import torch
from datasets import load_from_disk
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import sys
current_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
from Tokenizer.ir_tokenizer import load_tokenizer

# --- GPU 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name()}")

# --- 1. 加载您的数据集 ---
# 假设您的数据集保存在这个路径下
dataset_path = "/home/damaoooo/Downloads/regraphv2/IR/hf_save" 
try:
    raw_datasets = load_from_disk(dataset_path)
    # 如果您的数据集有不同的划分（如 'train', 'validation'），可以分别选择
    # raw_datasets = raw_datasets['train'] 
except FileNotFoundError:
    print(f"错误：在路径 '{dataset_path}'下找不到数据集。")
    print("请确保您已将数据集保存在该目录，或修改为正确的路径。")
    # 为了能继续运行示例，我们创建一个假的占位数据集
    from datasets import Dataset
    dummy_data = {
        'file_path': ['/path/to/file1.ll', '/path/to/file2.ll'],
        'ddg_graph': [[[1, 2]], [[3, 4], [5, 6]]],
        'cfg_graph': [[], []],
        'input_ids': [
            [2, 101, 2054, 2003, 2074, 102, 3],
            [2, 101, 2023, 2003, 1037, 2111, 2111, 2111, 2111, 2111, 2111, 2111, 2111, 2111, 102, 3] # 一个较长的例子
        ]
    }
    raw_datasets = Dataset.from_dict(dummy_data)


print("原始数据集示例:")
print(raw_datasets[0])


# --- 2. 准备 Tokenizer 和模型 ---
# 您需要一个与您的 input_ids 匹配的 Tokenizer。
# 如果您是自己训练的 Tokenizer，请从文件加载。
# 这里我们使用一个标准的 BERT Tokenizer 作为示例。
# 确保 PAD, MASK 等特殊 token 是您期望的。
tokenizer_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/output_tokenizer/llvm_ir_bpe.json" # 或者您自己的 tokenizer 路径
tokenizer = load_tokenizer(tokenizer_path)

# 定义模型的最大长度
MAX_LENGTH = 128 # 您可以根据您的数据和显存调整，例如 256, 512

# --- 3. 数据处理函数 (截断与填充) ---
# 这个函数会处理 input_ids，同时自动生成 attention_mask
def process_function(examples):
    # examples 是一个字典，key 是列名，value 是一个 list
    # 我们只对 input_ids 进行处理
    # truncation=True: 如果序列超过 MAX_LENGTH，就截断
    # padding='max_length': 如果序列不足 MAX_LENGTH，就用 [PAD] token 填充
    # max_length: 指定目标长度
    # return_tensors="pt": 返回 PyTorch tensors (如果用 TF 则为 "tf")
    
    # 因为您的 input_ids 已经是 token ID 列表了，我们直接传递给 tokenizer
    # tokenizer 会智能地处理它们，进行 padding 和 truncation
    # 注意：设置 is_split_into_words=True 是不正确的，因为输入已经是ID了
    # 我们需要手动处理
    
    processed_examples = {'input_ids': [], 'attention_mask': []}
    
    for ids in examples['input_ids']:
        # 1. 截断
        truncated_ids = ids[:MAX_LENGTH]
        
        # 2. 创建 attention mask
        attention_mask = [1] * len(truncated_ids)
        
        # 3. 填充
        padding_length = MAX_LENGTH - len(truncated_ids)
        padded_ids = truncated_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        processed_examples['input_ids'].append(padded_ids)
        processed_examples['attention_mask'].append(attention_mask)
        
    return processed_examples

# 使用 .map() 方法应用处理函数
# batched=True 让函数一次接收多个样本，处理更快
# remove_columns 会删除不再需要的原始列，让数据集更整洁
columns_to_remove = raw_datasets.column_names
processed_datasets = raw_datasets.map(
    process_function,
    batched=True,
    remove_columns=columns_to_remove
)

print("\n处理后的数据集示例 (已截断和填充):")
print(processed_datasets[0])
print(f"Input IDs 长度: {len(processed_datasets[0]['input_ids'])}")


# --- 4. 创建 Data Collator for Language Modeling ---
# 这是预训练的关键！它会自动为你做两件事：
# 1. 将一批数据整理成一个 PyTorch/TensorFlow 张量。
# 2. 随机地将 input_ids 中的一些 token 替换为 [MASK] token (这就是 MLM)。
# 3. 创建对应的 `labels`，其中未被 mask 的 token 会被设置为 -100，模型在计算损失时会忽略它们。
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15 # 掩码 15% 的 token，这是 BERT 的标准设置
)

# --- 5. 准备训练 ---
# 初始化您的 BERT 模型。要从头开始训练，请使用 BertConfig。
config = BertConfig(
    vocab_size=tokenizer.vocab_size, # 确保词汇表大小匹配
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=MAX_LENGTH # 最大位置编码应与您的数据长度匹配
)
model = BertForMaskedLM(config)

# 将模型移动到 GPU
model = model.to(device)
print(f"模型已移动到: {next(model.parameters()).device}")

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./bert_pretrain_output", # 模型和 checkpoints 的输出目录
    overwrite_output_dir=True,
    num_train_epochs=3, # 训练轮次
    per_device_train_batch_size=16, # 根据您的显存大小调整
    save_steps=10_000, # 每 N 步保存一次模型
    save_total_limit=2,
    prediction_loss_only=True, # 预训练时，我们只关心损失
    logging_steps=500, # 每 N 步记录一次日志
    # GPU 相关配置
    dataloader_pin_memory=True,  # 加速数据传输到 GPU
    fp16=torch.cuda.is_available(),  # 如果有 GPU 则使用混合精度训练
    gradient_accumulation_steps=1,  # 梯度累积步数
    remove_unused_columns=False,  # 保持数据完整性
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator, # 使用我们创建的 MLM data collator
    train_dataset=processed_datasets,
    # 如果有验证集，也可以传入 eval_dataset
    # eval_dataset=processed_datasets["validation"] 
)
if __name__ == "__main__":
    # 开始训练
    print("\n准备开始训练...")
    trainer.train() # 取消注释即可开始训练

    print("\n设置完成！您可以调用 `trainer.train()` 来启动预训练。")