from Pretrain.pretrain_dataset import MyFinalDataCollator, load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertForMaskedLM
import os
from .pretrain_model import BinDebertaV2ModelForPretrain
import pickle
from torch.utils.data import DataLoader
from Tokenizer.ir_tokenizer import load_tokenizer
from Model.model_backbone import BinDebertaV2Model, create_deberta_v3_config


def debug_cpu():
    print("--- Running in debug mode ---")
    
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
    tokenizer = load_tokenizer(tokenizer_path)
    
    train_dataset_pool_path = "/home/damaoooo/Downloads/regraphv2/IR/train_dataset_pool"
    train_dataset_pool = load_dataset(train_dataset_pool_path)
    
    train_dataset_idx_path = '/home/damaoooo/Downloads/regraphv2/IR/train_task_dataset'
    train_dataset_idx = load_dataset(train_dataset_idx_path)
    
    train_dataset_map_path = '/home/damaoooo/Downloads/regraphv2/IR/train_positive_map.pkl'
    with open(train_dataset_map_path, 'rb') as f:
        train_positive_map = pickle.load(f)
    
    my_collator = MyFinalDataCollator(tokenizer=tokenizer, dataset_pool=train_dataset_pool, map_file=train_positive_map)
    
    model_config = create_deberta_v3_config(vocab_size=len(tokenizer.get_vocab()))
    model = BinDebertaV2ModelForPretrain(config=model_config)
    
    # 将模型设置为评估模式，这会关闭 dropout 等只在训练时使用的层
    model.train()
    
    # ---------------------------------------------------------------
    # 【补全部分】
    # ---------------------------------------------------------------
    
    # 1. 使用 PyTorch 的 DataLoader 来创建数据批次。
    #    这能很好地模拟 Trainer 内部的工作方式。
    #    - dataset: 我们用刚刚加载的 train_dataset_idx
    #    - batch_size: 设置一个小的批次大小，比如 4
    #    - collate_fn: 这是关键，我们使用自定义的 my_collator
    debug_dataloader = DataLoader(
        dataset=train_dataset_idx,
        batch_size=4,
        collate_fn=my_collator,
        shuffle=False # 调试时通常不需要打乱
    )

    # 2. 从 dataloader 中获取一个批次的数据
    print("Attempting to create one batch from the dataloader...")
    try:
        # 使用 iter 和 next 来安全地获取第一个（也是唯一一个我们需要的）批次
        one_batch = next(iter(debug_dataloader))
        print("Successfully created one batch.")
    except Exception as e:
        print(f"Error creating batch: {e}")
        # 如果数据整理器(collator)有错，这里通常会抛出异常
        return

    # 3. 检查批次数据的结构和形状 (非常重要的调试步骤)
    print("\n--- Batch Content (Keys and Shapes) ---")
    for key, value in one_batch.items():
        # one_batch 的 value 应该是 tensor
        print(f"Key: '{key}', Shape: {value.shape}, Dtype: {value.dtype}")
        
    # 4. 在 CPU 上运行模型的前向传播
    #    模型默认是在CPU上初始化的，所以不需要 .to('cpu')
    #    数据批次也应该在CPU上
    print("\n--- Running Model Forward Pass on CPU ---")
    try:
        # 使用 **one_batch 将字典解包为模型的关键字参数
        # e.g., model(input_ids=..., attention_mask=..., ...)
        outputs = model(**one_batch)
        
        print("Model forward pass successful!")
        
        # 5. 检查模型输出
        print("\n--- Model Output ---")
        print("Output keys:", outputs.keys())
        if 'loss' in outputs:
            print(f"Loss: {outputs['loss'].item()}") # .item() 从 tensor 中提取纯数值
        if 'logits' in outputs:
            print(f"Logits shape: {outputs.logits.shape}")
        outputs['loss'].backward()  # 计算梯度
        print("Backward pass successful!")

    except Exception as e:
        print(f"Error during model forward pass: {e}")
        # 如果模型内部逻辑有问题，例如维度不匹配，这里会报错
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈

    print("\n--- Debug mode finished ---")


def debug_gpu():
    print("--- Running in debug mode (GPU) ---")
    
    import torch
    
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run GPU debug.")
        return
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
    tokenizer = load_tokenizer(tokenizer_path)
    
    train_dataset_pool_path = "/home/damaoooo/Downloads/regraphv2/IR/train_dataset_pool"
    train_dataset_pool = load_dataset(train_dataset_pool_path)
    
    train_dataset_idx_path = '/home/damaoooo/Downloads/regraphv2/IR/train_task_dataset'
    train_dataset_idx = load_dataset(train_dataset_idx_path)
    
    train_dataset_map_path = '/home/damaoooo/Downloads/regraphv2/IR/train_positive_map.pkl'
    with open(train_dataset_map_path, 'rb') as f:
        train_positive_map = pickle.load(f)
    
    my_collator = MyFinalDataCollator(tokenizer=tokenizer, dataset_pool=train_dataset_pool, map_file=train_positive_map)
    
    model_config = create_deberta_v3_config(vocab_size=len(tokenizer.get_vocab()))
    model = BinDebertaV2ModelForPretrain(config=model_config)
    
    # 将模型移动到GPU并启用bf16
    model = model.to(device)
    
    # 启用gradient checkpointing以节省显存
    model.gradient_checkpointing_enable()
    
    # 启用bf16以节省显存
    from torch.cuda.amp import autocast
    model.train()
    
    print(f"Model moved to {device}")
    print("BF16 enabled for memory efficiency")
    
    # 使用 PyTorch 的 DataLoader 来创建数据批次
    debug_dataloader = DataLoader(
        dataset=train_dataset_idx,
        batch_size=1,  # 使用bf16后可以稍微增加批次大小
        collate_fn=my_collator,
        shuffle=False
    )

    # 从 dataloader 中获取一个批次的数据
    print("Attempting to create one batch from the dataloader...")
    try:
        one_batch = next(iter(debug_dataloader))
        print("Successfully created one batch.")
    except Exception as e:
        print(f"Error creating batch: {e}")
        return

    # 将批次数据移动到GPU
    print("\n--- Moving batch to GPU ---")
    try:
        one_batch_gpu = {}
        for key, value in one_batch.items():
            if torch.is_tensor(value):
                one_batch_gpu[key] = value.to(device)
            else:
                one_batch_gpu[key] = value
        print("Batch successfully moved to GPU")
    except Exception as e:
        print(f"Error moving batch to GPU: {e}")
        return

    # 检查批次数据的结构和形状
    print("\n--- Batch Content (Keys and Shapes) ---")
    for key, value in one_batch_gpu.items():
        if torch.is_tensor(value):
            print(f"Key: '{key}', Shape: {value.shape}, Dtype: {value.dtype}, Device: {value.device}")
        else:
            print(f"Key: '{key}', Type: {type(value)}")
        
    # 在 GPU 上运行模型的前向传播 (使用bf16)
    print("\n--- Running Model Forward Pass on GPU with BF16 ---")
    try:
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"GPU memory before forward pass: {memory_before:.2f} MB")
        
        # 使用autocast进行bf16计算
        with autocast(dtype=torch.bfloat16):
            outputs = model(**one_batch_gpu)
        
        print("Model forward pass successful with BF16!")
        
        # 检查模型输出
        print("\n--- Model Output ---")
        print("Output keys:", outputs.keys())
        if 'loss' in outputs:
            print(f"Loss: {outputs['loss'].item()}")
            print(f"Loss dtype: {outputs['loss'].dtype}")
        if 'logits' in outputs:
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Logits device: {outputs.logits.device}")
            print(f"Logits dtype: {outputs.logits.dtype}")
        
        # 反向传播 (loss会自动转换回float32进行梯度计算)
        outputs['loss'].backward()
        print("Backward pass successful!")
        
        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"GPU memory after forward+backward pass: {memory_after:.2f} MB")
            print(f"Memory increase: {memory_after - memory_before:.2f} MB")

    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理GPU内存
        torch.cuda.empty_cache()
        print("GPU cache cleared")

    print("\n--- GPU Debug mode finished ---")


def main():
    
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
    tokenizer = load_tokenizer(tokenizer_path)
    
    train_dataset_pool_path = "/home/damaoooo/Downloads/regraphv2/IR/train_dataset_pool"
    train_dataset_pool = load_dataset(train_dataset_pool_path)
    
    train_dataset_idx_path = '/home/damaoooo/Downloads/regraphv2/IR/train_task_dataset'
    train_dataset_idx = load_dataset(train_dataset_idx_path)
    
    train_dataset_map_path = '/home/damaoooo/Downloads/regraphv2/IR/train_positive_map.pkl'
    with open(train_dataset_map_path, 'rb') as f:
        train_positive_map = pickle.load(f)
    

    
    # dataset = dataset.remove_columns({
    #     "cfg_graph": "cfg_adj_list",
    #     "ddg_graph": "ddg_adj_list",
    # })
    my_collator = MyFinalDataCollator(tokenizer=tokenizer, dataset_pool=train_dataset_pool, map_file=train_positive_map)

    model_config = create_deberta_v3_config(vocab_size=len(tokenizer.get_vocab()))
    model = BinDebertaV2ModelForPretrain(config=model_config)
    
    train_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        fp16=False, # 关闭fp16
        bf16=True,  # 开启bf16
        remove_unused_columns=False,
        # dataloader_num_workers=max(4, os.cpu_count() // 2),  # Use half of the available CPU cores
        dataloader_num_workers=max(4, os.cpu_count() // 2),
        torch_compile=False,
        gradient_checkpointing=True,
        logging_dir="./logs",
        learning_rate=5e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=500000,
        save_total_limit=3,
        # === 5. 日志与评估设置 (Logging & Evaluation) ===
        logging_strategy="steps",            # 按步数记录日志。
        logging_steps=100,                   # 每隔100步在控制台打印一次训练损失等信息。
        report_to="tensorboard",             # 将日志报告给TensorBoard。你也可以设置为 "wandb" 或 "comet_ml" 等。
    )
    model.gradient_checkpointing_enable()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset_idx,
        data_collator=my_collator,
    )
    
    # 检查输出目录中是否存在检查点
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir):
    #     # 寻找最新的检查点文件夹
    #     checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    #     if checkpoints:
    #         last_checkpoint = os.path.join(training_args.output_dir, max(checkpoints, key=lambda x: int(x.split('-')[-1])))

    # 方式一：自动从最新的检查点恢复
    # 如果 output_dir 存在检查点，Trainer会自动从那里恢复
    # 如果你想确保是这样，可以传入 resume_from_checkpoint=True
    # train_result = trainer.train(resume_from_checkpoint=True)

    # 方式二：从指定的检查点恢复
    # train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 如果是从头开始训练
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    # --- 训练完成后 ---

    # 保存最终的模型、分词器和配置
    # 这会创建一个干净的、可以被 from_pretrained 加载的最终模型文件夹
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

    # 记录训练过程中的一些指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # 保存Trainer的状态，包括随机种子等


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "debug_cpu":
            debug_cpu()
        elif mode == "debug_gpu":
            debug_gpu()
        elif mode == "train":
            main()
        else:
            print("Usage: python run_pretrain.py [debug_cpu|debug_gpu|train]")
            print("  debug_cpu  - Run CPU debugging")
            print("  debug_gpu  - Run GPU debugging")
            print("  train      - Start training")
    else:
        # 默认运行CPU调试
        debug_gpu()