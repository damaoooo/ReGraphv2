from Pretrain.pretrain_dataset import MyFinalDataCollator, load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertForMaskedLM
import os


def main():
    
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_path)
    dataset_path = "/home/damaoooo/Downloads/regraphv2/IR/binary_save_truncated"
    dataset = load_dataset(dataset_path)
    
    my_collator = MyFinalDataCollator(tokenizer=tokenizer)
    
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    
    train_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        fp16=True,
        dataloader_num_workers=max(4, os.cpu_count() // 2),  # Use half of the available CPU cores
        torch_compile=True,  # Enable TorchScript compilation for performance
        logging_dir="./logs",
        learning_rate=5e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=500000,
        save_total_limit=3,
        # === 5. 日志与评估设置 (Logging & Evaluation) ===
        logging_strategy="steps",            # 按步数记录日志。
        logging_steps=100,                   # 每隔100步在控制台打印一次训练损失等信息。
        report_to="tensorboard",             # 将日志报告给TensorBoard。你也可以设置为 "wandb" 或 "comet_ml" 等。
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
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
    train_result = trainer.train(resume_from_checkpoint=True)
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