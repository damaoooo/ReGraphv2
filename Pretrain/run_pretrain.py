# file: pretrain/run_pretrain.py

import torch
from transformers import (
    PreTrainedTokenizerFast,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Import our own dataset function
from ir_dataset import load_and_tokenize_dataset

def create_deberta_v3_config(vocab_size: int, max_seq_len: int = 4096) -> DebertaV2Config:
    """
    Create a new DeBERTa V3 model config with the given vocab size and max sequence length.
    
    Args:
        vocab_size (int): Vocabulary size, must match the tokenizer.
        max_seq_len (int): Maximum sequence length supported by the model.

    Returns:
        DebertaV2Config: Configuration object for DeBERTa V3.
    """
    print(f"Creating DeBERTa V3 config: Vocab Size={vocab_size}, Max Length={max_seq_len}")
    
    # These parameters are for the "base" model; adjust as needed for your resources
    config = DebertaV2Config(
        vocab_size=vocab_size,
        max_position_embeddings=max_seq_len,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        relative_attention=True,
        pos_att_type="p2c|c2p",  # Standard for DeBERTa V2/V3
        torch_dtype="bfloat16",  # Recommended for training with Flash Attention
        use_flash_attn=True,  # Enable Flash Attention if supported
    )
    return config

def main():
    # --- 1. Define all paths and core parameters ---
    TOKENIZER_PATH = "/home/damaoooo/Downloads/regraphv2/DataProcess/small_sample_tokenizer/llvm_ir_bpe.json"
    CORPUS_DIR = "/home/damaoooo/Downloads/regraphv2/DataProcess/small_sample_corpus"
    OUTPUT_DIR = "output/deberta-v3-pretrained-4096"
    MAX_SEQ_LENGTH = 4096

    # --- 2. Load and configure Tokenizer ---
    print("Loading and configuring Tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    
    special_tokens_map = {
        'pad_token': '<pad>', 'unk_token': '<unk>', 'bos_token': '<bos>',
        'eos_token': '<eos>', 'mask_token': '<mask>',
        'additional_special_tokens': ["<func>", "<bb>", "<var>", "<const>"]
    }
    tokenizer.add_special_tokens(special_tokens_map)
    tokenizer.cls_token = tokenizer.bos_token
    tokenizer.sep_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}")

    # --- 3. Load and process dataset ---
    tokenized_dataset = load_and_tokenize_dataset(CORPUS_DIR, tokenizer, MAX_SEQ_LENGTH)

    # --- 4. Create model ---
    config = create_deberta_v3_config(vocab_size=len(tokenizer), max_seq_len=MAX_SEQ_LENGTH)
    
    print("Initializing DeBERTa V3 model from scratch...")
    model = DebertaV2ForMaskedLM(config=config)
    print(f"Model created! Number of parameters: {model.num_parameters():,}")

    # --- 5. Set up training ---
    print("Configuring training arguments and Trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,  # 4096 sequence length is very memory intensive!
        gradient_accumulation_steps=32, # Effective batch size = 2 * 32 = 64
        save_steps=10_000,
        save_total_limit=3,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        fp16=True if torch.cuda.is_available() else False, # Mixed precision training
        logging_steps=100, # More frequent logging

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # --- 6. Start training ---
    print("="*50)
    print("Everything is ready. Starting pretraining!")
    print("="*50)
    
    trainer.train()

    # --- 7. Save final model ---
    print("Training complete! Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}-final")
    print("All tasks completed!")

if __name__ == "__main__":
    main()
