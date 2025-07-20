
from transformers import (
    PreTrainedTokenizerFast,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from transformers.models.deberta_v2 import (
    DebertaV2ForMaskedLM,
    DebertaV2Model,
)
import torch
from typing import Optional
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Attention, DisentangledSelfAttention


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


def get_model(vocab_size: int, max_seq_len: int = 4096) -> DebertaV2ForMaskedLM:
    config = create_deberta_v3_config(vocab_size, max_seq_len)
    model = DebertaV2ForMaskedLM(config=config)

class BinDebertaV2Attention(DisentangledSelfAttention):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        pass