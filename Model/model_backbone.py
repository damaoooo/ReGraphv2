
from transformers import (
    PreTrainedTokenizerFast,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from collections.abc import Sequence

from transformers.utils import auto_docstring

from transformers.models.deberta_v2 import (
    DebertaV2ForMaskedLM,
    DebertaV2ForTokenClassification
)

from transformers.modeling_outputs import BaseModelOutput

import torch
import torch.nn as nn
from typing import Optional, Union
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DisentangledSelfAttention,
    DebertaV2Attention,
    DebertaV2Layer,
    DebertaV2Encoder,
    DebertaV2Model
)

@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)

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

class BinDebertaDisentangleV2Attention(DisentangledSelfAttention):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        
        self.ddg_bias_embedding = nn.Embedding(2, 1)
        self.num_cfg_buckets = 16
        self.cfg_bias_embedding = nn.Embedding(self.num_cfg_buckets + 1, 1)

    def forward(
        self,
        hidden_states,
        attention_mask,
        ddg_adj_matrix=None,
        cfg_adj_matrix=None,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None
    ):
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(query_layer, scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        attention_scores = attention_scores
        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
        )
        # My Own logic
        if cfg_adj_matrix is not None:
            bucket_indices = torch.zeros_like(cfg_adj_matrix, dtype=torch.long)
            no_edge_mask = cfg_adj_matrix == 0
            edge_mask = ~no_edge_mask
            edge_bucket_values = (cfg_adj_matrix[edge_mask] * (self.num_cfg_buckets - 1)).round().long() + 1
            bucket_indices[edge_mask] = edge_bucket_values
            cfg_bias = self.cfg_bias_embedding(bucket_indices).squeeze(-1).unsqueeze(1)
            attention_scores = attention_scores + cfg_bias

        if ddg_adj_matrix is not None:
            ddg_bias = self.ddg_bias_embedding(ddg_adj_matrix.long()).squeeze(-1).unsqueeze(1)
            attention_scores = attention_scores + ddg_bias

 

        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        # bsz x height x length x dimension
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if not output_attentions:
            return (context_layer, None)
        return (context_layer, attention_probs)
    

class BinDebertaV2Attention(DebertaV2Attention):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.self = BinDebertaDisentangleV2Attention(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        cfg_adj_matrix=None,
        ddg_adj_matrix=None,
        output_attentions: bool = False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_output, att_matrix = self.self(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            cfg_adj_matrix=cfg_adj_matrix,
            ddg_adj_matrix=ddg_adj_matrix,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return (attention_output, None)


class BinDebertaLV2Layer(DebertaV2Layer):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.attention = BinDebertaV2Attention(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        cfg_adj_matrix=None,
        ddg_adj_matrix=None,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        attention_output, att_matrix = self.attention(
            hidden_states,
            attention_mask,
            cfg_adj_matrix=cfg_adj_matrix,
            ddg_adj_matrix=ddg_adj_matrix,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return (layer_output, None)

class BinDebertaV2Encoder(DebertaV2Encoder):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([BinDebertaLV2Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        cfg_adj_matrix=None,
        ddg_adj_matrix=None,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states: Optional[tuple[torch.Tensor]] = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):
            output_states, attn_weights = layer_module(
                next_kv,
                attention_mask,
                cfg_adj_matrix=cfg_adj_matrix,
                ddg_adj_matrix=ddg_adj_matrix,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )



class BinDebertaV2Model(DebertaV2Model):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.encoder = BinDebertaV2Encoder(config)
        self.post_init()

    def _rasterize_graph(self, edge_list_tensor, batch_size, seq_len, device):
        adj_matrix = torch.zeros(batch_size, seq_len, seq_len, device=device)
        for batch in range(batch_size):
            for edge in edge_list_tensor[batch]:
                if len(edge) == 0:
                    continue
                if edge[0] == -1:
                    continue
                prob = 1
                if len(edge) == 4:
                    s_start, s_end, d_start, d_end = edge
                else:
                    assert len(edge) == 5
                    s_start, s_end, d_start, d_end, prob = edge
                adj_matrix[batch, int(s_start):int(s_end), int(d_start):int(d_end)] = prob
        return adj_matrix

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cfg_adj_list: Optional[torch.Tensor] = None,
        ddg_adj_list: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        """
        
        My New forward method for BinDebertaV2Model.

        This method processes the input data through the model, handling both
        input_ids and inputs_embeds, and applies attention mechanisms with
        optional graph adjacency matrices for CFG and DDG.

        Args:
            cfg_adj_list (Optional[torch.Tensor]): List of CFG adjacency matrices for each batch.
            ddg_adj_list (Optional[torch.Tensor]): List of DDG adjacency matrices for each batch.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        if cfg_adj_list is not None:
            cfg_adj_matrix = self._rasterize_graph(cfg_adj_list, input_shape[0], input_shape[1], device)
        else:
            cfg_adj_matrix = None

        if ddg_adj_list is not None:
            ddg_adj_matrix = self._rasterize_graph(ddg_adj_list, input_shape[0], input_shape[1], device)
        else:
            ddg_adj_matrix = None

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            cfg_adj_matrix=cfg_adj_matrix,
            ddg_adj_matrix=ddg_adj_matrix,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )

if __name__ == "__main__":
    # Small test to ensure the model can be created
    vocab_size = 65535
    model = BinDebertaV2Model(create_deberta_v3_config(vocab_size))
    
    # Generate dummy input
    input_ids = torch.randint(0, vocab_size, (2, 128))

    # Generate dummy ddg_adj_list and cfg_adj_list
    # in the ddg_adj_list, we assume each edge is represented as a tuple of (source_start, source_end, dest_start, dest_end)
    # and source_start < source_end and dest_start < dest_end
    ddg_adj_list = [
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),  # Batch 0
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])   # Batch 1
    ]

    # in the cfg_adj_list, we assume each edge is represented as a tuple of (source_start, source_end, dest_start, dest_end, probability)
    cfg_adj_list = [
        torch.tensor([[0, 1, 2, 3, 0.5], [1, 2, 3, 4, 0.8]]),  # Batch 0
        torch.tensor([[0, 1, 2, 3, 0.6], [1, 2, 3, 4, 0.9]])   # Batch 1
    ]

    model_output = model(
        input_ids=input_ids,
        ddg_adj_list=ddg_adj_list,
        cfg_adj_list=cfg_adj_list,
        attention_mask=torch.ones((2, 128), dtype=torch.bool)
    )

    print("Model output shape:", model_output.last_hidden_state.shape)

    # Try it on GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        input_ids = input_ids.to('cuda')
        ddg_adj_list = [adj.to('cuda') for adj in ddg_adj_list]
        cfg_adj_list = [adj.to('cuda') for adj in cfg_adj_list]
        
        model_output = model(
            input_ids=input_ids,
            ddg_adj_list=ddg_adj_list,
            cfg_adj_list=cfg_adj_list,
            attention_mask=torch.ones((2, 128), dtype=torch.bool).to('cuda')
        )
        
        print("Model output shape on GPU:", model_output.last_hidden_state.shape)
    else:
        print("CUDA is not available, running on CPU.")
        print("Model output shape on CPU:", model_output.last_hidden_state.shape)


