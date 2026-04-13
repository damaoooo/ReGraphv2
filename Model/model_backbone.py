import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RoFormerConfig, RoFormerModel
from transformers.models.roformer.modeling_roformer import EncoderDecoderCache, RoFormerSelfAttention


class RoFormerSdpaSelfAttention(RoFormerSelfAttention):
    """RoFormer self-attention backed by scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        encoder_hidden_states=None,
        past_key_values=None,
        output_attentions=False,
        cache_position=None,
    ):
        if output_attentions:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                sinusoidal_pos=sinusoidal_pos,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

        batch_size, _, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        is_cross_attention = encoder_hidden_states is not None
        is_updated = False
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                curr_past_key_values = (
                    past_key_values.cross_attention_cache
                    if is_cross_attention
                    else past_key_values.self_attention_cache
                )
            else:
                curr_past_key_values = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            key_layer = curr_past_key_values.layers[self.layer_idx].keys
            value_layer = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_layer = (
                self.key(current_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )
            value_layer = (
                self.value(current_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )

            if not is_cross_attention and sinusoidal_pos is not None:
                if self.rotary_value:
                    query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer
                    )

            if past_key_values is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_values.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        attn_scale = 1.0 / math.sqrt(self.attention_head_size)
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=attn_scale,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer, None


class FastRoFormerModel(RoFormerModel):
    """Native RoFormer with SDPA-backed self-attention for the fast path."""

    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.use_sdpa_attention = bool(getattr(config, "use_sdpa_attention", True))
        if self.use_sdpa_attention:
            for layer_idx, layer in enumerate(self.encoder.layer):
                original_attention = layer.attention.self
                sdpa_attention = RoFormerSdpaSelfAttention(config, layer_idx=layer_idx)
                sdpa_attention.load_state_dict(original_attention.state_dict())
                layer.attention.self = sdpa_attention

    def compute_last_layer_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            return_dict=True,
        )
        if not outputs.attentions:
            raise ValueError("RoFormer attention weights are unavailable.")
        return outputs.attentions[-1].mean(dim=1)


class RoFormerEncoder(nn.Module):
    """Text-only RoFormer encoder used by ASM pretraining."""

    def __init__(self, config: RoFormerConfig):
        super().__init__()
        self.roformer = FastRoFormerModel(config)

    def compute_last_layer_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.roformer.compute_last_layer_attention_weights(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> dict:
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=output_attentions,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        pooled_feature = self.mean_pooling(sequence_output, attention_mask)

        result = {
            "sequence_output": sequence_output,
            "fused_feature": pooled_feature,
        }
        if output_attentions:
            result["attentions"] = outputs.attentions
        return result
