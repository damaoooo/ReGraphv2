import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RoFormerConfig, RoFormerModel
from transformers.models.roformer.modeling_roformer import EncoderDecoderCache, RoFormerSelfAttention

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError:  # pragma: no cover - validated in runtime tests
    GATv2Conv = None
    global_mean_pool = None


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
                if is_cross_attention:
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
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


class SpanGraphBranch(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        heads: int,
        dropout: float,
        num_layers: int,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        if GATv2Conv is None or global_mean_pool is None:
            raise ImportError("torch_geometric is required for the post-RoFormer GATv2 graph branches.")

        self.hidden_size = hidden_size
        self.input_proj = nn.Identity() if input_size == hidden_size else nn.Linear(input_size, hidden_size)
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_features.numel() == 0:
            return node_features.new_zeros((num_graphs, self.hidden_size))

        x = self.input_proj(node_features)
        for conv, norm in zip(self.convs, self.norms):
            conv_out = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + self.dropout(F.gelu(conv_out)))

        return global_mean_pool(x, node_batch, size=num_graphs)


class RoFormerGraph(nn.Module):
    """RoFormer encoder followed by two GATv2 branches (CFG + DDG).

    The text mean-pool, CFG graph embedding, and DDG graph embedding are
    concatenated into 3*hidden_size and projected back to hidden_size via
    a linear layer + LayerNorm before being returned as ``fused_feature``.
    """

    def __init__(self, config: RoFormerConfig):
        super().__init__()
        self.roformer = FastRoFormerModel(config)

        self.use_cfg = bool(getattr(config, "use_cfg", False))
        self.use_ddg = bool(getattr(config, "use_ddg", False))
        hidden_size = config.hidden_size

        if self.use_cfg:
            self.cfg_branch = SpanGraphBranch(
                input_size=hidden_size,
                hidden_size=hidden_size,
                heads=config.graph_attention_heads,
                dropout=config.graph_dropout,
                num_layers=config.graph_layers,
                edge_dim=1,
            )
        else:
            self.cfg_branch = None

        if self.use_ddg:
            self.ddg_branch = SpanGraphBranch(
                input_size=hidden_size,
                hidden_size=hidden_size,
                heads=config.graph_attention_heads,
                dropout=config.graph_dropout,
                num_layers=config.graph_layers,
                edge_dim=None,
            )
        else:
            self.ddg_branch = None

        self.feature_fusion = nn.Linear(3 * hidden_size, hidden_size)
        self.feature_norm = nn.LayerNorm(hidden_size)

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pooling utilities
    # ------------------------------------------------------------------

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def span_pooling(
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        node_spans: Optional[torch.Tensor],
        node_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if node_spans is None or node_batch is None or node_spans.numel() == 0:
            return sequence_output.new_zeros((0, sequence_output.size(-1)))

        masked_output = sequence_output * attention_mask.unsqueeze(-1).to(sequence_output.dtype)
        prefix = torch.cat(
            [masked_output.new_zeros(masked_output.size(0), 1, masked_output.size(-1)), masked_output.cumsum(dim=1)],
            dim=1,
        )

        starts = node_spans[:, 0].long()
        ends = node_spans[:, 1].long() + 1
        batches = node_batch.long()

        span_sum = prefix[batches, ends] - prefix[batches, starts]
        span_len = (ends - starts).clamp_min(1).unsqueeze(-1).to(sequence_output.dtype)
        return span_sum / span_len

    def _encode_branch(
        self,
        branch: Optional[SpanGraphBranch],
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        node_spans: Optional[torch.Tensor],
        node_batch: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = sequence_output.size(0)
        hidden_size = sequence_output.size(-1)
        device = sequence_output.device
        if branch is None:
            return sequence_output.new_zeros((batch_size, hidden_size), device=device)
        if node_spans is None or node_batch is None or edge_index is None:
            return sequence_output.new_zeros((batch_size, hidden_size), device=device)

        node_features = self.span_pooling(sequence_output, attention_mask, node_spans, node_batch)
        branch_edge_attr = edge_attr if edge_attr is not None and edge_attr.numel() > 0 else None
        return branch(node_features, edge_index, node_batch, batch_size, branch_edge_attr)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        ddg_node_spans: Optional[torch.LongTensor] = None,
        ddg_node_batch: Optional[torch.LongTensor] = None,
        ddg_edge_index: Optional[torch.LongTensor] = None,
        cfg_node_spans: Optional[torch.LongTensor] = None,
        cfg_node_batch: Optional[torch.LongTensor] = None,
        cfg_edge_index: Optional[torch.LongTensor] = None,
        cfg_edge_attr: Optional[torch.FloatTensor] = None,
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
        text_feature = self.mean_pooling(sequence_output, attention_mask)

        ddg_feature = self._encode_branch(
            self.ddg_branch,
            sequence_output,
            attention_mask,
            ddg_node_spans,
            ddg_node_batch,
            ddg_edge_index,
            None,
        )
        cfg_feature = self._encode_branch(
            self.cfg_branch,
            sequence_output,
            attention_mask,
            cfg_node_spans,
            cfg_node_batch,
            cfg_edge_index,
            cfg_edge_attr,
        )

        fused = torch.cat([text_feature, ddg_feature, cfg_feature], dim=-1)
        fused_feature = self.feature_norm(self.feature_fusion(fused))

        result = {
            "sequence_output": sequence_output,
            "fused_feature": fused_feature,
        }
        if output_attentions:
            result["attentions"] = outputs.attentions
        return result
