from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertModel

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError:  # pragma: no cover - validated in runtime tests
    GATv2Conv = None
    global_mean_pool = None

class GraphModernBertModel(ModernBertModel):
    """ModernBERT without implicit attention-backend switches."""

    def forward(self, *args, token_type_ids=None, **kwargs):
        # LLVM inputs are single sequences; ModernBERT has no segment embeddings.
        if kwargs.get("output_attentions", False):
            raise ValueError("Attention weight export is disabled on the ModernBERT branch.")
        return super().forward(*args, **kwargs)

    def compute_last_layer_attention_weights(
        self, input_ids, attention_mask=None, token_type_ids=None, inputs_embeds=None
    ):
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            return_dict=True,
        )
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
            raise ImportError("torch_geometric is required for the post-ModernBERT GATv2 graph branches.")

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


class ModernBertGraph(nn.Module):
    """ModernBERT encoder followed by two GATv2 branches (CFG + DDG).

    The text mean-pool, CFG graph embedding, and DDG graph embedding are
    concatenated into 3*hidden_size and projected back to hidden_size via
    a linear layer + LayerNorm before being returned as ``fused_feature``.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.encoder = GraphModernBertModel(config)

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
        return self.encoder.compute_last_layer_attention_weights(
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
        outputs = self.encoder(
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
