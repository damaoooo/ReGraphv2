from typing import Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertPreTrainedModel,
    ModernBertPredictionHead,
)

from Model.model_backbone import ModernBertGraph
from .pretrain_config import PretrainConfig


class ModernBertPretrainModel(ModernBertPreTrainedModel):
    config_class = PretrainConfig
    base_model_prefix = "graph_encoder"
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: PretrainConfig):
        if config.model_type != "modernbert":
            raise ValueError("This branch requires a ModernBERT checkpoint; RoFormer weights are incompatible.")
        super().__init__(config)
        self.graph_encoder = ModernBertGraph(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_size),
            nn.LayerNorm(config.embedding_size),
        )
        self.post_init()

    def tie_weights(self, **kwargs):
        """Tie the ModernBERT MLM decoder to the graph encoder's token table."""
        if not getattr(self.config, "tie_word_embeddings", True):
            return
        input_embeddings = self.get_input_embeddings()
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and input_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight

    def get_input_embeddings(self):
        return self.graph_encoder.encoder.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.graph_encoder.encoder.embeddings.tok_embeddings = value

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, value):
        self.decoder = value

    def encode(
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
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Inference-only embedding path.

        This bypasses the MLM head entirely and returns only the normalized
        contrastive embedding (plus attentions when explicitly requested).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = bool(kwargs.pop("output_attentions", False))

        encoder_out = self.graph_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ddg_node_spans=ddg_node_spans,
            ddg_node_batch=ddg_node_batch,
            ddg_edge_index=ddg_edge_index,
            cfg_node_spans=cfg_node_spans,
            cfg_node_batch=cfg_node_batch,
            cfg_edge_index=cfg_edge_index,
            cfg_edge_attr=cfg_edge_attr,
            output_attentions=output_attentions,
        )

        contrastive_embed = self.linear(encoder_out["fused_feature"])
        contrastive_embed = F.normalize(contrastive_embed, p=2, dim=1)

        if not return_dict:
            return (contrastive_embed,)

        result = {
            "embedding": contrastive_embed,
        }
        if output_attentions:
            result["attentions"] = encoder_out.get("attentions")
        return result

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
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = bool(kwargs.pop("output_attentions", False))

        encoder_out = self.graph_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ddg_node_spans=ddg_node_spans,
            ddg_node_batch=ddg_node_batch,
            ddg_edge_index=ddg_edge_index,
            cfg_node_spans=cfg_node_spans,
            cfg_node_batch=cfg_node_batch,
            cfg_edge_index=cfg_edge_index,
            cfg_edge_attr=cfg_edge_attr,
            output_attentions=output_attentions,
        )

        sequence_output = encoder_out["sequence_output"]
        fused_feature = encoder_out["fused_feature"]

        prediction_logits = self.decoder(self.head(sequence_output))

        mlm_loss = None
        if labels is not None:
            if (labels != -100).any():
                loss_fct = nn.CrossEntropyLoss()
                mlm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
            else:
                mlm_loss = prediction_logits.new_zeros(())

        contrastive_embed = self.linear(fused_feature)
        contrastive_embed = F.normalize(contrastive_embed, p=2, dim=1)

        if not return_dict:
            output = (prediction_logits, contrastive_embed)
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        result = {
            "loss": mlm_loss,
            "embedding": contrastive_embed,
        }
        if output_attentions:
            result["attentions"] = encoder_out.get("attentions")
        return result


@torch.no_grad()
def concat_all_gather(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class MoCoPretrainModel(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.config = config
        self.moco_buffer_size = config.moco_buffer_size
        self.moco_momentum = config.moco_momentum
        self.moco_temperature = config.moco_temperature

        self.encoder_q = ModernBertPretrainModel(config)
        self.encoder_k = ModernBertPretrainModel(config)
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        for (_, param_q), (_, param_k) in zip(self.encoder_q.named_parameters(), self.encoder_k.named_parameters()):
            if param_q.shape != param_k.shape:
                raise RuntimeError(f"Parameter size mismatch: {param_q.shape} vs {param_k.shape}")

        self.register_buffer("queue", torch.randn(config.embedding_size, self.moco_buffer_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_labels", torch.full((self.moco_buffer_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue: torch.Tensor
        self.queue_labels: torch.Tensor
        self.queue_ptr: torch.Tensor

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (1.0 - self.moco_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if batch_size > self.moco_buffer_size:
            raise RuntimeError(
                f"MoCo batch size ({batch_size}) exceeds queue size ({self.moco_buffer_size})."
            )

        end = ptr + batch_size
        if end <= self.moco_buffer_size:
            self.queue[:, ptr:end] = keys.T
            self.queue_labels[ptr:end] = labels
        else:
            first = self.moco_buffer_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue_labels[ptr:] = labels[:first]
            self.queue[:, : end - self.moco_buffer_size] = keys[first:].T
            self.queue_labels[: end - self.moco_buffer_size] = labels[first:]

        ptr = (ptr + batch_size) % self.moco_buffer_size
        self.queue_ptr[0] = ptr

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder_q.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder_q.gradient_checkpointing_disable()

    def forward(self, view1: Dict, view2: Dict, return_loss: bool = True):
        q_outputs = self.encoder_q(**view1)
        q_emb = q_outputs["embedding"]
        mlm_loss = q_outputs["loss"]

        with torch.no_grad():
            if self.training:
                self._momentum_update_key_encoder()
            k_outputs = self.encoder_k(**view2)
            k_emb = k_outputs["embedding"]

        l_pos = torch.einsum("nc,nc->n", [q_emb, k_emb]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q_emb, self.queue.clone().detach()])

        q_labels = view1["group_ids"].view(-1, 1)
        k_labels = self.queue_labels.view(1, -1)
        l_neg = l_neg.masked_fill(torch.eq(q_labels, k_labels), -1e9)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.moco_temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        contrastive_loss = nn.CrossEntropyLoss()(logits, labels)

        if self.training:
            self._dequeue_and_enqueue(k_emb, view2["group_ids"])

        total_loss = mlm_loss + contrastive_loss

        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss.item() if mlm_loss is not None else None,
            "contrastive_loss": contrastive_loss.item() if contrastive_loss is not None else None,
        }
