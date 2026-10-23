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
        # Keep registered module names stable for existing checkpoints.
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_size),
            nn.LayerNorm(config.embedding_size),
        )
        self.post_init()

    @property
    def mlm_transform(self) -> ModernBertPredictionHead:
        """Per-token feature transform before vocabulary prediction."""
        return self.head

    @property
    def mlm_decoder(self) -> nn.Linear:
        """Vocabulary projection, optionally tied to input embeddings."""
        return self.decoder

    @property
    def embedding_projection(self) -> nn.Sequential:
        """Project fused text/graph features into the retrieval space."""
        return self.linear

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

    def forward_embedding(
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
        """Return retrieval embeddings without computing MLM predictions.

        Used by the MoCo key branch and retrieval inference. Callers control
        training mode and gradient tracking; this method changes neither.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = bool(kwargs.pop("output_attentions", False))

        encoder_outputs = self.graph_encoder(
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

        retrieval_embedding = self.embedding_projection(encoder_outputs["fused_feature"])
        retrieval_embedding = F.normalize(retrieval_embedding, p=2, dim=1)

        if not return_dict:
            return (retrieval_embedding,)

        result = {
            "embedding": retrieval_embedding,
        }
        if output_attentions:
            result["attentions"] = encoder_outputs.get("attentions")
        return result

    def encode(self, *args, **kwargs):
        """Compatibility entry point; use forward_embedding in new code."""
        return self.forward_embedding(*args, **kwargs)

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

        encoder_outputs = self.graph_encoder(
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

        sequence_output = encoder_outputs["sequence_output"]
        fused_feature = encoder_outputs["fused_feature"]

        # Preserve full logits for tuple callers; training only needs MLM targets.
        mlm_logits = None
        if not return_dict:
            mlm_logits = self.mlm_decoder(self.mlm_transform(sequence_output))
        mlm_loss = None
        if labels is not None:
            mlm_positions = labels != -100
            if mlm_positions.any():
                if return_dict:
                    masked_mlm_logits = self.mlm_decoder(
                        self.mlm_transform(sequence_output[mlm_positions])
                    )
                    mlm_loss = F.cross_entropy(masked_mlm_logits, labels[mlm_positions])
                else:
                    mlm_loss = F.cross_entropy(
                        mlm_logits.view(-1, self.config.vocab_size), labels.view(-1)
                    )
            else:
                mlm_loss = sequence_output.new_zeros(())

        retrieval_embedding = self.embedding_projection(fused_feature)
        retrieval_embedding = F.normalize(retrieval_embedding, p=2, dim=1)

        if not return_dict:
            output = (mlm_logits, retrieval_embedding)
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        result = {
            "loss": mlm_loss,
            "embedding": retrieval_embedding,
        }
        if output_attentions:
            result["attentions"] = encoder_outputs.get("attentions")
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

        # Query learns by backpropagation; key tracks it via momentum updates.
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

    def compile_encoders(self, **compile_kwargs):
        """torch.compile the two encoders; the MoCo queue update in forward stays eager.

        Compiling the whole MoCo model fails: forward updates queue_labels in place while the compiled
        backward still reads it. Assigning instance attributes keeps parameter names, so checkpoints
        save and load as before.
        """
        # Both encoders run the same layer code, with and without grad, in training and in evaluation,
        # on varying shapes: the default limit of 8 graphs per function sends some of them back to eager
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
        self.encoder_q.forward = torch.compile(self.encoder_q.forward, **compile_kwargs)
        self.encoder_k.forward_embedding = torch.compile(self.encoder_k.forward_embedding, **compile_kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder_q.gradient_checkpointing_disable()

    def forward(self, view1: Dict, view2: Dict, return_loss: bool = True):
        """Use a masked query view and an unmasked positive key view."""
        query_outputs = self.encoder_q(**view1)
        query_embedding = query_outputs["embedding"]
        mlm_loss = query_outputs["loss"]

        with torch.no_grad():
            if self.training:
                self._momentum_update_key_encoder()
            key_outputs = self.encoder_k.forward_embedding(**view2)
            key_embedding = key_outputs["embedding"]

        positive_logits = torch.einsum("nc,nc->n", [query_embedding, key_embedding]).unsqueeze(-1)
        negative_logits = torch.einsum("nc,ck->nk", [query_embedding, self.queue.clone().detach()])

        query_group_ids = view1["group_ids"].view(-1, 1)
        queue_group_ids = self.queue_labels.view(1, -1)
        negative_logits = negative_logits.masked_fill(torch.eq(query_group_ids, queue_group_ids), -1e9)

        contrastive_logits = torch.cat([positive_logits, negative_logits], dim=1)
        contrastive_logits /= self.moco_temperature

        # Column zero holds the positive key; other columns hold queued negatives.
        positive_targets = torch.zeros(
            contrastive_logits.shape[0], dtype=torch.long, device=contrastive_logits.device
        )
        contrastive_loss = F.cross_entropy(contrastive_logits, positive_targets)

        if self.training:
            self._dequeue_and_enqueue(key_embedding, view2["group_ids"])

        total_loss = mlm_loss + contrastive_loss

        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss.item() if mlm_loss is not None else None,
            "contrastive_loss": contrastive_loss.item() if contrastive_loss is not None else None,
        }
