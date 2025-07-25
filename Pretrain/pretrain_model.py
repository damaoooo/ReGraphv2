from Model.model_backbone import BinDebertaV2Model, create_deberta_v3_config
import torch
import torch.nn as nn
from typing import Optional, Union
from transformers import DebertaV2ForMaskedLM
import torch.nn.functional as F


class BinDebertaV2ModelForPretrain(DebertaV2ForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bindeberta = BinDebertaV2Model(config)
        self.contrastive_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.contractive_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        
        self.post_init()
        
        
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        contract_input_ids: Optional[torch.Tensor] = None,
        contract_attention_mask: Optional[torch.Tensor] = None,
        cfg_graph: Optional[torch.Tensor] = None,
        ddg_graph: Optional[torch.Tensor] = None,
        ):
        
        # MLM forward pass
        
        outputs = self.bindeberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cfg_adj_list=cfg_graph[:input_ids.shape[0], :, :] if len(cfg_graph.shape) > 2 else None,
            ddg_adj_list=ddg_graph[:input_ids.shape[0], :, :] if len(ddg_graph.shape) > 2 else None,
        )

        prediction_score_mlm = self.cls(outputs.last_hidden_state)
        loss_mlm = self.mlm_loss_fn(prediction_score_mlm.view(-1, self.config.vocab_size), labels.view(-1))
            
            
        # Contrastive forward pass
        contract_output = self.bindeberta(
            input_ids=contract_input_ids,
            attention_mask=contract_attention_mask,
            cfg_adj_list=cfg_graph if len(cfg_graph.shape) > 2 else None,
            ddg_adj_list=ddg_graph if len(ddg_graph.shape) > 2 else None,
        ).last_hidden_state
        
        # Shape of contract_output: [batch_size, seq_length, hidden_size]
        cls_output = contract_output[:, 0, :]  # Get the CLS token output
        batch_size = cls_output.size(0) // 3
        anchor_output = cls_output[:batch_size]
        positive_output = cls_output[batch_size:batch_size * 2]
        negative_output = cls_output[batch_size * 2:]
        # output size: [batch_size, hidden_size]
        anchor_proj = self.contrastive_head(anchor_output)
        positive_proj = self.contrastive_head(positive_output)
        negative_proj = self.contrastive_head(negative_output)
        
        anchor_proj = F.normalize(anchor_proj, p=2, dim=-1)
        positive_proj = F.normalize(positive_proj, p=2, dim=-1)
        negative_proj = F.normalize(negative_proj, p=2, dim=-1)
        
        # Compute contrastive loss
        loss_contrastive = self.contractive_loss_fn(anchor_proj, positive_proj, negative_proj)
        
        total_loss = loss_mlm + loss_contrastive
        return {
            "loss": total_loss,
            "mlm_loss": loss_mlm.item(),
            "contractive_loss": loss_contrastive.item(),
        }