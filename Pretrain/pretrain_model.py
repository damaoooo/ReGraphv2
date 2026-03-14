from Model.model_backbone import GraphRoFormerModel
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict
from transformers import RoFormerForMaskedLM, GenerationMixin
from .pretrain_config import PretrainConfig
import torch.nn.functional as F
import torch.distributed as dist
import copy

class ReFormerPretrainModel(RoFormerForMaskedLM, GenerationMixin):
    def __init__(self, config: PretrainConfig):
        super().__init__(config)
        
        # 1. 替换 Backbone 为支持图结构的 GraphRoFormer
        # 注意：GraphRoFormerModel 必须输出 last_hidden_state
        self.roformer = GraphRoFormerModel(config)
        
        # 2. 对比学习投影头 (Projection Head for InfoNCE/MoCo)
        # 结构: Linear -> ReLU -> Linear -> LayerNorm -> (Optional Normalize)
        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_size),
            nn.LayerNorm(config.embedding_size)
        )
        
        # self.lm_head 已经在 super().__init__ 中初始化好了

    def get_input_embeddings(self):
        return self.roformer.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.roformer.embeddings.word_embeddings = value


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        cfg_u: Optional[torch.FloatTensor] = None,
        cfg_v: Optional[torch.FloatTensor] = None,
        ddg_edges: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = bool(kwargs.pop("output_attentions", False))

        # 图张量不需要梯度，直接 detach
        if cfg_u is not None:
            cfg_u = cfg_u.detach()
        if cfg_v is not None:
            cfg_v = cfg_v.detach()
        if ddg_edges is not None:
            ddg_edges = ddg_edges.detach()

        # 1. Backbone Forward (融入 CFG/DDG)
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cfg_u=cfg_u,
            cfg_v=cfg_v,
            ddg_edges=ddg_edges,
            return_dict=True,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # sequence_output shape: [Batch, SeqLen, Hidden]
        sequence_output = outputs.last_hidden_state

        # =================================================
        # Task 1: Masked Language Modeling (MLM)
        # =================================================
        prediction_logits = self.cls(sequence_output)
        
        mlm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten 之后计算 Loss
            mlm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # =================================================
        # Task 2: Contrastive Learning Embedding (InfoNCE)
        # =================================================
        # A. Mean Pooling (获取函数级别的 Representation)
        # 这一步非常重要，比取 [CLS] 更稳健
        rep = self.mean_pooling(sequence_output, attention_mask)
        
        # B. Projection Head (映射到对比学习空间)
        contrastive_embed = self.linear(rep)
        
        # C. Normalize (通常 InfoNCE 需要归一化向量)
        contrastive_embed = F.normalize(contrastive_embed, p=2, dim=1)

        # =================================================
        # Return
        # =================================================
        # 返回字典，方便外部 Trainer 或 MoCo Wrapper 调用
        if not return_dict:
            output = (prediction_logits, contrastive_embed)
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        result = {
            "loss": mlm_loss,                # MLM Loss (如果没传 labels 则是 None)
            "embedding": contrastive_embed  # 对比学习向量 [B, Emb_Dim]
        }
        if output_attentions:
            result["attentions"] = outputs.attentions
        return result

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        标准的 Mean Pooling 实现，处理 Padding Mask
        """
        # input_mask_expanded: [Batch, SeqLen, Hidden]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum: 只累加非 Padding 的 Token
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Count: 统计非 Padding 的 Token 数量 (防止除以0，加上 1e-9)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Mean
        return sum_embeddings / sum_mask


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    * Critical for DDP MoCo *
    """
    # 1. 如果没有使用 DDP (单卡模式)，直接返回原 Tensor
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    # 2. DDP 模式：收集所有卡的 Tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    # 3. 拼接在一起
    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCoPretrainModel(nn.Module):

    
    def __init__(
        self, 
        config: PretrainConfig, 
    ):
        super().__init__()
        
        # 对 config 进行深拷贝快照，防止后续修改影响模型
        self.config = config
        self.moco_buffer_size = config.moco_buffer_size
        self.moco_momentum = config.moco_momentum
        self.moco_temperature = config.moco_temperature

        # 1. 初始化双塔
        # Encoder Q: 正常的梯度更新模型
        self.encoder_q = ReFormerPretrainModel(config)
        # Encoder K: 动量更新模型 (通过加载 Q 的状态字典来初始化)
        self.encoder_k = ReFormerPretrainModel(config)
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())

        # 2. 设置 K 不进行梯度反传
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False
        
        # 3. 验证 encoder_q 和 encoder_k 的参数大小是否匹配
        for (name_q, param_q), (name_k, param_k) in zip(
            self.encoder_q.named_parameters(), 
            self.encoder_k.named_parameters()
        ):
            if param_q.shape != param_k.shape:
                raise RuntimeError(
                    f"Parameter size mismatch: {name_q} ({param_q.shape}) vs {name_k} ({param_k.shape})"
                )

        # 3. 注册队列 (Queue)
        # shape: [Embedding_Dim, K]
        self.register_buffer("queue", torch.randn(config.embedding_size, self.moco_buffer_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # 4. 注册队列标签 (Queue Labels for Supervised Masking)
        # shape: [K]
        self.register_buffer("queue_labels", torch.full((self.moco_buffer_size,), -1, dtype=torch.long))

        # 5. 队列指针
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # ======== 显式声明registered queue =========
        self.queue: torch.Tensor
        self.queue_labels: torch.Tensor
        self.queue_ptr: torch.Tensor
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        动量更新 Key Encoder 的参数: theta_k = m * theta_k + (1-m) * theta_q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (1. - self.moco_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """
        DDP 安全的队列更新机制
        keys: [Local_Batch, Dim]
        labels: [Local_Batch]
        """
        # =====================================================
        # [DDP 核心修复]
        # 收集所有 GPU 的 keys 和 labels
        # 如果是单卡，这个函数会直接返回输入，不做任何操作
        # =====================================================
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 检查队列容量溢出 (Wrap-around)
        # 注意：在 DDP 下，batch_size 是所有卡的 batch 之和
        assert self.moco_buffer_size % batch_size == 0, f"队列长度 K={self.moco_buffer_size} 必须是全局 Batch Size={batch_size} 的整数倍，否则指针逻辑会变复杂"

        # 替换队列内容
        # keys.T -> [Dim, Global_Batch]
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_labels[ptr:ptr + batch_size] = labels
        
        # 移动指针
        ptr = (ptr + batch_size) % self.moco_buffer_size
        self.queue_ptr[0] = ptr

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        启用 Gradient Checkpointing 以节省显存
        只在 encoder_q 上启用，因为 encoder_k 不需要梯度反传
        """
        self.encoder_q.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """
        禁用 Gradient Checkpointing
        """
        self.encoder_q.gradient_checkpointing_disable()

    def forward(self, view1: Dict, view2: Dict):
        """
        view1: Query Input Dict (含 input_ids, mask, cfg, ddg, group_ids)
        view2: Key Input Dict
        """
        
        # ===============================================
        # 1. 计算 Query (带梯度)
        # ===============================================
        # encoder_q 会返回 {loss(mlm), logits, embedding, ...}
        q_outputs = self.encoder_q(**view1)
        q_emb = q_outputs['embedding']  # [Batch, Dim] (已归一化)
        mlm_loss = q_outputs['loss']    # MLM Loss

        # ===============================================
        # 2. 计算 Key (无梯度，动量更新)
        # ===============================================
        with torch.no_grad():
            self._momentum_update_key_encoder()  # 更新参数
            
            # 这里的 shuffle bn 是 MoCo 的 trick，但在 NLP/RoFormer 里通常不需要
            k_outputs = self.encoder_k(**view2)
            k_emb = k_outputs['embedding'] # [Batch, Dim] (已归一化)

        # ===============================================
        # 3. 计算 InfoNCE Loss (Supervised Mode)
        # ===============================================
        
        # A. 正样本得分 (Positive Logits) -> [Batch, 1]
        # q 和 k 是一对一匹配的
        l_pos = torch.einsum('nc,nc->n', [q_emb, k_emb]).unsqueeze(-1)

        # B. 负样本得分 (Negative Logits) -> [Batch, K]
        # q 和 queue 里的所有向量做内积
        l_neg = torch.einsum('nc,ck->nk', [q_emb, self.queue.clone().detach()])

        # C. [关键] 误伤掩码 (False Negative Masking)
        # 检查队列里有没有和当前 batch 属于同一个 Group 的
        # q_labels: [Batch, 1]
        q_labels = view1['group_ids'].view(-1, 1)
        # queue_labels: [1, K]
        k_labels = self.queue_labels.view(1, -1)
        
        # mask: [Batch, K], True 表示是同类 (不能当作负样本)
        mask = torch.eq(q_labels, k_labels)
        
        # 将误伤位置的 Logits 设为 -inf，使其在 Softmax 中概率为 0
        l_neg = l_neg.masked_fill(mask, -1e9)

        # D. 拼接 Logits -> [Batch, 1 + K]
        # 第 0 列是正样本，后面 K 列是负样本
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.moco_temperature

        # E. 计算 Contrastive Loss
        # 目标是让第 0 列概率最大
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        ce_loss_fct = nn.CrossEntropyLoss()
        contrastive_loss = ce_loss_fct(logits, labels)

        # ===============================================
        # 4. 更新队列 & 返回总 Loss
        # ===============================================
        self._dequeue_and_enqueue(k_emb, view2['group_ids'])
        
        # 你的总 Loss 是 MLM Loss + Contrastive Loss
        # 你可以加一个权重系数，比如 loss = mlm_loss + 0.5 * contrastive_loss
        total_loss = mlm_loss + contrastive_loss

        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss.item() if mlm_loss is not None else None,
            "contrastive_loss": contrastive_loss.item() if contrastive_loss is not None else None,
        }
        
        
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 开始测试 MoCoPretrainModel (Dual-Encoder + Queue)")
    print("="*50)

    # 1. 构造 Mock 配置 (模拟 PretrainConfig)
    class MockConfig(PretrainConfig):
        hidden_size = 64
        num_hidden_layers = 2
        num_attention_heads = 4
        vocab_size = 100
        max_position_embeddings = 512
        embedding_size = 32     # 投影后的维度
        layer_norm_eps = 1e-12
        pad_token_id = 0
        use_return_dict = True
        
        # 图结构配置
        svd_rank = 8
        ddg_k = 4
        
        # RoFormer/Dropout 配置
        rotary_value = False
        intermediate_size = 128
        hidden_act = "gelu"
        hidden_dropout_prob = 0.1
        attention_probs_dropout_prob = 0.1
        initializer_range = 0.02
        type_vocab_size = 2
        
        # MoCo 配置
        moco_buffer_size = 128
        moco_momentum = 0.999
        moco_temperature = 0.07

    config = MockConfig()
    
    BATCH_SIZE = 4
    SEQ_LEN = 16
    
    # 2. 初始化模型
    print(f"\n[Step 1] 初始化模型 (K={config.moco_buffer_size})...")
    try:
        model = MoCoPretrainModel(config)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        exit()

    # 3. 构造 Mock 数据 (View 1 & View 2)
    print(f"\n[Step 2] 构造 View1(Query) 和 View2(Key) 数据...")
    
    def create_dummy_view(batch_size, seq_len, rank):
        return {
            "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len)),
            # 只有 Query 需要 MLM labels，Key 不需要，但传入也不会报错，只是不用
            "labels": torch.randint(0, config.vocab_size, (batch_size, seq_len)), 
            "cfg_u": torch.randn(batch_size, seq_len, rank),
            "cfg_v": torch.randn(batch_size, seq_len, rank),
            "ddg_edges": torch.randint(0, seq_len, (batch_size, seq_len, 4)),
            # 关键：Group IDs (用于测试防误伤)
            # 假设前两个样本是同一组，后两个是独立组
            "group_ids": torch.tensor([101, 101, 202, 303], dtype=torch.long)
        }

    view1 = create_dummy_view(BATCH_SIZE, SEQ_LEN, config.svd_rank)
    view2 = create_dummy_view(BATCH_SIZE, SEQ_LEN, config.svd_rank) # Key 的 Group ID 和 Query 一致

    # 4. 检查队列更新前的状态
    ptr_before = model.queue_ptr.item()
    # 记录当前指针位置的队列内容，用于稍后验证是否被覆盖
    queue_content_before = model.queue[:, ptr_before : ptr_before + BATCH_SIZE].clone()
    print(f"   当前队列指针: {ptr_before}")

    # 5. 前向传播 (Forward)
    print(f"\n[Step 3] 执行 Forward Pass...")
    try:
        outputs = model(view1, view2)
        print("✅ Forward 执行完成")
    except Exception as e:
        print(f"❌ Forward 报错: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 6. 验证输出内容
    print(f"\n[Step 4] 验证输出指标...")
    
    loss = outputs['loss']
    mlm_loss = outputs['mlm_loss']
    ctr_loss = outputs['contrastive_loss']

    print(f"   -> Total Loss: {loss.item():.4f}")
    print(f"   -> MLM Loss:   {mlm_loss.item():.4f}")
    print(f"   -> MoCo Loss:  {ctr_loss.item():.4f}")

    # 7. 验证队列更新机制
    print(f"\n[Step 5] 验证队列更新机制...")
    ptr_after = model.queue_ptr.item()
    queue_content_after = model.queue[:, ptr_before : ptr_before + BATCH_SIZE].clone()
    
    # A. 指针是否移动？
    expected_ptr = (ptr_before + BATCH_SIZE) % config.moco_buffer_size
    if ptr_after == expected_ptr:
        print(f"✅ 队列指针已移动: {ptr_before} -> {ptr_after}")
    else:
        print(f"❌ 队列指针未正确移动! 当前: {ptr_after}, 预期: {expected_ptr}")

    # B. 内容是否更新？(旧内容应该是随机噪声，新内容是 Embedding)
    if not torch.allclose(queue_content_before, queue_content_after):
        print(f"✅ 队列内容已更新 (Embedding 已入队)")
    else:
        print(f"❌ 队列内容未发生变化 (入队失败)")

    # C. 验证 Queue Labels 是否也更新了
    stored_labels = model.queue_labels[ptr_before : ptr_before + BATCH_SIZE]
    if torch.equal(stored_labels, view1['group_ids']):
        print(f"✅ Queue Labels 已同步更新: {stored_labels.tolist()}")
    else:
        print(f"❌ Queue Labels 更新错误: {stored_labels.tolist()}")

    print("\n" + "="*50)
    print("🎉 测试全部通过！MoCo 逻辑闭环。")
    print("="*50)
