# file: graph_roformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from transformers import RoFormerPreTrainedModel, RoFormerModel, RoFormerConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerLayer
from transformers.models.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding
from transformers.pytorch_utils import apply_chunking_to_forward

# 1. 魔改 Attention 层：这是唯一需要动大手术的地方
class GraphFlashRoFormerSelfAttention(nn.Module):
    def __init__(self, config: RoFormerConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})"
            )
        
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = int(config.hidden_size / config.num_attention_heads)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 使用 RoFormer 的正弦位置编码来生成 RoPE
        # 安全地获取 rotary_value 配置，默认为 False
        self.rotary_value: bool = getattr(config, 'rotary_value', False)
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, self.head_dim
        )

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        cfg_u: torch.Tensor, # Shape: [B, S, Rank]
        cfg_v: torch.Tensor, # Shape: [B, S, Rank]
        ddg_edges: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # input: [batch, seq, dim]
        B, L, _ = hidden_states.shape
        
        # 处理DDG
        ddg_src = ddg_edges[:, :, 0:2]
        ddg_dst = ddg_edges[:, :, 2:4]
        edge_mask = (ddg_edges != -1).all(dim=-1)  # Shape: [B, max_edges]

        # X <- X + Avg(Parents)  (用 start 位置作为节点代表)
        src_pos = ddg_src[:, :, 0].clamp(min=0)  # Shape: [B, max_edges]
        dst_pos = ddg_dst[:, :, 0].clamp(min=0)  # Shape: [B, max_edges]

        parent_feat = hidden_states.gather(
            1, src_pos.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        )
        parent_feat = parent_feat * edge_mask.unsqueeze(-1)

        agg = torch.zeros_like(hidden_states)
        agg.scatter_add_(
            1, dst_pos.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)), parent_feat
        )

        deg = torch.zeros(B, L, device=hidden_states.device, dtype=hidden_states.dtype)
        deg.scatter_add_(1, dst_pos, edge_mask.to(hidden_states.dtype))
        deg = deg.clamp_min(1.0)

        hidden_states = hidden_states + agg / deg.unsqueeze(-1)

        q: torch.Tensor = self.query(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # Shape: [B, num_heads, S, head_dim]
        k: torch.Tensor = self.key(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v: torch.Tensor = self.value(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE (处理序列位置)
        # 生成位置编码
        sinusoidal_pos: torch.Tensor = self.embed_positions(hidden_states.shape[:2])[None, None, :, :]
        
        # 应用旋转位置编码到 q 和 k (标准RoPE实现)
        # sinusoidal_pos: [1, 1, L, head_dim] -> sin/cos: [1, 1, L, head_dim/2]
        sin = sinusoidal_pos[..., 0::2]
        cos = sinusoidal_pos[..., 1::2]

        # RoPE 应用公式 - 支持 fp16 和 bf16
        if q.dtype in (torch.float16, torch.bfloat16):
            sin = sin.to(q.dtype)
            cos = cos.to(q.dtype)

        def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            x_rot = torch.empty_like(x)
            x_rot[..., 0::2] = x1 * cos - x2 * sin
            x_rot[..., 1::2] = x1 * sin + x2 * cos
            return x_rot

        q_rot = apply_rope(q, sin, cos)  # Shape: [B, num_heads, S, head_dim]
        k_rot = apply_rope(k, sin, cos)  # Shape: [B, num_heads, S, head_dim]

        if self.rotary_value:
            v = apply_rope(v, sin, cos)
        
        
        # put CFG
        d = self.head_dim
        q_rot /= d ** 0.25
        k_rot /= d ** 0.25
        q_rot = torch.concat([q_rot, cfg_u.unsqueeze(1).expand(-1, self.num_heads, -1, -1)], dim=-1)
        k_rot = torch.concat([k_rot, cfg_v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)], dim=-1)

        # --- 核心：Flash Attention + Graph Bias ---
        # 注意：attn_mask 在 SDPA 里通常是加法 mask (0 是保留，-inf 是遮蔽/bias)
        
        # Pad V to match Q/K size after adding CFG
        pad_size = q_rot.size(-1) - v.size(-1)
        if pad_size > 0:
            v = F.pad(v, (0, pad_size), value=0.0)
            
        
        # PyTorch 2.0+ 神器：自动选择 FlashAttention, xFormers 或 CuDNN
        attn_output = F.scaled_dot_product_attention(
            q_rot, k_rot, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=1.0,
            is_causal=False  # 双向 Attention
        )
        if pad_size > 0:
            attn_output = attn_output[..., :self.head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        
        return attn_output
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """旋转输入张量的一半隐藏维度"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

# 2. 组装 Layer
class GraphRoFormerLayer(RoFormerLayer):
    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        # 替换掉原本慢吞吞的 SelfAttention，换成咱们的 Flash 版
        self.attention.self = GraphFlashRoFormerSelfAttention(config)

    def forward(
        self,
        hidden_states,
        cfg_u: torch.Tensor,
        cfg_v: torch.Tensor,
        ddg_edges: torch.Tensor,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs
    ):
        if self.is_decoder:
            raise NotImplementedError("GraphRoFormerLayer does not support decoder mode.")
        if cfg_u is None or cfg_v is None or ddg_edges is None:
            raise ValueError("cfg_u, cfg_v, and ddg_edges are required for GraphRoFormerLayer.")

        self_attention_output = self.attention.self(
            hidden_states,
            cfg_u=cfg_u,
            cfg_v=cfg_v,
            ddg_edges=ddg_edges,
        )
        attention_output = self.attention.output(self_attention_output, hidden_states)

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,)
        if output_attentions:
            outputs = outputs + (None,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# 3. 组装 Model
class GraphRoFormerModel(RoFormerPreTrainedModel):
    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.config = config
        
        # 验证配置参数
        if not hasattr(config, 'num_hidden_layers') or config.num_hidden_layers <= 0:
            raise ValueError(f"Invalid num_hidden_layers: {getattr(config, 'num_hidden_layers', None)}")
        if not hasattr(config, 'hidden_size') or config.hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {getattr(config, 'hidden_size', None)}")
        if not hasattr(config, 'num_attention_heads') or config.num_attention_heads <= 0:
            raise ValueError(f"Invalid num_attention_heads: {getattr(config, 'num_attention_heads', None)}")
        
        self.embeddings = RoFormerModel(config).embeddings # 复用 Embedding 层
        
        # 堆叠我们的魔改 Layer
        self.layers = nn.ModuleList([
            GraphRoFormerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        cfg_u: torch.Tensor,
        cfg_v: torch.Tensor,
        ddg_edges: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide input_ids or inputs_embeds.")
        if cfg_u is None or cfg_v is None or ddg_edges is None:
            raise ValueError("cfg_u, cfg_v, and ddg_edges are required for GraphRoFormerModel.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=None,
                output_attentions=output_attentions,
                cfg_u=cfg_u,
                cfg_v=cfg_v,
                ddg_edges=ddg_edges,
                return_dict=return_dict,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            output = (hidden_states, None, all_hidden_states, all_self_attentions, None)
            return tuple(v for v in output if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )
    
    
if __name__ == "__main__":
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    
    # ==========================================
    # 1. 准备阶段
    # ==========================================
    device = 'cuda'
    print(f"正在使用设备: {device}")

    # 你的新配置
    hidden_layers = 24
    max_position_embeddings = 4096
    rank = 32
    sequence_length = 4096
    attention_heads = 12
    batch_size = 8
    hidden_size = 768  # head_dim = 1024/16 = 64
    
    config = RoFormerConfig(
        hidden_size=hidden_size,
        num_attention_heads=attention_heads,
        num_hidden_layers=hidden_layers,
        max_position_embeddings=max_position_embeddings,
        rotary_value=True
    )
    
    # 实例化模型 (GraphRoFormerModel)
    model = GraphRoFormerModel(config).to(device).to(torch.bfloat16)
    model = torch.compile(model)  # PyTorch 2.0+ 编译模型以提升性能
    # 构造数据
    input_ids = torch.randint(0, 1000, (batch_size, sequence_length), device=device)
    cfg_u = torch.randn(batch_size, sequence_length, rank, dtype=torch.bfloat16).to(device)
    cfg_v = torch.randn(batch_size, sequence_length, rank, dtype=torch.bfloat16).to(device)
    
    # 构造简单的 DDG
    ddg_edges = torch.full((batch_size, 50, 4), -1, dtype=torch.long).to(device)
    # 随便填点数据防止报错
    ddg_edges[:, :30, :] = 1 

    # ==========================================
    # 2. Warmup (预热)
    # ==========================================
    print("正在预热 CUDA Kernels...")
    with torch.no_grad():
        _ = model(input_ids=input_ids, cfg_u=cfg_u, cfg_v=cfg_v, ddg_edges=ddg_edges)
    torch.cuda.synchronize()

    # ==========================================
    # 3. 开始 Profile (开启显存记录!)
    # ==========================================
    print("开始 Profile (包含显存分析)...")
    
    # 【宏观视角】先重置一下显存峰值统计
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    

    outputs = model(input_ids=input_ids, cfg_u=cfg_u, cfg_v=cfg_v, ddg_edges=ddg_edges)


    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    # ==========================================
    # 4. 打印结果
    # ==========================================
    print("-" * 60)
    print(f"GraphRoFormerModel output shape: {outputs.last_hidden_state.shape}")
    print(f"Model is running on device: {device} with dtype: {outputs.last_hidden_state.dtype}")
    print("-" * 60)
    
    # 1. 宏观显存统计 (最直观)
    print(f"【宏观显存统计】")
    print(f"运行前显存占用: {start_mem / 1024**2:.2f} MB")
    print(f"运行后显存占用: {end_mem / 1024**2:.2f} MB")
    print(f"过程峰值显存 (Peak): {peak_mem / 1024**2:.2f} MB (这是你需要关注的极限)")
    print("-" * 60)
