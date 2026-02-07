import datasets
from transformers import PreTrainedTokenizerFast
from Tokenizer.ir_tokenizer import load_tokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from dataclasses import dataclass, field
import torch
from torch.nn.utils.rnn import pad_sequence
import random

from typing import List, Dict, Any, Optional
from .pretrain_config import PretrainConfig, DEFAULT_CONFIG


def factorize_cfg_to_uv_batch(edge_tensor, rank, total_seq_len=None, device='cpu', svd_lowrank_threshold=128):
    """
    输入:
        edge_tensor: Tensor [B, E, 5]
            每条边为 [src_start, src_end, dst_start, dst_end, value]
            E 维可能有 padding，整行全 -1 代表无效
        rank: 期望的低秩维数 R
        total_seq_len: 可选，若指定则输出对齐到该长度
        svd_lowrank_threshold: 当 block 数量超过该阈值时，使用近似 SVD

    输出:
        U_batch: Tensor [B, Max_S, R]
        V_batch: Tensor [B, Max_S, R]

    约束:
        - 如果实际矩阵秩 < rank，输出会在后面用 0 填充到 rank
        - 如果实际矩阵秩 > rank，会被截断到 rank

    满足: A[i, j] ≈ U[i] @ V[j].T
    """
    if edge_tensor.device.type != device:
        edge_tensor = edge_tensor.to(device)
    batch_size, _, _ = edge_tensor.shape
    u_list = []
    v_list = []
    lengths = []

    for b in range(batch_size):
        edges = edge_tensor[b]
        valid_mask = ~(edges == -1).all(dim=-1)
        edges = edges[valid_mask]

        if edges.numel() == 0:
            actual_len = total_seq_len if total_seq_len is not None else 0
            empty = torch.zeros((actual_len, rank), device=device, dtype=torch.float32)
            u_list.append(empty)
            v_list.append(empty)
            lengths.append(0)
            continue

        ranges_src = edges[:, 0:2].to(torch.long)
        ranges_dst = edges[:, 2:4].to(torch.long)
        all_ranges = torch.cat([ranges_src, ranges_dst], dim=0)

        max_val = torch.max(all_ranges[:, 1])
        current_seq_len = total_seq_len if total_seq_len is not None else max_val.item()
        key_mul = max_val + 1

        keys = all_ranges[:, 0] * key_mul + all_ranges[:, 1]
        unique_keys = torch.unique(keys, sorted=True)
        unique_starts = unique_keys // key_mul
        unique_ends = unique_keys % key_mul

        block_sizes = (unique_ends - unique_starts).to(torch.float32)
        sqrt_sizes = torch.sqrt(block_sizes)
        num_blocks = unique_keys.numel()

        src_keys = keys[:ranges_src.shape[0]]
        dst_keys = keys[ranges_src.shape[0]:]
        i_idx = torch.searchsorted(unique_keys, src_keys)
        j_idx = torch.searchsorted(unique_keys, dst_keys)

        vals = edges[:, 4].to(torch.float32)
        weight = vals * sqrt_sizes[i_idx] * sqrt_sizes[j_idx]

        flat_idx = i_idx * num_blocks + j_idx
        B_flat = torch.zeros(num_blocks * num_blocks, device=device, dtype=torch.float32)
        B_flat.scatter_add_(0, flat_idx, weight)
        B_tilde = B_flat.view(num_blocks, num_blocks)

        real_rank = min(rank, num_blocks)
        # 大矩阵时使用近似 SVD 加速
        if num_blocks >= svd_lowrank_threshold and real_rank < num_blocks:
            U_small, S_vals, V_small = torch.linalg.svd_lowrank(
                B_tilde, q=real_rank, niter=2
            )
        else:
            U_small, S_vals, Vh_small = torch.linalg.svd(B_tilde, full_matrices=False)
            V_small = Vh_small[:real_rank, :].T
        U_small = U_small[:, :real_rank]
        S_vals = S_vals[:real_rank]

        sqrt_S = torch.sqrt(S_vals).unsqueeze(0)
        U_block_emb = (U_small * sqrt_S) / sqrt_sizes.unsqueeze(1)
        V_block_emb = (V_small * sqrt_S) / sqrt_sizes.unsqueeze(1)

        if real_rank < rank:
            pad_cols = rank - real_rank
            U_block_emb = torch.cat(
                [U_block_emb, torch.zeros((num_blocks, pad_cols), device=device, dtype=U_block_emb.dtype)],
                dim=1
            )
            V_block_emb = torch.cat(
                [V_block_emb, torch.zeros((num_blocks, pad_cols), device=device, dtype=V_block_emb.dtype)],
                dim=1
            )

        repeat_counts = block_sizes.to(torch.long)
        U_compact = torch.repeat_interleave(U_block_emb, repeat_counts, dim=0)
        V_compact = torch.repeat_interleave(V_block_emb, repeat_counts, dim=0)

        starts_expanded = torch.repeat_interleave(unique_starts, repeat_counts)
        total_tokens = int(repeat_counts.sum().item())
        offsets = torch.arange(total_tokens, device=device) - torch.repeat_interleave(
            torch.cumsum(repeat_counts, dim=0) - repeat_counts, repeat_counts
        )
        full_indices = starts_expanded + offsets

        U_final = torch.zeros((current_seq_len, rank), device=device, dtype=U_compact.dtype)
        V_final = torch.zeros((current_seq_len, rank), device=device, dtype=V_compact.dtype)

        valid_idx_mask = full_indices < current_seq_len
        if valid_idx_mask.all():
            U_final[full_indices] = U_compact
            V_final[full_indices] = V_compact
        else:
            valid_indices = full_indices[valid_idx_mask]
            U_final[valid_indices] = U_compact[valid_idx_mask]
            V_final[valid_indices] = V_compact[valid_idx_mask]

        u_list.append(U_final)
        v_list.append(V_final)
        lengths.append(U_final.shape[0])

    u_batch = pad_sequence(u_list, batch_first=True)
    v_batch = pad_sequence(v_list, batch_first=True)

    if total_seq_len is not None:
        if u_batch.shape[1] < total_seq_len:
            pad_len = total_seq_len - u_batch.shape[1]
            u_batch = torch.nn.functional.pad(u_batch, (0, 0, 0, pad_len))
            v_batch = torch.nn.functional.pad(v_batch, (0, 0, 0, pad_len))
        elif u_batch.shape[1] > total_seq_len:
            u_batch = u_batch[:, :total_seq_len]
            v_batch = v_batch[:, :total_seq_len]

    return u_batch, v_batch

@dataclass
class MoCoDataCollator: # 这里我们简化，不再继承，因为它逻辑很不一样了
    tokenizer: PreTrainedTokenizerFast
    dataset_pool: datasets.Dataset
    map_file: Dict[int, List[int]]
    group_id_mapping: Dict[int, int]
    config: PretrainConfig = field(default_factory=lambda: PretrainConfig())
    mlm: bool = None  # 将从config中获取
    mlm_probability: float = None  # 将从config中获取
    edge_pad_value: int = None  # 将从config中获取
    
    def __post_init__(self):
        """初始化后设置默认值"""
        if self.mlm is None:
            self.mlm = self.config.mlm
        if self.mlm_probability is None:
            self.mlm_probability = self.config.mlm_probability
        if self.edge_pad_value is None:
            self.edge_pad_value = self.config.edge_pad_value

    def __call__(self, examples: Dict[str, List]) -> Dict[str, Any]:
        # --- 分离自定义列 ---
        if isinstance(examples, dict):
            anchor_indices = examples['anchor_idx']
        else:
            anchor_indices = [example['anchor_idx'] for example in examples]
            
        # 2. 采样 Positive & 查 Group ID
        positive_indices = []
        batch_group_ids = []

        for anchor_idx in anchor_indices:
            # A. 采样 Positive (Key)
            # map_file 保证了 list 里的都是同源函数
            if anchor_idx in self.map_file and self.map_file[anchor_idx]:
                pos_idx = random.choice(self.map_file[anchor_idx])
            else:
                pos_idx = anchor_idx # 兜底：孤儿节点自己指自己
            positive_indices.append(pos_idx)

            # B. 查 Group ID (用于 MoCo Queue Masking 防止误伤)
            # 直接查表，O(1) 复杂度，极速
            if anchor_idx in self.group_id_mapping:
                gid = self.group_id_mapping[anchor_idx]
            else:
                gid = anchor_idx # 兜底
            batch_group_ids.append(gid)

        # 转为 Tensor (供模型使用)
        batch_group_ids_tensor = torch.tensor(batch_group_ids, dtype=torch.long)
            
        # 3. 统一从大 Pool 里取数据
        # 这里的 total_indices 长度是 2 * batch_size
        total_indices = anchor_indices + positive_indices
        batch_cache = self.dataset_pool.select(total_indices)
        
        batch_size = len(anchor_indices)
        
        # 4. 预处理 Input IDs (处理截断)
        all_input_ids = []
        for input_ids in batch_cache['input_ids']:
            if len(input_ids) > self.config.max_seq_length:
                # 截断并保留 EOS (假设 EOS 是最后一个 token)
                all_input_ids.append(input_ids[:self.config.max_seq_length-1] + [self.tokenizer.eos_token_id])
            else:
                all_input_ids.append(input_ids)

        # --- 分流处理 ---
        # Query (View 1): 做 MLM，增加难度
        query_input_ids = all_input_ids[:batch_size]
        # Key (View 2): 不做 MLM，保持稳定 (Padding Only)
        key_input_ids = all_input_ids[batch_size:]

        query_features = [{"input_ids": seq} for seq in query_input_ids]
        key_features = [{"input_ids": seq} for seq in key_input_ids]

        # 5. 构建 Text Collators
        # A. Query Collator (MLM)
        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=self.mlm_probability, 
            pad_to_multiple_of=8
        )
        batch_q = mlm_collator(query_features) 

        # B. Key Collator (Padding)
        pad_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, 
            padding='longest', 
            max_length=self.config.max_seq_length,
            pad_to_multiple_of=8
        )
        batch_k = pad_collator(key_features)

        # 6. 处理图结构 (CFG & DDG)
        cfg_graph_all = batch_cache['cfg_graph']
        ddg_graph_all = batch_cache['ddg_graph']
        
        # 定义内部函数处理图 (复用逻辑)
        def process_graphs(cfg_list, ddg_list, seq_len_tensor):
            # 1. DDG 处理 (Padding)
            padded_ddg = self.pad_graph(ddg_list)
            ddg_tensor = torch.tensor(padded_ddg, dtype=torch.long)
            
            # 2. CFG 处理 (Padding -> SVD)
            if len(cfg_list) == 0 or len(cfg_list[0]) == 0:
                # 处理极端情况：如果没有任何边，直接返回全零的 U 和 V
                batch_size = len(cfg_list)
                current_max_len = seq_len_tensor.shape[1]
                u_empty = torch.zeros((batch_size, current_max_len, self.config.svd_rank), dtype=torch.float32)
                v_empty = torch.zeros((batch_size, current_max_len, self.config.svd_rank), dtype=torch.float32)
                return ddg_tensor, u_empty, v_empty
            
            padded_cfg = self.pad_graph(cfg_list, feature_length=5)
            
            # 这里的 seq_len 取决于当前 batch text padding 后的实际长度
            current_max_len = seq_len_tensor.shape[1] 
            
            # 调用你的 SVD 分解函数 (CPU 上运行)
            U, V = factorize_cfg_to_uv_batch(
                torch.tensor(padded_cfg, dtype=torch.long), 
                rank=self.config.svd_rank, 
                total_seq_len=current_max_len, 
                device='cpu' 
            )
            return ddg_tensor, U, V

        # 处理 Query 的图
        ddg_q, u_q, v_q = process_graphs(
            cfg_graph_all[:batch_size], 
            ddg_graph_all[:batch_size], 
            batch_q['input_ids']
        )
        
        # 处理 Key 的图
        ddg_k, u_k, v_k = process_graphs(
            cfg_graph_all[batch_size:], 
            ddg_graph_all[batch_size:], 
            batch_k['input_ids']
        )

        # 7. 组装最终返回字典 (Supervised MoCo 格式)
        return {
            "view1": {
                "input_ids": batch_q["input_ids"],
                "attention_mask": batch_q["attention_mask"],
                "labels": batch_q["labels"], # 只有 Query 有 MLM label
                "cfg_u": u_q,
                "cfg_v": v_q,
                "ddg_edges": ddg_q,
                "group_ids": batch_group_ids_tensor # 传入 Ground Truth ID
            },
            "view2": {
                "input_ids": batch_k["input_ids"],
                "attention_mask": batch_k["attention_mask"],
                # Key 不需要 labels
                "cfg_u": u_k,
                "cfg_v": v_k,
                "ddg_edges": ddg_k,
                "group_ids": batch_group_ids_tensor # Key 共享同一个 ID
            }
        }

    def pad_graph(self, graph, feature_length: int = 4):
        max_edges = max(len(g) for g in graph)
        padded_graphs = []
        for g in graph:
            padding_needed = max_edges - len(g)
            padded_graph = g + [[-1] * feature_length] * padding_needed
            padded_graphs.append(padded_graph)
        return padded_graphs
        
        
        

def load_dataset(dataset_path: str) -> datasets.Dataset:
    """
    加载数据集并应用必要的预处理
    """
    # 加载数据集
    dataset = datasets.load_from_disk(dataset_path)
    return dataset

def compute_group_ids(data: Dict[int, List[int]]) -> Dict[int, int]:
    key_to_group_id = {}
    for anchor_key, positive_keys in data.items():
        if anchor_key in key_to_group_id:
            continue

        if positive_keys:
            min_positive = min(positive_keys)
            if anchor_key < min_positive:
                group_id = anchor_key
                key_to_group_id[anchor_key] = group_id
                for member in positive_keys:
                    key_to_group_id[member] = group_id
        else:
            key_to_group_id[anchor_key] = anchor_key
            
    return key_to_group_id


if __name__ == "__main__":
    # 测试代码
    import pickle
    tokenizer = load_tokenizer("/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json")
    dataset_pool = load_dataset("/home/damaoooo/Downloads/regraphv2/IR/train_dataset_pool")
    dataset = load_dataset("/home/damaoooo/Downloads/regraphv2/IR/train_task_dataset")
    with open("/home/damaoooo/Downloads/regraphv2/IR/train_positive_map.pkl", "rb") as f:
        map_file = pickle.load(f)
    
    group_id_mapping = compute_group_ids(map_file)
    
    collator = MoCoDataCollator(tokenizer=tokenizer, dataset_pool=dataset_pool, map_file=map_file, group_id_mapping=group_id_mapping)
    batch = collator(dataset[:2])  # 取前两个样本进行测试
    print(batch)