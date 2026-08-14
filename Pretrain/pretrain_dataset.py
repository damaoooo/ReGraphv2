import datasets
from transformers import PreTrainedTokenizerFast
from Tokenizer.ir_tokenizer import load_tokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from dataclasses import dataclass, field
import torch
import random

from typing import List, Dict, Any, Optional
from Model.graph_utils import build_batched_span_graph_tensors
from .pretrain_config import PretrainConfig, DEFAULT_CONFIG


@dataclass
class MoCoDataCollator: # 这里我们简化，不再继承，因为它逻辑很不一样了
    tokenizer: PreTrainedTokenizerFast
    dataset_pool: datasets.Dataset
    map_file: Dict[int, List[int]]
    group_id_mapping: Dict[int, int]
    config: PretrainConfig = field(default_factory=lambda: PretrainConfig())
    mlm: bool = None  # 将从config中获取
    mlm_probability: float = None  # 将从config中获取
    
    def __post_init__(self):
        """初始化后设置默认值"""
        if self.mlm is None:
            self.mlm = self.config.mlm
        if self.mlm_probability is None:
            self.mlm_probability = self.config.mlm_probability
        self.use_cfg = self.config.use_cfg
        self.use_ddg = self.config.use_ddg

    def _build_graph_inputs(
        self,
        cfg_list: List[List[Any]],
        ddg_list: List[List[Any]],
        input_ids: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        seq_lengths = [int((seq != self.tokenizer.pad_token_id).sum().item()) for seq in input_ids]

        ddg_tensors = (
            build_batched_span_graph_tensors(ddg_list, seq_lengths, include_edge_attr=False)
            if self.use_ddg
            else {"node_spans": None, "node_batch": None, "edge_index": None, "edge_attr": None}
        )
        cfg_tensors = (
            build_batched_span_graph_tensors(cfg_list, seq_lengths, include_edge_attr=True)
            if self.use_cfg
            else {"node_spans": None, "node_batch": None, "edge_index": None, "edge_attr": None}
        )

        return {
            "ddg_node_spans": ddg_tensors["node_spans"],
            "ddg_node_batch": ddg_tensors["node_batch"],
            "ddg_edge_index": ddg_tensors["edge_index"],
            "cfg_node_spans": cfg_tensors["node_spans"],
            "cfg_node_batch": cfg_tensors["node_batch"],
            "cfg_edge_index": cfg_tensors["edge_index"],
            "cfg_edge_attr": cfg_tensors["edge_attr"],
        }

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

        cfg_graph_all = batch_cache['cfg_graph'] if self.use_cfg else None
        ddg_graph_all = batch_cache['ddg_graph'] if self.use_ddg else None

        query_graph_inputs = self._build_graph_inputs(
            cfg_graph_all[:batch_size] if self.use_cfg else [],
            ddg_graph_all[:batch_size] if self.use_ddg else [],
            batch_q["input_ids"],
        )

        key_graph_inputs = self._build_graph_inputs(
            cfg_graph_all[batch_size:] if self.use_cfg else [],
            ddg_graph_all[batch_size:] if self.use_ddg else [],
            batch_k["input_ids"],
        )

        # 7. 组装最终返回字典 (Supervised MoCo 格式)
        return {
            "view1": {
                "input_ids": batch_q["input_ids"],
                "attention_mask": batch_q["attention_mask"],
                "labels": batch_q["labels"],
                "group_ids": batch_group_ids_tensor,
                **query_graph_inputs,
            },
            "view2": {
                "input_ids": batch_k["input_ids"],
                "attention_mask": batch_k["attention_mask"],
                "group_ids": batch_group_ids_tensor,
                **key_graph_inputs,
            }
        }
        
        
        

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
    tokenizer = load_tokenizer("/path/to/rell/Tokenizer/output_tokenizer/llvm_ir_bpe.json")
    dataset_pool = load_dataset("/path/to/rell/IR/train_dataset_pool")
    dataset = load_dataset("/path/to/rell/IR/train_task_dataset")
    with open("/path/to/rell/IR/train_positive_map.pkl", "rb") as f:
        map_file = pickle.load(f)
    
    group_id_mapping = compute_group_ids(map_file)
    
    collator = MoCoDataCollator(tokenizer=tokenizer, dataset_pool=dataset_pool, map_file=map_file, group_id_mapping=group_id_mapping)
    batch = collator(dataset[:2])  # 取前两个样本进行测试
    print(batch)
