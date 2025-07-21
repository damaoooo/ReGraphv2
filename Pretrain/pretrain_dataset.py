import datasets
from transformers import PreTrainedTokenizerFast
from Tokenizer.ir_tokenizer import load_tokenizer
from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass
import torch

from typing import List, Dict, Any, Optional

@dataclass
class MyFinalDataCollator: # 这里我们简化，不再继承，因为它逻辑很不一样了
    tokenizer: PreTrainedTokenizerFast
    mlm: bool = True
    mlm_probability: float = 0.15
    # 为 ddg_graph 的边定义列表定义一个填充值
    edge_pad_value: int = -1 

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # --- 分离自定义列 ---
        cfg_graph = [example.pop("cfg_graph") for example in examples]
        ddg_graph = [example.pop("ddg_graph") for example in examples] # List[List[List[int]]]

        # --- 使用标准MLM Collator处理文本部分 ---
        mlm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=self.mlm, mlm_probability=self.mlm_probability)
        batch = mlm_collator(examples)
        
        
        # --- 填充 ddg_graph ---
        # 1. 找到这个batch里最多的边数
        max_edges = max(len(graph) for graph in ddg_graph)

        # 2. 填充每个graph的边列表，使其长度都为max_edges
        padded_ddg_graphs = []
        for graph in ddg_graph:
            # graph 是一个 List[[s1,e1,s2,e2], ...]
            padding_needed = max_edges - len(graph)
            # 使用一个不可能的坐标[-1,-1,-1,-1]作为填充
            padded_graph = graph + [[self.edge_pad_value] * 4] * padding_needed
            padded_ddg_graphs.append(padded_graph)
            
        # 3. 转换成Tensor
        batch["ddg_graph"] = torch.tensor(padded_ddg_graphs, dtype=torch.long)
        
        # --- 填充 cfg_graph ---
        # 1. 找到这个batch里最多的边数
        max_cfg_edges = max(len(graph) for graph in cfg_graph)
        # 2. 填充每个graph的边列表，使其长度都为max_cfg_edges
        padded_cfg_graphs = []
        for graph in cfg_graph:
            padding_needed = max_cfg_edges - len(graph)
            # 使用一个不可能的坐标[-1,-1]作为填充
            padded_graph = graph + [[self.edge_pad_value, self.edge_pad_value]] * padding_needed
            padded_cfg_graphs.append(padded_graph)
        # 3. 转换成Tensor
        batch["cfg_graph"] = torch.tensor(padded_cfg_graphs, dtype=torch.long)
        
        return batch
        

def load_dataset(dataset_path: str) -> datasets.Dataset:
    """
    加载数据集并应用必要的预处理
    """
    # 加载数据集
    dataset = datasets.load_from_disk(dataset_path)
    return dataset