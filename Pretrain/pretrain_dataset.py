import datasets
from transformers import PreTrainedTokenizerFast
from Tokenizer.ir_tokenizer import load_tokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from dataclasses import dataclass
import torch
import random

from typing import List, Dict, Any, Optional

@dataclass
class MyFinalDataCollator: # 这里我们简化，不再继承，因为它逻辑很不一样了
    tokenizer: PreTrainedTokenizerFast
    dataset_pool: datasets.Dataset
    map_file: Dict[int, List[int]]
    mlm: bool = True
    mlm_probability: float = 0.15
    # 为 ddg_graph 的边定义列表定义一个填充值
    edge_pad_value: int = -1

    def __call__(self, examples: Dict[str, List]) -> Dict[str, Any]:
        # --- 分离自定义列 ---
        if isinstance(examples, dict):
            anchor_indices = examples['anchor_idx']
        else:
            anchor_indices = [example['anchor_idx'] for example in examples]
            
        positive_sample_indices = []
        for anchor_idx in anchor_indices:
            positive_sample_indices.append(random.choice(self.map_file[anchor_idx]))
                
        negative_sample_indices = []
        for anchor_idx in anchor_indices:
            # Negative sample should not be the same as positive sample and not in the positive map values
            negative_sample_index = random.randint(0, len(self.dataset_pool) - 1)
            while negative_sample_index in self.map_file[anchor_idx] or negative_sample_index == anchor_idx:
                negative_sample_index = random.randint(0, len(self.dataset_pool) - 1)
            negative_sample_indices.append(negative_sample_index)
            
        total_indices = anchor_indices + positive_sample_indices + negative_sample_indices
        
        assert len(total_indices) == len(anchor_indices) * 3, "Total indices should be three times the number of anchors"
        
        batch_cache = self.dataset_pool.select(total_indices)
        mlm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=self.mlm, mlm_probability=self.mlm_probability, pad_to_multiple_of=8)
        batch = mlm_collator(batch_cache['input_ids'])
        
        # Shape: [batch_size, max_seq_length]
        max_seq_length = batch['input_ids'].shape[1]
        pad_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding='max_length', max_length=max_seq_length, pad_to_multiple_of=8)
        new_batch_cache = pad_collator({'input_ids': batch_cache['input_ids']})

        batch['contract_input_ids'] = new_batch_cache['input_ids']
        batch['contract_attention_mask'] = new_batch_cache['attention_mask']
        batch['attention_mask'] = new_batch_cache['attention_mask'][:len(anchor_indices)]
        
        cfg_graph = batch_cache['cfg_graph']
        ddg_graph = batch_cache['ddg_graph']
        
        # Shape: [batch_size, max_edges, edge_feature_length]
        padded_ddg_graphs = self.pad_graph(ddg_graph)
        
        # Shape: [batch_size, max_edges, edge_feature_length]
        padded_cfg_graphs = self.pad_graph(cfg_graph, feature_length=5)
        
        batch["ddg_graph"] = torch.tensor(padded_ddg_graphs, dtype=torch.long)
        batch["cfg_graph"] = torch.tensor(padded_cfg_graphs, dtype=torch.long)
        
        batch['input_ids'] = batch['input_ids'][:len(anchor_indices)]
        batch['labels'] = batch['labels'][:len(anchor_indices)]
        
        return batch

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


if __name__ == "__main__":
    # 测试代码
    tokenizer = load_tokenizer("/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json")
    dataset = load_dataset("/home/damaoooo/Downloads/regraphv2/IR/train_dataset")
    
    collator = MyFinalDataCollator(tokenizer=tokenizer)
    batch = collator(dataset[:2])  # 取前两个样本进行测试
    print(batch)