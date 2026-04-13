from dataclasses import dataclass, field
import random
from typing import Any, Dict, List

import datasets
import torch
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, PreTrainedTokenizerFast

from Tokenizer.ir_tokenizer import load_tokenizer
from Utils.utils import DEFAULT_TOKENIZER_PATH
from .pretrain_config import PretrainConfig


@dataclass
class MoCoDataCollator:
    tokenizer: PreTrainedTokenizerFast
    dataset_pool: datasets.Dataset
    map_file: Dict[int, List[int]]
    group_id_mapping: Dict[int, int]
    config: PretrainConfig = field(default_factory=lambda: PretrainConfig())
    mlm: bool = None
    mlm_probability: float = None

    def __post_init__(self):
        if self.mlm is None:
            self.mlm = self.config.mlm
        if self.mlm_probability is None:
            self.mlm_probability = self.config.mlm_probability

    def _truncate_input_ids(self, input_ids: List[int]) -> List[int]:
        if len(input_ids) > self.config.max_seq_length:
            return input_ids[: self.config.max_seq_length - 1] + [self.tokenizer.eos_token_id]
        return input_ids

    def __call__(self, examples: Dict[str, List]) -> Dict[str, Any]:
        if isinstance(examples, dict):
            anchor_indices = examples["anchor_idx"]
        else:
            anchor_indices = [example["anchor_idx"] for example in examples]

        positive_indices = []
        batch_group_ids = []

        for anchor_idx in anchor_indices:
            if anchor_idx in self.map_file and self.map_file[anchor_idx]:
                pos_idx = random.choice(self.map_file[anchor_idx])
            else:
                pos_idx = anchor_idx
            positive_indices.append(pos_idx)

            batch_group_ids.append(self.group_id_mapping.get(anchor_idx, anchor_idx))

        batch_group_ids_tensor = torch.tensor(batch_group_ids, dtype=torch.long)

        total_indices = anchor_indices + positive_indices
        batch_cache = self.dataset_pool.select(total_indices)
        batch_size = len(anchor_indices)

        all_input_ids = [self._truncate_input_ids(input_ids) for input_ids in batch_cache["input_ids"]]
        query_input_ids = all_input_ids[:batch_size]
        key_input_ids = all_input_ids[batch_size:]

        query_features = [{"input_ids": seq} for seq in query_input_ids]
        key_features = [{"input_ids": seq} for seq in key_input_ids]

        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=self.mlm,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )
        batch_q = mlm_collator(query_features)

        pad_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            max_length=self.config.max_seq_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )
        batch_k = pad_collator(key_features)

        return {
            "view1": {
                "input_ids": batch_q["input_ids"],
                "attention_mask": batch_q["attention_mask"],
                "labels": batch_q["labels"],
                "group_ids": batch_group_ids_tensor,
            },
            "view2": {
                "input_ids": batch_k["input_ids"],
                "attention_mask": batch_k["attention_mask"],
                "group_ids": batch_group_ids_tensor,
            },
        }


def load_dataset(dataset_path: str) -> datasets.Dataset:
    return datasets.load_from_disk(dataset_path)


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
    import pickle

    config = PretrainConfig()
    tokenizer = load_tokenizer(DEFAULT_TOKENIZER_PATH)
    dataset_pool = load_dataset(config.train_dataset_pool_path)
    dataset = load_dataset(config.train_dataset_idx_path)
    with open(config.train_dataset_map_path, "rb") as handle:
        map_file = pickle.load(handle)

    group_id_mapping = compute_group_ids(map_file)
    collator = MoCoDataCollator(
        tokenizer=tokenizer,
        dataset_pool=dataset_pool,
        map_file=map_file,
        group_id_mapping=group_id_mapping,
        config=config,
    )
    batch = collator(dataset[:2])
    print(batch.keys())
