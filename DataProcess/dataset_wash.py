import datasets
import os
import sys
import glob

parent_path = os.path.dirname(os.getcwd())
sys.path.append(parent_path)
from Tokenizer.ir_tokenizer import load_tokenizer

dataset_path = glob.glob("/home/damaoooo/Downloads/regraphv2/IR/binary_save2/*.parquet")
dataset = datasets.Dataset.from_parquet(dataset_path, split='train')

tokenizer = load_tokenizer("/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json")

max_length = 4096
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

def trunker(example):
    ids = example['input_ids']
    if len(ids) > max_length:
        truncated_ids = ids[:max_length-1] + [eos_token_id]  # 保留 EOS token
    else:
        truncated_ids = ids
    if example['ddg_graph'] is not None:
        example['ddg_graph'] = [edge for edge in example['ddg_graph'] if max(edge) < max_length]
    if example['cfg_graph'] is not None:
        example['cfg_graph'] = [edge for edge in example['cfg_graph'] if max(edge) < max_length]
    example['input_ids'] = truncated_ids
    return example

def filter_function(example):
    return example['cfg_graph'] is not None and example['ddg_graph'] is not None

num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

processed_dataset = dataset.filter(filter_function,num_proc=num_cores, writer_batch_size=100).map(trunker, num_proc=num_cores, writer_batch_size=100)
print("Finished processing dataset.")
processed_dataset.save_to_disk("/home/damaoooo/Downloads/regraphv2/IR/binary_save_truncated")

            
        