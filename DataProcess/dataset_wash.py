import datasets
import os
import sys
import glob
import typer
from typing_extensions import Annotated

# 如果从其他地方调用，可以导入配置
# parent_path = os.path.dirname(os.getcwd())
# sys.path.append(parent_path)
from Tokenizer.ir_tokenizer import load_tokenizer


def main(
    dataset_path: Annotated[str, typer.Option(help="Path to the dataset.", rich_help_panel="Custom Arguments")],
    output_path: Annotated[str, typer.Option(help="Path to save the processed dataset.", rich_help_panel="Custom Arguments")],
    tokenizer_path: Annotated[str, typer.Option(help="Path to the tokenizer file.", rich_help_panel="Custom Arguments")] = "/home/damaoooo/Datasets/IR/Dataset-1-all/train_corpus_tokenizer/llvm_ir_bpe.json",
    max_seq_length: Annotated[int, typer.Option(help="Maximum sequence length.",rich_help_panel="Custom Arguments")] = 2048,
):
    
    files = glob.glob(os.path.join(dataset_path, "*.parquet"))
    dataset = datasets.Dataset.from_parquet(files)

    tokenizer = load_tokenizer(tokenizer_path)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    def trunker(example):
        ids = example['input_ids']
        if len(ids) > max_seq_length:
            truncated_ids = ids[:max_seq_length-1] + [eos_token_id]  # 保留 EOS token
        else:
            truncated_ids = ids
        if example['ddg_graph'] is not None:
            example['ddg_graph'] = [edge for edge in example['ddg_graph'] if max(edge) < max_seq_length]
        if example['cfg_graph'] is not None:
            example['cfg_graph'] = [edge for edge in example['cfg_graph'] if max(edge) < max_seq_length]
        example['input_ids'] = truncated_ids
        return example

    def filter_function(example):
        return example['cfg_graph'] is not None and example['ddg_graph'] is not None

    num_cores = os.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    processed_dataset = dataset.filter(filter_function,num_proc=num_cores, writer_batch_size=100).map(trunker, num_proc=num_cores, writer_batch_size=100)
    print("Finished processing dataset.")
    processed_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    typer.run(main)