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
from Utils.utils import DEFAULT_TOKENIZER_PATH

def clip_span(start, end, last_position):
    """Clip an inclusive token span to `last_position`; None if the span starts after it."""
    if start > last_position:
        return None
    return start, min(end, last_position)


def truncate_example(example, max_seq_length, eos_token_id):
    """Cut a sample to `max_seq_length` tokens and keep its graph spans on the kept text.

    The last kept position is replaced by EOS, so spans may reach at most `max_seq_length - 2`. A span that
    crosses the cut is clipped there; an edge is dropped only when one of its spans lies entirely past it.
    """
    ids = example['input_ids']
    if len(ids) <= max_seq_length:
        return example
    example['input_ids'] = ids[:max_seq_length - 1] + [eos_token_id]
    last_position = max_seq_length - 2
    for name in ('ddg_graph', 'cfg_graph'):
        if example[name] is None:
            continue
        kept = []
        for edge in example[name]:
            source = clip_span(edge[0], edge[1], last_position)
            target = clip_span(edge[2], edge[3], last_position)
            if source is not None and target is not None:
                kept.append([*source, *target, *edge[4:]])  # CFG edges keep their probability
        example[name] = kept
    return example


def main(
    dataset_path: Annotated[str, typer.Option(help="Path to the dataset.", rich_help_panel="Custom Arguments")],
    output_path: Annotated[str, typer.Option(help="Path to save the processed dataset.", rich_help_panel="Custom Arguments")],
    tokenizer_path: Annotated[str, typer.Option(help="Path to the tokenizer file.", rich_help_panel="Custom Arguments")] = DEFAULT_TOKENIZER_PATH,
    max_seq_length: Annotated[int, typer.Option(help="Maximum sequence length.",rich_help_panel="Custom Arguments")] = 2048,
):
    
    dataset = datasets.load_from_disk(dataset_path)
    tokenizer = load_tokenizer(tokenizer_path)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    def trunker(example):
        return truncate_example(example, max_seq_length, eos_token_id)

    def filter_function(example):
        return example['cfg_graph'] is not None and example['ddg_graph'] is not None

    num_cores = os.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    processed_dataset = dataset.filter(filter_function,num_proc=num_cores, writer_batch_size=100).map(trunker, num_proc=num_cores, writer_batch_size=100)
    print("Finished processing dataset.")
    processed_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    typer.run(main)
