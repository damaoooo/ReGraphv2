import os

import datasets
import typer
from typing_extensions import Annotated

from Tokenizer.ir_tokenizer import load_tokenizer
from Utils.utils import DEFAULT_TOKENIZER_PATH


def main(
    dataset_path: Annotated[
        str,
        typer.Option(help="Path to the dataset.", rich_help_panel="Custom Arguments"),
    ],
    output_path: Annotated[
        str,
        typer.Option(
            help="Path to save the processed dataset.",
            rich_help_panel="Custom Arguments",
        ),
    ],
    tokenizer_path: Annotated[
        str,
        typer.Option(
            help="Path to the tokenizer file.",
            rich_help_panel="Custom Arguments",
        ),
    ] = DEFAULT_TOKENIZER_PATH,
    max_seq_length: Annotated[
        int,
        typer.Option(
            help="Maximum sequence length.",
            rich_help_panel="Custom Arguments",
        ),
    ] = 2048,
):
    dataset = datasets.load_from_disk(dataset_path)
    tokenizer = load_tokenizer(tokenizer_path)
    eos_token_id = tokenizer.eos_token_id

    def truncater(example):
        input_ids = example["input_ids"]
        if len(input_ids) > max_seq_length:
            example["input_ids"] = input_ids[: max_seq_length - 1] + [eos_token_id]
        return example

    def filter_function(example):
        return example["input_ids"] is not None and len(example["input_ids"]) > 0

    num_cores = os.cpu_count() or 1
    print(f"Number of CPU cores: {num_cores}")

    processed_dataset = dataset.filter(
        filter_function,
        num_proc=num_cores,
        writer_batch_size=100,
    ).map(
        truncater,
        num_proc=num_cores,
        writer_batch_size=100,
    )
    print("Finished processing dataset.")
    processed_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    typer.run(main)
