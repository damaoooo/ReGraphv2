"""Build a Hugging Face dataset from the graph generator's results.db."""
from pathlib import Path

import typer

from Utils.utils import DEFAULT_TOKENIZER_PATH
from .dataset_builder_new import DatasetBuilder

app = typer.Typer(help="Build a Hugging Face dataset from generated IR graphs.", add_completion=False)


@app.command()
def main(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory containing results.db"),
    output_dir: Path = typer.Argument(..., help="New or empty Hugging Face output directory"),
    tokenizer_path: Path = typer.Option(
        Path(DEFAULT_TOKENIZER_PATH), "--tokenizer-path", "-t",
        exists=True, dir_okay=False, help="IR tokenizer JSON"
    ),
    num_processes: int | None = typer.Option(
        None, "--num-processes", "-p", min=1, help="Workers; defaults to CPU count"
    ),
):
    builder = DatasetBuilder(str(tokenizer_path), str(input_dir / "results.db"), num_processes)
    builder.process_dataset(str(output_dir))


if __name__ == "__main__":
    app()
