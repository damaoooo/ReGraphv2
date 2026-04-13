"""
Command line interface for ASM dataset building
"""
import os

import typer
from rich.console import Console

from Tokenizer.ir_tokenizer import load_tokenizer
from Utils.utils import DEFAULT_TOKENIZER_PATH

from .dataset_builder_new import DatasetBuilder
from .dataset_utils import find_asm_files

app = typer.Typer(help="ASM Dataset Builder")
console = Console()


@app.command("directory")
def process_directory(
    input_dir: str = typer.Argument(..., help="Directory containing ASM files"),
    output_file: str = typer.Argument(..., help="Output directory for results"),
    tokenizer_path: str = typer.Option(
        DEFAULT_TOKENIZER_PATH,
        "--tokenizer-path",
        "-t",
        help="Path to tokenizer file",
    ),
    use_hf: bool = typer.Option(
        False,
        "--use-hf",
        help="Use Hugging Face dataset instead of parquet/json summaries",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        "-b",
        help="Batch size for processing (smaller for better load balancing)",
    ),
    num_processes: int = typer.Option(
        None,
        "--num-processes",
        "-p",
        help="Number of processes to use",
    ),
    no_cleanup: bool = typer.Option(
        False,
        "--no-cleanup",
        help="Retained for compatibility; ASM processing does not create temp files",
    ),
    use_parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Use parallel chunk processing instead of batched",
    ),
    cache: bool = typer.Option(
        True,
        "--no-cache",
        help="Disable caching of file discovery results",
    ),
):
    """Process ASM files from a directory for dataset creation."""
    input_dir = os.path.abspath(input_dir)

    if not os.path.exists(input_dir):
        console.print(f"[red]Directory not found: {input_dir}")
        raise typer.Exit(1)

    if not os.path.isdir(input_dir):
        console.print(f"[red]Path is not a directory: {input_dir}")
        raise typer.Exit(1)

    asm_files = find_asm_files(input_dir)
    if not asm_files:
        console.print(f"[red]No .asm files found under: {input_dir}")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(asm_files)} ASM files")
    console.print("[yellow]Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)
    console.print("[green]Tokenizer loaded successfully")

    builder = DatasetBuilder(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        input_dir=input_dir,
        num_processes=num_processes,
        cleanup_temp_files=not no_cleanup,
        cache=cache,
    )

    builder.process_dataset(
        output_file,
        batch_size=batch_size,
        use_parallel=use_parallel,
        use_hf=use_hf,
    )


if __name__ == "__main__":
    app()
