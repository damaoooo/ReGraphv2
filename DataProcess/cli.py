"""
Command line interface for dataset builder
"""
import os
import sys
import sqlite3
import typer
from rich.console import Console

from Tokenizer.ir_tokenizer import load_tokenizer
from Utils.utils import DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH, DEFAULT_CFG_SO_PATH, DEFAULT_TOKENIZER_PATH

from .dataset_builder_new import DatasetBuilder
from .dataset_utils import find_llvm_files, cleanup_residual_temp_files



# Initialize typer app and rich console
app = typer.Typer(help="LLVM IR Dataset Builder with Graph Structures")
console = Console()


@app.command("directory")
def process_directory(
    input_dir: str = typer.Argument(..., help="Directory containing LLVM IR files"),
    output_file: str = typer.Argument(..., help="Output JSON file for results"),
    tokenizer_path: str = typer.Option(
        DEFAULT_TOKENIZER_PATH,
        "--tokenizer-path", "-t",
        help="Path to tokenizer file"
    ),
    use_hf: bool = typer.Option(False, "--use-hf", help="Use Hugging Face dataset instead of json"),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Batch size for processing (smaller for better load balancing)"),
    num_processes: int = typer.Option(None, "--num-processes", "-p", help="Number of processes to use"),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Keep temporary files (dot and instrumented .ll)"),
    use_parallel: bool = typer.Option(False, "--parallel", help="Use parallel chunk processing instead of batched"),
    cache: bool = typer.Option(True, "--no-cache", help="Disable caching of results")
):
    """Process LLVM IR files from a directory for dataset creation"""
    
    # Convert to absolute path and search for IR files
    input_dir = os.path.abspath(input_dir)
    
    if not os.path.exists(input_dir):
        console.print(f"[red]Directory not found: {input_dir}")
        raise typer.Exit(1)
    
    if not os.path.isdir(input_dir):
        console.print(f"[red]Path is not a directory: {input_dir}")
        raise typer.Exit(1)

    db_file_origin = os.path.join(input_dir, 'results.db')
    db_file = f'file:{db_file_origin}?mode=ro'
    if not os.path.exists(db_file_origin):
        console.print(f"[red]Database file not found: {db_file_origin}")
        raise typer.Exit(1)

    # Load tokenizer
    console.print("[yellow]Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)
    console.print("[green]Tokenizer loaded successfully")
    
    # Create dataset builder
    builder = DatasetBuilder(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        db_file=db_file,
        num_processes=num_processes,
        cleanup_temp_files=not no_cleanup,
        cache=cache
    )
    
    # Process files
    builder.process_dataset(output_file, batch_size, use_parallel, use_hf=use_hf)
    # cleanup_residual_temp_files(input_dir)


if __name__ == "__main__":
    app()
