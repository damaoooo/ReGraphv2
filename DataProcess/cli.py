"""
Command line interface for dataset builder
"""
import os
import sys
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
    ddg_so_path: str = typer.Option(
        DEFAULT_DDG_SO_PATH,
        "--lib-so-path", "-l",
        help="Path to DDG printer library"
    ),
    purify_so_path: str = typer.Option(
        DEFAULT_PURIFY_SO_PATH,
        "--purify-so-path", "-p",
        help="Path to metadata stripping library"
    ),
    cfg_so_path: str = typer.Option(
        DEFAULT_CFG_SO_PATH,
        "--cfg-so-path", "-c",
        help="Path to CFG printer library"
    ),
    use_hf: bool = typer.Option(False, "--use-hf", help="Use Hugging Face dataset instead of json"),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Batch size for processing (smaller for better load balancing)"),
    num_processes: int = typer.Option(None, "--num-processes", "-p", help="Number of processes to use"),
    pattern: str = typer.Option("**/*.ll", "--pattern", help="File pattern to match"),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Keep temporary files (dot and instrumented .ll)"),
    use_parallel: bool = typer.Option(False, "--parallel", help="Use parallel chunk processing instead of batched"),
    max_workers_multiplier: float = typer.Option(1.5, "--workers-multiplier", help="Multiplier for max workers (handles I/O bound operations)"),
    skip_filtering: bool = typer.Option(False, "--skip-filtering", help="Skip pre-filtering of small files"),
    cache: bool = typer.Option(True, "--no-cache", help="Disable caching of results")
):
    """Process LLVM IR files from a directory for dataset creation"""
    
    # Convert to absolute path and search for IR files
    console.print(f"[yellow]Searching for LLVM IR files in {input_dir} with pattern {pattern}")
    input_dir = os.path.abspath(input_dir)
    
    if not os.path.exists(input_dir):
        console.print(f"[red]Directory not found: {input_dir}")
        raise typer.Exit(1)
    
    if not os.path.isdir(input_dir):
        console.print(f"[red]Path is not a directory: {input_dir}")
        raise typer.Exit(1)
    
    cleanup_residual_temp_files(input_dir)
    
    # Find all LLVM IR files and convert to absolute paths
    file_paths = find_llvm_files(input_dir, pattern)
    file_paths = [os.path.abspath(f) for f in file_paths]
    
    # Remove those files end with '.dot' or '_instrumented.ll' or '_purified.ll'
    file_paths = [f for f in file_paths if not (f.endswith('.dot') or 
                                                f.endswith('_instrumented.ll') or 
                                                f.endswith('_purified.ll'))]
    
    console.print(f"[green]Found {len(file_paths)} files to process")
    
    if not file_paths:
        console.print("[red]No LLVM IR files found in the directory!")
        raise typer.Exit(1)
    
    # Load tokenizer
    console.print("[yellow]Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)
    console.print("[green]Tokenizer loaded successfully")
    
    # Create dataset builder
    builder = DatasetBuilder(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        file_list=file_paths,
        ddg_so_path=ddg_so_path,
        purify_so_path=purify_so_path,
        cfg_so_path=cfg_so_path,
        num_processes=num_processes,
        cleanup_temp_files=not no_cleanup,
        max_workers_multiplier=max_workers_multiplier,
        cache=cache
    )
    
    # Process files
    builder.process_dataset(output_file, batch_size, use_parallel, skip_filtering, use_hf=use_hf)
    cleanup_residual_temp_files(input_dir)


if __name__ == "__main__":
    app()
