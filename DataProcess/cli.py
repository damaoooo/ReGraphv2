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
from .provenance import DEFAULT_POLICY, load_policy, select_inputs, write_selection
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
    cache: bool = typer.Option(True, "--no-cache", help="Disable caching of results"),
    selection_policy: str = typer.Option(
        DEFAULT_POLICY, "--selection-policy",
        help="Versioned JSON policy deciding which verified functions enter this build (default keeps all, "
             "recording approximation tags)"),
    allow_legacy_inputs: bool = typer.Option(
        False, "--allow-legacy-inputs",
        help="Also use function files/rows without pipeline state (marked legacy-unverified in provenance)"),
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

    # 完整性检查 + 数据选择策略：只用状态有效、hash 一致的函数；记录来源与选择结果
    policy = load_policy(selection_policy)
    file_list, provenance, summary = select_inputs(input_dir, db_file_origin, policy, allow_legacy_inputs)
    previous = os.path.join(output_file, "selection.json")
    if os.path.exists(os.path.join(output_file, "progress.txt")) and os.path.exists(previous):
        import json
        with open(previous) as handle:
            old_digest = json.load(handle).get("selection_digest")
        if old_digest != summary["selection_digest"]:
            console.print("[red]Output directory was built from a different input selection; "
                          "resuming would mix stale and current data. Use a new output directory.")
            raise typer.Exit(1)
    write_selection(output_file, provenance, summary)
    console.print(f"[green]Selection {summary['policy']['name']}: {summary['selected']} selected of "
                  f"{summary['candidates']} verified candidates; integrity rejections: {summary['rejected_for_integrity']}; "
                  f"policy exclusions: {summary['excluded_by_policy']}")

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
    builder.process_dataset(output_file, batch_size, use_parallel, use_hf=use_hf, file_list=file_list)
    # cleanup_residual_temp_files(input_dir)


if __name__ == "__main__":
    app()
