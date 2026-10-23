#!/usr/bin/env python3
"""
Task 3: Extract individual functions from LLVM IR files
"""
import sys
import os
import json
import shutil
import tempfile
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import typer
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from utils import console, run_command
from pipeline_cache import (
    build_input_signature,
    compute_file_sha256,
    get_tool_identity,
    iter_completed_tasks,
)

# Configuration
EXTRACT_SCRIPT = str(Path(__file__).with_name('split_llvm_ir.sh'))

app = typer.Typer()

def build_extraction_signature(file_path: str) -> dict:
    """Bind extracted functions to the input module and extraction tools."""
    extraction_config = {
        'script': compute_file_sha256(EXTRACT_SCRIPT),
        'extractor': compute_file_sha256(__file__),
        'tools': [get_tool_identity(tool) for tool in ('llvm-nm', 'llvm-extract')],
    }
    return build_input_signature(file_path, extraction_config)


def is_extraction_complete(file_path: str, output_path: str) -> bool:
    """Check the manifest and every extracted file before skipping a module."""
    try:
        output_directory = Path(output_path)
        completion_record = json.loads((output_directory / '.complete.json').read_text())
        if completion_record['signature'] != build_extraction_signature(file_path):
            return False
        actual_filenames = {
            path.name for path in output_directory.iterdir() if path.name != '.complete.json'
        }
        expected_checksums = completion_record['files']
        if actual_filenames != set(expected_checksums):
            return False
        return all(
            compute_file_sha256(output_directory / name) == checksum
            for name, checksum in expected_checksums.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def extract_functions(file_path: str, output_path: str) -> tuple[bool, str, str]:
    """Extract into a staging directory, then publish the complete function set."""
    expected_signature = build_extraction_signature(file_path)
    output_directory = Path(output_path)
    staging_directory = Path(tempfile.mkdtemp(prefix='.extract-', dir=output_directory.parent))
    backup_directory = output_directory.with_name(f'.{output_directory.name}.previous')
    try:
        command = ['bash', EXTRACT_SCRIPT, file_path, str(staging_directory)]
        success, stdout, stderr = run_command(command, f"Extracting functions from {file_path}")
        if not success:
            return success, stdout, stderr

        file_checksums = {
            path.name: compute_file_sha256(path)
            for path in staging_directory.iterdir() if path.is_file()
        }
        completion_record = {'signature': expected_signature, 'files': file_checksums}
        (staging_directory / '.complete.json').write_text(json.dumps(completion_record))

        # Move the previous set aside before publishing; only complete sets are visible.
        if backup_directory.exists():
            shutil.rmtree(backup_directory)
        if output_directory.exists():
            os.replace(output_directory, backup_directory)
        os.replace(staging_directory, output_directory)
        if backup_directory.exists():
            shutil.rmtree(backup_directory)
        return success, stdout, stderr
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)

def extract_functions_wrapper(args):
    """Wrapper function for multiprocessing"""
    return extract_functions(*args)

@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .bc files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
):
    """Extract individual functions from LLVM IR files"""
    if workers < 1:
        raise typer.BadParameter('workers must be positive')
    
    if not os.path.exists(input_path):
        console.print(f"[red]Error: Input path {input_path} does not exist.[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[green]Processing directory: {input_path}[/green]")

    # Prepare function extraction commands
    console.print("[bold blue]Preparing function extraction tasks...[/bold blue]")
    extract_commands = []
    skipped_extract = 0
    
    for root, dirs, files in os.walk(input_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and not d.endswith(('_functions', '_functions.previous'))]
        for file in files:
            if file.endswith(".bc"):
                file_path = os.path.join(root, file)
                output_folder_path = os.path.splitext(file_path)[0] + "_functions"
                
                # Check if folder already exists and has files (resuming)
                if resume and is_extraction_complete(file_path, output_folder_path):
                    skipped_extract += 1
                    continue
                    
                cmd_arg = [file_path, output_folder_path]
                extract_commands.append(cmd_arg)

    if resume and skipped_extract > 0:
        console.print(f"[yellow]Skipping {skipped_extract} already extracted function sets[/yellow]")

    # Execute function extraction
    console.print(f"[bold blue]Starting Task 3: Extracting functions from {len(extract_commands)} files[/bold blue]")
    
    if len(extract_commands) == 0:
        console.print("[yellow]No files to extract functions from, task completed[/yellow]")
        return

    success_count = 0
    failed_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        extract_task = progress.add_task("Extracting individual functions", total=len(extract_commands))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for future, cmd in iter_completed_tasks(executor, extract_functions_wrapper, extract_commands, workers * 2):
                try:
                    success, stdout, stderr = future.result()
                    if not success:
                        console.print(f"[red]Failed to extract functions from file: {cmd[0]}[/red]")
                        if stderr:
                            console.print(f"[red]Error: {stderr[:200]}...[/red]")
                        failed_count += 1
                    else:
                        success_count += 1
                except Exception as exc:
                    console.print(f"[red]File {cmd[0]} generated an exception: {exc}[/red]")
                    failed_count += 1
                finally:
                    progress.update(extract_task, advance=1)

    console.print(f"[bold green]Task 3 completed! Success: {success_count}, Failed: {failed_count}[/bold green]")
    if failed_count:
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
