#!/usr/bin/env python3
"""
Task 3: Extract individual functions from LLVM IR files
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import typer
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from utils import console, run_command, ensure_directory, directory_exists_and_not_empty

# Configuration
EXTRACT_SCRIPT = "/home/damaoooo/Downloads/regraphv2/Scripts/split_llvm_ir.sh"

app = typer.Typer()

def extract_functions(file_path: str, output_path: str):
    """Extract individual functions from a LLVM IR file"""
    command = [EXTRACT_SCRIPT, file_path, output_path]
    return run_command(command, f"Extracting functions from {file_path}")

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
    
    if not os.path.exists(input_path):
        console.print(f"[red]Error: Input path {input_path} does not exist.[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[green]Processing directory: {input_path}[/green]")

    # Prepare function extraction commands
    console.print("[bold blue]Preparing function extraction tasks...[/bold blue]")
    extract_commands = []
    skipped_extract = 0
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".bc"):
                file_path = os.path.join(root, file)
                output_folder_path = os.path.splitext(file_path)[0] + "_functions"
                
                # Check if folder already exists and has files (resuming)
                if resume and directory_exists_and_not_empty(output_folder_path):
                    skipped_extract += 1
                    continue
                    
                ensure_directory(output_folder_path)
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
            future_to_cmd = {executor.submit(extract_functions_wrapper, cmd): cmd for cmd in extract_commands}
            
            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
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

if __name__ == "__main__":
    app()
