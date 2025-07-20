#!/usr/bin/env python3
"""
Task 2: Re-optimize LLVM IR files using clang
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import typer
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from utils import console, run_command, ensure_directory, file_exists_and_not_empty

app = typer.Typer()

def reoptimize_file(file_path: str, output_path: str):
    """Re-optimize a single LLVM IR file using clang"""
    command = ["clang", "-m32", "-O3", "-c", "-emit-llvm", "-fno-inline", file_path, "-o", output_path]
    return run_command(command, f"Re-optimizing {file_path}")

def reoptimize_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    return reoptimize_file(*args)

@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .ll files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
):
    """Re-optimize LLVM IR files using clang"""
    
    if not os.path.exists(input_path):
        console.print(f"[red]Error: Input path {input_path} does not exist.[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[green]Processing directory: {input_path}[/green]")

    # Prepare re-optimization commands
    console.print("[bold blue]Preparing re-optimization tasks...[/bold blue]")
    reopt_commands = []
    skipped_reopt = 0
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".ll"):
                file_path = os.path.join(root, file)
                output_file_path = os.path.splitext(file_path)[0] + ".bc"
                
                # Check if file already exists and we're resuming
                if resume and file_exists_and_not_empty(output_file_path):
                    skipped_reopt += 1
                    continue
                    
                cmd_arg = [file_path, output_file_path]
                reopt_commands.append(cmd_arg)

    if resume and skipped_reopt > 0:
        console.print(f"[yellow]Skipping {skipped_reopt} already re-optimized files[/yellow]")

    # Execute re-optimization
    console.print(f"[bold blue]Starting Task 2: Re-optimizing {len(reopt_commands)} LLVM IR files[/bold blue]")
    
    if len(reopt_commands) == 0:
        console.print("[yellow]No files to re-optimize, task completed[/yellow]")
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
        reopt_task = progress.add_task("Re-optimizing LLVM IR files", total=len(reopt_commands))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_cmd = {executor.submit(reoptimize_file_wrapper, cmd): cmd for cmd in reopt_commands}
            
            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    success, stdout, stderr = future.result()
                    if not success:
                        console.print(f"[red]Failed to re-optimize file: {cmd[0]}[/red]")
                        if stderr:
                            console.print(f"[red]Error: {stderr[:200]}...[/red]")
                        failed_count += 1
                    else:
                        success_count += 1
                except Exception as exc:
                    console.print(f"[red]File {cmd[0]} generated an exception: {exc}[/red]")
                    failed_count += 1
                finally:
                    progress.update(reopt_task, advance=1)

    console.print(f"[bold green]Task 2 completed! Success: {success_count}, Failed: {failed_count}[/bold green]")

if __name__ == "__main__":
    app()
