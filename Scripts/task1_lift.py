#!/usr/bin/env python3
"""
Task 1: Lift binary files to LLVM IR using IDA Pro
"""
import os
import typer
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import console, run_command, ensure_directory, file_exists_and_not_empty

# Configuration
IDA_PATH = '/home/damaoooo/ida-pro-9.1/idat'
LIFT_SCRIPT = "/home/damaoooo/Downloads/regraphv2/Scripts/ida2llvm.py"
BINARY_PATH = "/home/damaoooo/Downloads/regraphv2/Binaries"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lift_task1_log.txt")

app = typer.Typer()

def lift_file(file_path: str, ida_path: str, lift_script: str, output_path: str):
    """Lift a single binary file to LLVM IR using IDA Pro"""
    command = [
        ida_path,
        '-c',
        '-A',
        '-S{} {}'.format(lift_script, output_path),
        '-L{}'.format(LOG_PATH),
        file_path,
    ]
    return run_command(command, f"Lifting {file_path}")

def lift_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    return lift_file(*args)

@app.command()
def main(
    input_path: str = typer.Option("", help="Input directory (defaults to DataProcess-1)"),
    ida_path: str = typer.Option(IDA_PATH, help="Path to IDA Pro"),
    output: str = typer.Option(..., help="Output directory"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
):
    """Lift binary files to LLVM IR using IDA Pro"""
    
    # Determine input path
    if input_path:
        db = os.path.basename(input_path)
        db_path = input_path
        if not os.path.exists(db_path):
            console.print(f"[red]Error: Input path {db_path} does not exist.[/red]")
            raise typer.Exit(code=1)
    else:
        db = "DataProcess-1"
        db_path = os.path.join(BINARY_PATH, db)
        if not os.path.exists(db_path):
            console.print(f"[red]Error: DataProcess-1 path {db_path} does not exist.[/red]")
            raise typer.Exit(code=1)

    output_path = os.path.join(output, db)
    ensure_directory(output_path)
    
    console.print(f"[green]Input: {db_path}[/green]")
    console.print(f"[green]Output: {output_path}[/green]")

    # Prepare lift commands
    console.print("[bold blue]Preparing lift tasks...[/bold blue]")
    lift_commands = []
    skipped_lift = 0
    
    for root, dirs, files in os.walk(db_path):
        for file in files:
            # Skip IDA database files
            if file.endswith((".i64", ".idb")):
                continue
                
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, db_path)
            output_dir = os.path.join(output_path, relative_path)
            ensure_directory(output_dir)
            output_file_path = os.path.join(output_dir, file) + ".ll"
            
            # Check if file already exists and we're resuming
            if resume and file_exists_and_not_empty(output_file_path):
                skipped_lift += 1
                continue
                
            cmd_arg = [file_path, ida_path, LIFT_SCRIPT, output_file_path]
            lift_commands.append(cmd_arg)

    if resume and skipped_lift > 0:
        console.print(f"[yellow]Skipping {skipped_lift} already lifted files[/yellow]")

    # Execute lifting
    console.print(f"[bold blue]Starting Task 1: Lifting {len(lift_commands)} binary files to LLVM IR[/bold blue]")
    
    if len(lift_commands) == 0:
        console.print("[yellow]No files to lift, task completed[/yellow]")
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
        lift_task = progress.add_task("Lifting binary files to LLVM IR", total=len(lift_commands))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_cmd = {executor.submit(lift_file_wrapper, cmd): cmd for cmd in lift_commands}
        
            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    success, stdout, stderr = future.result()
                    if not success:
                        console.print(f"[red]Failed to lift file: {cmd[0]}[/red]")
                        if stderr:
                            console.print(f"[red]Error: {stderr[:200]}...[/red]")
                        failed_count += 1
                    else:
                        success_count += 1
                except Exception as exc:
                    console.print(f"[red]File {cmd[0]} generated an exception: {exc}[/red]")
                    failed_count += 1
                finally:
                    progress.update(lift_task, advance=1)

    console.print(f"[bold green]Task 1 completed! Success: {success_count}, Failed: {failed_count}[/bold green]")

if __name__ == "__main__":
    app()
