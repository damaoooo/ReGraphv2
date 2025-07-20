#!/usr/bin/env python3
"""
Main pipeline controller for lift dataset operations
Can run individual tasks or the complete pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import typer
import subprocess
import multiprocessing
from utils import console

# Configuration
BINARY_PATH = "/home/damaoooo/Downloads/regraphv2/Binaries"

app = typer.Typer()

def run_task(script_name: str, args: list):
    """Run a task script with given arguments"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    command = ["python3", script_path] + args
    
    console.print(f"[blue]Running: {' '.join(command)}[/blue]")
    result = subprocess.run(command)
    return result.returncode == 0

@app.command()
def task1(
    input_path: str = typer.Option("", help="Input directory (defaults to DataProcess-1)"),
    ida_path: str = typer.Option("", help="Path to IDA Pro"),
    output: str = typer.Option(..., help="Output directory"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
):
    """Run Task 1: Lift binary files to LLVM IR"""
    args = ["--output", output, "--workers", str(workers)]
    if input_path:
        args.extend(["--input-path", input_path])
    if ida_path:
        args.extend(["--ida-path", ida_path])
    if resume:
        args.append("--resume")
    
    success = run_task("task1_lift.py", args)
    if success:
        console.print("[bold green]Task 1 completed successfully![/bold green]")
    else:
        console.print("[bold red]Task 1 failed![/bold red]")
        raise typer.Exit(code=1)

@app.command()
def task2(
    input_path: str = typer.Option(..., help="Input directory containing .ll files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
):
    """Run Task 2: Re-optimize LLVM IR files"""
    args = ["--input-path", input_path, "--workers", str(workers)]
    if resume:
        args.append("--resume")
    
    success = run_task("task2_reoptimize.py", args)
    if success:
        console.print("[bold green]Task 2 completed successfully![/bold green]")
    else:
        console.print("[bold red]Task 2 failed![/bold red]")
        raise typer.Exit(code=1)

@app.command()
def task3(
    input_path: str = typer.Option(..., help="Input directory containing .bc files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
):
    """Run Task 3: Extract individual functions"""
    args = ["--input-path", input_path, "--workers", str(workers)]
    if resume:
        args.append("--resume")
    
    success = run_task("task3_extract.py", args)
    if success:
        console.print("[bold green]Task 3 completed successfully![/bold green]")
    else:
        console.print("[bold red]Task 3 failed![/bold red]")
        raise typer.Exit(code=1)

@app.command()
def pipeline(
    db1: bool = typer.Option(False, help="Use DataProcess-1 as input"),
    input_path: str = typer.Option("", help="Custom input directory"),
    ida_path: str = typer.Option("", help="Path to IDA Pro"),
    output: str = typer.Option(..., help="Output directory"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
    start_from: int = typer.Option(1, help="Start from task number (1, 2, or 3)"),
):
    """Run the complete pipeline or start from a specific task"""
    
    if db1 and input_path:
        console.print("[red]Error: Cannot specify both --db1 and --input_path. Choose one.[/red]")
        raise typer.Exit(code=1)

    # Determine the actual input path
    if input_path:
        actual_input_path = input_path
    else:
        actual_input_path = os.path.join(BINARY_PATH, "DataProcess-1")
    
    if not os.path.exists(actual_input_path):
        console.print(f"[red]Error: Input path {actual_input_path} does not exist.[/red]")
        raise typer.Exit(code=1)

    db = os.path.basename(actual_input_path)
    final_output_path = os.path.join(output, db)

    console.print(f"[bold green]Starting pipeline from task {start_from}[/bold green]")
    console.print(f"[green]Input: {actual_input_path}[/green]")
    console.print(f"[green]Output: {final_output_path}[/green]")

    # Task 1: Lift binary files to LLVM IR
    if start_from <= 1:
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        console.print("[bold blue]TASK 1: Lifting binary files to LLVM IR[/bold blue]")
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        
        args = ["--output", output, "--workers", str(workers)]
        if input_path:
            args.extend(["--input-path", input_path])
        if ida_path:
            args.extend(["--ida-path", ida_path])
        if resume:
            args.append("--resume")
        
        if not run_task("task1_lift.py", args):
            console.print("[bold red]Task 1 failed! Pipeline stopped.[/bold red]")
            raise typer.Exit(code=1)

    # Task 2: Re-optimize LLVM IR files
    if start_from <= 2:
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        console.print("[bold blue]TASK 2: Re-optimizing LLVM IR files[/bold blue]")
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        
        args = ["--input-path", final_output_path, "--workers", str(workers)]
        if resume:
            args.append("--resume")
        
        if not run_task("task2_reoptimize.py", args):
            console.print("[bold red]Task 2 failed! Pipeline stopped.[/bold red]")
            raise typer.Exit(code=1)

    # Task 3: Extract individual functions
    if start_from <= 3:
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        console.print("[bold blue]TASK 3: Extracting individual functions[/bold blue]")
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        
        args = ["--input-path", final_output_path, "--workers", str(workers)]
        if resume:
            args.append("--resume")
        
        if not run_task("task3_extract.py", args):
            console.print("[bold red]Task 3 failed! Pipeline stopped.[/bold red]")
            raise typer.Exit(code=1)

    console.print("[bold green]=" * 60 + "[/bold green]")
    console.print("[bold green]ALL TASKS COMPLETED SUCCESSFULLY![/bold green]")
    console.print("[bold green]=" * 60 + "[/bold green]")

if __name__ == "__main__":
    app()
