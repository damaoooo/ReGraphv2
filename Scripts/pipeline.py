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
from utils import console, normalize_clang_opt_level

# Configuration
BINARY_PATH = "/home/damaoooo/Downloads/regraphv2/Binaries"

app = typer.Typer()

def run_task(script_name: str, args: list):
    """Run a task script with given arguments"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    command = [sys.executable, script_path] + args
    
    console.print(f"[blue]Running: {' '.join(command)}[/blue]")
    result = subprocess.run(command)
    return result.returncode == 0

@app.command()
def task1(
    input_path: str = typer.Option("", help="Input directory (defaults to DataProcess-1)"),
    output: str = typer.Option(..., help="Output directory"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
    backend: str = typer.Option(
        "ida",
        "--backend",
        help="Lifter backend: ida or ghidra",
    ),
    conda_env: str = typer.Option(
        os.environ.get("REGRAPH_CONDA_ENV", ""),
        "--conda-env",
        help="Conda environment for lifter subprocesses; empty uses current Python",
    ),
    ida_path: str = typer.Option("", "--ida-path", help="Path to IDA Pro for the ida backend"),
    ghidra_home: str = typer.Option("", "--ghidra-home", help="Ghidra installation root for the ghidra backend"),
    analyze_headless: str = typer.Option(
        "",
        "--analyze-headless",
        help="Path to Ghidra support/analyzeHeadless for the ghidra backend",
    ),
    ghidra_target: str = typer.Option(
        "host",
        "--ghidra-target",
        help="pcode2llvm.py target triple source for the ghidra backend: host or ghidra",
    ),
    ghidra_max_cpu: int = typer.Option(1, "--ghidra-max-cpu", help="Ghidra CPUs per task1 worker"),
    ghidra_decompile_timeout: int = typer.Option(
        60,
        "--ghidra-decompile-timeout",
        help="Ghidra decompiler timeout per function in seconds",
    ),
    ghidra_analysis_timeout: int = typer.Option(
        300,
        "--ghidra-analysis-timeout",
        help="Ghidra analysis timeout per binary in seconds",
    ),
    ghidra_no_analysis: bool = typer.Option(
        False,
        "--ghidra-no-analysis",
        help="Skip Ghidra auto-analysis for debugging",
    ),
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial",
        help="Allow partial Ghidra lifts. Off by default for reproducibility",
    ),
    start_from_step2: bool = typer.Option(
        False,
        help="Start directly from Step 2 (scan .i64 and lift), skip Step 1",
    ),
):
    """Run Task 1: Lift binary files to LLVM IR"""
    args = ["--output", output, "--workers", str(workers)]
    args.extend(["--backend", backend])
    if conda_env:
        args.extend(["--conda-env", conda_env])
    if input_path:
        args.extend(["--input-path", input_path])
    if ida_path:
        args.extend(["--ida-path", ida_path])
    if ghidra_home:
        args.extend(["--ghidra-home", ghidra_home])
    if analyze_headless:
        args.extend(["--analyze-headless", analyze_headless])
    args.extend(["--ghidra-target", ghidra_target])
    args.extend(["--ghidra-max-cpu", str(ghidra_max_cpu)])
    args.extend(["--ghidra-decompile-timeout", str(ghidra_decompile_timeout)])
    args.extend(["--ghidra-analysis-timeout", str(ghidra_analysis_timeout)])
    if ghidra_no_analysis:
        args.append("--ghidra-no-analysis")
    if resume:
        args.append("--resume")
    if allow_partial:
        args.append("--allow-partial")
    if start_from_step2:
        args.append("--start-from-step2")
    
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
    opt_level: str = typer.Option(
        "O3",
        "--opt-level",
        help="Task 2 optimization level, e.g. O0, O1, O2, O3, Os, Og, Oz, Oc, Oc2",
    ),
    arch: str = typer.Option(
        "auto",
        "--arch",
        help="Target bitness for Task 2 clang: auto, m32, or m64",
    ),
):
    """Run Task 2: Re-optimize LLVM IR files"""
    normalized_opt_level = normalize_clang_opt_level(opt_level)
    args = ["--input-path", input_path, "--workers", str(workers)]
    args.extend(["--opt-level", normalized_opt_level])
    args.extend(["--arch", arch])
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
def task4(
    input_path: str = typer.Option(..., help="Input directory containing .bc files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
    arch: str = typer.Option(
        "auto",
        "--arch",
        help="Target bitness for Task 4 clang: auto, m32, or m64",
    ),
):
    """Run Task 4: Recompile optimized .bc files to .re artifacts"""
    args = ["--input-path", input_path, "--workers", str(workers)]
    args.extend(["--arch", arch])
    if resume:
        args.append("--resume")

    success = run_task("task4_recompile.py", args)
    if success:
        console.print("[bold green]Task 4 completed successfully![/bold green]")
    else:
        console.print("[bold red]Task 4 failed![/bold red]")
        raise typer.Exit(code=1)

@app.command()
def pipeline(
    db1: bool = typer.Option(False, help="Use DataProcess-1 as input"),
    input_path: str = typer.Option("", help="Custom input directory"),
    ida_path: str = typer.Option("", help="Path to IDA Pro"),
    lifter_backend: str = typer.Option("ida", "--lifter-backend", help="Task 1 lifter backend: ida or ghidra"),
    conda_env: str = typer.Option(
        os.environ.get("REGRAPH_CONDA_ENV", ""),
        "--conda-env",
        help="Conda environment for Task 1 lifter subprocesses; empty uses current Python",
    ),
    ghidra_home: str = typer.Option("", "--ghidra-home", help="Ghidra installation root for Task 1"),
    analyze_headless: str = typer.Option(
        "",
        "--analyze-headless",
        help="Path to Ghidra support/analyzeHeadless for Task 1",
    ),
    ghidra_target: str = typer.Option(
        "host",
        "--ghidra-target",
        help="pcode2llvm.py target triple source for Task 1 Ghidra backend: host or ghidra",
    ),
    ghidra_max_cpu: int = typer.Option(1, "--ghidra-max-cpu", help="Ghidra CPUs per Task 1 worker"),
    ghidra_decompile_timeout: int = typer.Option(
        60,
        "--ghidra-decompile-timeout",
        help="Ghidra decompiler timeout per function in seconds",
    ),
    ghidra_analysis_timeout: int = typer.Option(
        300,
        "--ghidra-analysis-timeout",
        help="Ghidra analysis timeout per binary in seconds",
    ),
    ghidra_no_analysis: bool = typer.Option(
        False,
        "--ghidra-no-analysis",
        help="Skip Ghidra auto-analysis for Task 1 debugging",
    ),
    task1_allow_partial: bool = typer.Option(
        False,
        "--task1-allow-partial",
        help="Allow partial Ghidra lifts in Task 1. Off by default for reproducibility",
    ),
    output: str = typer.Option(..., help="Output directory"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
    start_from: int = typer.Option(1, help="Start from task number (1, 2, 3, or 4)"),
    task1_start_from_step2: bool = typer.Option(
        False,
        help="Task 1: start directly from Step 2 (scan .i64 and lift), skip Step 1",
    ),
    enable_recompile: bool = typer.Option(
        False,
        help="Enable optional Task 4 recompile (.bc -> .re). Disabled by default",
    ),
    opt_level: str = typer.Option(
        "O3",
        "--opt-level",
        help="Task 2 optimization level, e.g. O0, O1, O2, O3, Os, Og, Oz, Oc, Oc2",
    ),
    arch: str = typer.Option(
        "auto",
        "--arch",
        help="Target bitness for Task 2 clang: auto, m32, or m64",
    ),
):
    """Run the complete pipeline or start from a specific task"""
    normalized_opt_level = normalize_clang_opt_level(opt_level)

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
    final_output_path = output

    console.print(f"[bold green]Starting pipeline from task {start_from}[/bold green]")
    console.print(f"[green]Input: {actual_input_path}[/green]")
    console.print(f"[green]Output: {final_output_path}[/green]")
    console.print(f"[green]Task 1 lifter backend: {lifter_backend}[/green]")
    console.print(f"[green]Task 2 opt level: {normalized_opt_level}[/green]")
    console.print(f"[green]Task 2 arch mode: {arch}[/green]")

    # Task 1: Lift binary files to LLVM IR
    if start_from <= 1:
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        console.print("[bold blue]TASK 1: Lifting binary files to LLVM IR[/bold blue]")
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        
        args = ["--output", output, "--workers", str(workers), "--backend", lifter_backend]
        if conda_env:
            args.extend(["--conda-env", conda_env])
        if input_path:
            args.extend(["--input-path", input_path])
        if ida_path:
            args.extend(["--ida-path", ida_path])
        if ghidra_home:
            args.extend(["--ghidra-home", ghidra_home])
        if analyze_headless:
            args.extend(["--analyze-headless", analyze_headless])
        args.extend(["--ghidra-target", ghidra_target])
        args.extend(["--ghidra-max-cpu", str(ghidra_max_cpu)])
        args.extend(["--ghidra-decompile-timeout", str(ghidra_decompile_timeout)])
        args.extend(["--ghidra-analysis-timeout", str(ghidra_analysis_timeout)])
        if ghidra_no_analysis:
            args.append("--ghidra-no-analysis")
        if resume:
            args.append("--resume")
        if task1_allow_partial:
            args.append("--allow-partial")
        if task1_start_from_step2:
            args.append("--start-from-step2")
        
        if not run_task("task1_lift.py", args):
            console.print("[bold red]Task 1 failed! Pipeline stopped.[/bold red]")
            raise typer.Exit(code=1)

    # Task 2: Re-optimize LLVM IR files
    if start_from <= 2:
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        console.print("[bold blue]TASK 2: Re-optimizing LLVM IR files[/bold blue]")
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        
        args = ["--input-path", final_output_path, "--workers", str(workers)]
        args.extend(["--opt-level", normalized_opt_level])
        args.extend(["--arch", arch])
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
        args.extend(["--arch", arch])
        if resume:
            args.append("--resume")
        
        if not run_task("task3_extract.py", args):
            console.print("[bold red]Task 3 failed! Pipeline stopped.[/bold red]")
            raise typer.Exit(code=1)

    # Task 4: Recompile .bc files to .re (optional)
    if enable_recompile and start_from <= 4:
        console.print("[bold blue]=" * 60 + "[/bold blue]")
        console.print("[bold blue]TASK 4: Recompiling .bc files to .re[/bold blue]")
        console.print("[bold blue]=" * 60 + "[/bold blue]")

        args = ["--input-path", final_output_path, "--workers", str(workers)]
        if resume:
            args.append("--resume")

        if not run_task("task4_recompile.py", args):
            console.print("[bold red]Task 4 failed! Pipeline stopped.[/bold red]")
            raise typer.Exit(code=1)

    console.print("[bold green]=" * 60 + "[/bold green]")
    console.print("[bold green]ALL TASKS COMPLETED SUCCESSFULLY![/bold green]")
    console.print("[bold green]=" * 60 + "[/bold green]")

if __name__ == "__main__":
    app()
