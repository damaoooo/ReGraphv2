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
from utils import (
    console,
    run_command,
    ensure_directory,
    file_exists_and_not_empty,
    normalize_clang_opt_level,
)

app = typer.Typer()
TASK2_STATE_DIRNAME = ".task2_reoptimize_state"


def opt_level_state_token(opt_level: str) -> str:
    return opt_level.lstrip("-").replace(os.sep, "_")


def marker_path_for_output(input_root: str, output_path: str, opt_level: str) -> str:
    relative_output_path = os.path.relpath(output_path, input_root)
    return os.path.join(
        input_root,
        TASK2_STATE_DIRNAME,
        opt_level_state_token(opt_level),
        relative_output_path + ".done",
    )


def reoptimize_file(file_path: str, output_path: str, opt_level: str, marker_path: str):
    """Re-optimize a single LLVM IR file using clang"""
    if os.path.exists(marker_path):
        os.remove(marker_path)

    command = ["clang", "-m32", opt_level, "-c", "-emit-llvm", "-fno-inline", file_path, "-o", output_path]
    success, stdout, stderr = run_command(command, f"Re-optimizing {file_path} with {opt_level}")

    if success and not file_exists_and_not_empty(output_path):
        return False, stdout, f"clang finished successfully but output is missing: {output_path}"

    if success:
        ensure_directory(os.path.dirname(marker_path))
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(f"{opt_level}\n")

    return success, stdout, stderr

def reoptimize_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    return reoptimize_file(*args)

@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .ll files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
    opt_level: str = typer.Option(
        "O3",
        "--opt-level",
        help="clang optimization level for Task 2, e.g. O0, O1, O2, O3, Os, Oz",
    ),
):
    """Re-optimize LLVM IR files using clang"""

    normalized_input_path = os.path.abspath(input_path)
    normalized_opt_level = normalize_clang_opt_level(opt_level)

    if not os.path.exists(normalized_input_path):
        console.print(f"[red]Error: Input path {normalized_input_path} does not exist.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Processing directory: {normalized_input_path}[/green]")
    console.print(f"[green]Task 2 opt level: {normalized_opt_level}[/green]")

    # Prepare re-optimization commands
    console.print("[bold blue]Preparing re-optimization tasks...[/bold blue]")
    reopt_commands = []
    skipped_reopt = 0
    rerun_existing = 0

    for root, dirs, files in os.walk(normalized_input_path):
        if TASK2_STATE_DIRNAME in dirs:
            dirs.remove(TASK2_STATE_DIRNAME)

        for file in files:
            if file.endswith(".ll"):
                file_path = os.path.join(root, file)
                output_file_path = os.path.splitext(file_path)[0] + ".bc"
                marker_path = marker_path_for_output(
                    normalized_input_path,
                    output_file_path,
                    normalized_opt_level,
                )

                # Check if file already exists and we're resuming
                if resume and file_exists_and_not_empty(output_file_path) and os.path.exists(marker_path):
                    skipped_reopt += 1
                    continue

                if resume and file_exists_and_not_empty(output_file_path):
                    rerun_existing += 1

                cmd_arg = [file_path, output_file_path, normalized_opt_level, marker_path]
                reopt_commands.append(cmd_arg)

    if resume and skipped_reopt > 0:
        console.print(f"[yellow]Skipping {skipped_reopt} already re-optimized files[/yellow]")
    if resume and rerun_existing > 0:
        console.print(
            f"[yellow]Re-running {rerun_existing} existing .bc files because opt level changed or no resume marker was found[/yellow]"
        )

    # Execute re-optimization
    console.print(
        f"[bold blue]Starting Task 2: Re-optimizing {len(reopt_commands)} LLVM IR files with {normalized_opt_level}[/bold blue]"
    )
    
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
