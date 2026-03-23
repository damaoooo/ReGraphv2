#!/usr/bin/env python3
"""
Task 4: Recompile optimized LLVM bitcode files (.bc) to .re binaries.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from utils import console, file_exists_and_not_empty, run_command

app = typer.Typer()


def recompile_file(file_path: str, output_path: str):
    """Recompile a single optimized .bc file to a .re binary artifact."""
    command = ["clang", "-c", "-fno-inline", file_path, "-o", output_path]
    return run_command(command, f"Recompiling {file_path}")


def recompile_file_wrapper(args):
    """Wrapper function for multiprocessing."""
    return recompile_file(*args)


@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .bc files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
):
    """Recompile optimized .bc files to .re files with clang."""

    if not os.path.exists(input_path):
        console.print(f"[red]Error: Input path {input_path} does not exist.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Processing directory: {input_path}[/green]")

    console.print("[bold blue]Preparing recompilation tasks...[/bold blue]")
    recompile_commands = []
    skipped_recompile = 0

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if not file.endswith(".bc"):
                continue

            file_path = os.path.join(root, file)
            output_file_path = os.path.splitext(file_path)[0] + ".re"

            if resume and file_exists_and_not_empty(output_file_path):
                skipped_recompile += 1
                continue

            recompile_commands.append([file_path, output_file_path])

    if resume and skipped_recompile > 0:
        console.print(f"[yellow]Skipping {skipped_recompile} already recompiled files[/yellow]")

    console.print(f"[bold blue]Starting Task 4: Recompiling {len(recompile_commands)} bitcode files[/bold blue]")

    if len(recompile_commands) == 0:
        console.print("[yellow]No files to recompile, task completed[/yellow]")
        return

    success_count = 0
    failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        recompile_task = progress.add_task("Recompiling bitcode files", total=len(recompile_commands))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_cmd = {
                executor.submit(recompile_file_wrapper, cmd): cmd for cmd in recompile_commands
            }

            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    success, stdout, stderr = future.result()
                    if not success:
                        console.print(f"[red]Failed to recompile file: {cmd[0]}[/red]")
                        if stderr:
                            console.print(f"[red]Error: {stderr[:200]}...[/red]")
                        failed_count += 1
                    else:
                        success_count += 1
                except Exception as exc:
                    console.print(f"[red]File {cmd[0]} generated an exception: {exc}[/red]")
                    failed_count += 1
                finally:
                    progress.update(recompile_task, advance=1)

    console.print(
        f"[bold green]Task 4 completed! Success: {success_count}, Failed: {failed_count}[/bold green]"
    )


if __name__ == "__main__":
    app()
