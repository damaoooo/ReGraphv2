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

from task2_reoptimize import ARCH_TO_CLANG_FLAG, normalize_arch_mode, resolve_arch
from utils import console, ensure_directory, file_exists_and_not_empty, run_command

app = typer.Typer()
TASK4_STATE_DIRNAME = ".task4_recompile_state"
TASK4_CLANG_RECOMPILE_FLAGS = (
    "-fno-inline",
    "-fno-pic",
    "-fno-pie",
)


def arch_state_token(arch: str) -> str:
    return f"{arch}_noinline_nopic_nopie"


def marker_path_for_output(input_root: str, output_path: str, arch: str) -> str:
    relative_output_path = os.path.relpath(output_path, input_root)
    return os.path.join(
        input_root,
        TASK4_STATE_DIRNAME,
        arch_state_token(arch),
        relative_output_path + ".done",
    )


def recompile_file(file_path: str, output_path: str, arch: str, marker_path: str):
    """Recompile a single optimized .bc file to a .re binary artifact."""
    if os.path.exists(marker_path):
        os.remove(marker_path)

    arch_flag = ARCH_TO_CLANG_FLAG[arch]
    command = [
        "clang",
        arch_flag,
        "-c",
        *TASK4_CLANG_RECOMPILE_FLAGS,
        file_path,
        "-o",
        output_path,
    ]
    success, stdout, stderr = run_command(command, f"Recompiling {file_path}")

    if success and not file_exists_and_not_empty(output_path):
        return False, stdout, f"clang finished successfully but output is missing: {output_path}"

    if success:
        ensure_directory(os.path.dirname(marker_path))
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(f"arch={arch}\n")
            handle.write(f"arch_flag={arch_flag}\n")
            handle.write(f"flags={' '.join(TASK4_CLANG_RECOMPILE_FLAGS)}\n")

    return success, stdout, stderr


def recompile_file_wrapper(args):
    """Wrapper function for multiprocessing."""
    return recompile_file(*args)


@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .bc files"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
    arch: str = typer.Option(
        "auto",
        "--arch",
        help="Target bitness for clang: auto, m32, or m64. auto infers from dataset-style file names.",
    ),
):
    """Recompile optimized .bc files to .re files with clang."""

    normalized_input_path = os.path.abspath(input_path)
    normalized_arch = normalize_arch_mode(arch)

    if not os.path.exists(normalized_input_path):
        console.print(f"[red]Error: Input path {normalized_input_path} does not exist.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Processing directory: {normalized_input_path}[/green]")
    console.print(f"[green]Task 4 arch mode: {normalized_arch}[/green]")
    console.print(f"[green]Task 4 clang flags: -m32/-m64 {' '.join(TASK4_CLANG_RECOMPILE_FLAGS)}[/green]")

    console.print("[bold blue]Preparing recompilation tasks...[/bold blue]")
    recompile_commands = []
    skipped_recompile = 0
    rerun_existing = 0
    arch_counts = {"m32": 0, "m64": 0}
    unknown_arch_files = []

    for root, dirs, files in os.walk(normalized_input_path):
        if TASK4_STATE_DIRNAME in dirs:
            dirs.remove(TASK4_STATE_DIRNAME)

        for file in files:
            if not file.endswith(".bc"):
                continue

            file_path = os.path.join(root, file)
            resolved_arch = resolve_arch(file_path, normalized_arch)
            if resolved_arch is None:
                unknown_arch_files.append(file_path)
                continue

            arch_counts[resolved_arch] += 1
            output_file_path = os.path.splitext(file_path)[0] + ".re"
            marker_path = marker_path_for_output(
                normalized_input_path,
                output_file_path,
                resolved_arch,
            )

            if resume and file_exists_and_not_empty(output_file_path) and os.path.exists(marker_path):
                skipped_recompile += 1
                continue

            if resume and file_exists_and_not_empty(output_file_path):
                rerun_existing += 1

            recompile_commands.append([file_path, output_file_path, resolved_arch, marker_path])

    if unknown_arch_files:
        console.print(
            f"[red]Error: could not infer 32/64-bit clang mode for {len(unknown_arch_files)} .bc files.[/red]"
        )
        for file_path in unknown_arch_files[:10]:
            console.print(f"[red]  - {file_path}[/red]")
        console.print("[red]Rename files with x86/x64/arm32/arm64/mips32/mips64, or pass --arch m32/--arch m64.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Task 4 resolved arches: m32={arch_counts['m32']}, m64={arch_counts['m64']}[/green]")

    if resume and skipped_recompile > 0:
        console.print(f"[yellow]Skipping {skipped_recompile} already recompiled files[/yellow]")
    if resume and rerun_existing > 0:
        console.print(
            f"[yellow]Re-running {rerun_existing} existing .re files because clang flags changed or no resume marker was found[/yellow]"
        )

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
