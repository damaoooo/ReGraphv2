#!/usr/bin/env python3
"""
Task: Export binaries to per-function ASM via IDA Pro (two-step process).

Step 1: Generate missing .i64 database files with IDA Pro.
Step 2: Export every function from each .i64 into a *_functions directory.
"""
import logging
import multiprocessing
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import console, ensure_directory, file_exists_and_not_empty


# Configuration
BINARY_PATH = "/home/damaoooo/Downloads/regraphv2/Binaries"
IDA_PATH = "/home/damaoooo/ida-pro-9.3"
IDA2ASM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ida2asm.py")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin2asm_log.txt")
DEFAULT_CONDA_ENV = "ReLL"

SKIPPED_BINARY_SUFFIXES = (
    ".i64",
    ".idb",
    ".id0",
    ".id1",
    ".id2",
    ".til",
    ".nam",
    ".asm",
    ".ll",
    ".bc",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".txt",
    ".log",
    ".md",
    ".py",
    ".sh",
)

IDA_TEMP_SUFFIXES = (
    ".idb",
    ".id0",
    ".id1",
    ".id2",
    ".til",
    ".nam",
)

app = typer.Typer()


def trim_output(output: str, limit: int = 4000) -> str:
    """Trim subprocess output before writing it into the shared log."""
    if not output:
        return ""
    if len(output) <= limit:
        return output
    return output[:limit] + "\n...[truncated]...\n"


def is_candidate_binary(file_name: str) -> bool:
    """Return True if the file looks like an input binary."""
    if file_name.startswith("."):
        return False

    return not file_name.lower().endswith(SKIPPED_BINARY_SUFFIXES)


def directory_contains_asm(path: str) -> bool:
    """Check whether a directory contains at least one .asm file."""
    if not os.path.isdir(path):
        return False

    for entry in os.listdir(path):
        file_path = os.path.join(path, entry)
        if os.path.isfile(file_path) and entry.endswith(".asm"):
            return True
    return False


def cleanup_ida_temp_files(input_root: str, logger: logging.Logger) -> None:
    """Remove stale temporary IDA files that can block .i64 regeneration."""
    removed = 0
    for root, dirs, files in os.walk(input_root):
        dirs[:] = [
            directory
            for directory in dirs
            if not directory.startswith(".")
            and directory != "__pycache__"
            and not directory.endswith("_functions")
        ]
        for file_name in files:
            if not file_name.lower().endswith(IDA_TEMP_SUFFIXES):
                continue
            file_path = os.path.join(root, file_name)
            try:
                os.remove(file_path)
                removed += 1
                logger.info("Removed stale IDA temp file: %s", file_path)
            except Exception as exc:
                logger.warning("Could not remove %s: %s", file_path, exc)

    if removed > 0:
        logger.info("Removed %s stale IDA temp files before Step 1", removed)


def generate_i64(binary_path: str, ida_path: str) -> tuple:
    """Generate a .i64 file for one binary with IDA Pro."""
    command = [
        os.path.join(ida_path, "idat"),
        "-A",
        "-B",
        binary_path,
    ]
    i64_path = binary_path + ".i64"

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        success = result.returncode == 0 and file_exists_and_not_empty(i64_path)
        if success:
            generated_asm = binary_path + ".asm"
            if os.path.exists(generated_asm):
                try:
                    os.remove(generated_asm)
                except OSError:
                    pass
        return (
            success,
            binary_path,
            command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
    except Exception as exc:
        return (
            False,
            binary_path,
            command,
            -1,
            "",
            f"{type(exc).__name__}: {exc}",
        )


def build_output_dir(i64_path: str, input_root: str, output_root: str) -> str:
    """Build the *_functions directory path for one .i64 file."""
    binary_path = os.path.splitext(i64_path)[0]
    folder_name = os.path.splitext(os.path.basename(binary_path))[0] + "_functions"

    if output_root:
        relative_dir = os.path.relpath(os.path.dirname(i64_path), input_root)
        if relative_dir == ".":
            return os.path.join(output_root, folder_name)
        return os.path.join(output_root, relative_dir, folder_name)

    return os.path.join(os.path.dirname(binary_path), folder_name)


def build_ida2asm_command(
    i64_path: str,
    output_dir: str,
    conda_env: str,
    save_database: bool,
) -> list[str]:
    """Build the ida2asm.py invocation."""
    command = []
    if conda_env:
        command.extend(["conda", "run", "-n", conda_env, "python"])
    else:
        command.append(sys.executable)

    command.extend(
        [
            IDA2ASM_PATH,
            "-f",
            i64_path,
            "-o",
            output_dir,
            "--log-file",
            os.path.join(output_dir, "ida2asm.log"),
        ]
    )

    if save_database:
        command.append("--save-database")

    return command


def export_single_i64(
    i64_path: str,
    output_dir: str,
    conda_env: str,
    save_database: bool,
) -> tuple:
    """Export one .i64 database into its *_functions directory."""
    ensure_directory(output_dir)
    command = build_ida2asm_command(i64_path, output_dir, conda_env, save_database)

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        generated_asm = directory_contains_asm(output_dir)
        success = generated_asm
        return (
            success,
            i64_path,
            output_dir,
            command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
    except Exception as exc:
        return (
            False,
            i64_path,
            output_dir,
            command,
            -1,
            "",
            f"{type(exc).__name__}: {exc}",
        )


@app.command()
def main(
    input_path: str = typer.Option("", help="Input directory (defaults to DataProcess-1)"),
    output: str = typer.Option(
        "",
        help="Output root directory. If omitted, create *_functions beside each input file.",
    ),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run and skip existing outputs"),
    start_from_step2: bool = typer.Option(
        False,
        help="Start directly from Step 2, skip .i64 generation and only scan existing .i64 files.",
    ),
    conda_env: str = typer.Option(
        DEFAULT_CONDA_ENV,
        help="Conda environment used to run ida2asm.py. Use empty string to use the current Python.",
    ),
    ida_path: str = typer.Option(
        IDA_PATH,
        help="Path to the IDA Pro installation used for .i64 generation.",
    ),
    save_database: bool = typer.Option(
        False,
        help="Pass --save-database through to ida2asm.py.",
    ),
):
    """
    Generate .i64 databases first, then export every function to ASM.
    """
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    separator = "=" * 60
    logger.info(separator)
    logger.info("bin2asm started at %s", datetime.now())
    logger.info(
        "Workers: %s, Resume: %s, StartFromStep2: %s, CondaEnv: %s, IDAPath: %s, SaveDatabase: %s",
        workers,
        resume,
        start_from_step2,
        conda_env or "<current-python>",
        ida_path,
        save_database,
    )
    logger.info(separator)

    if input_path:
        db = os.path.basename(input_path.rstrip("/"))
        db_path = input_path
    else:
        db = "DataProcess-1"
        db_path = os.path.join(BINARY_PATH, db)

    if not os.path.exists(db_path):
        console.print(f"[red]Error: Input path {db_path} does not exist.[/red]")
        raise typer.Exit(code=1)

    if not start_from_step2:
        idat_path = os.path.join(ida_path, "idat")
        if not os.path.exists(idat_path):
            console.print(f"[red]Error: IDA idat not found at {idat_path}[/red]")
            raise typer.Exit(code=1)

    output_root = ""
    if output:
        output_root = os.path.join(output, db)
        ensure_directory(output_root)

    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Binary to ASM Export[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[green]Input:  {db_path}[/green]")
    if output_root:
        console.print(f"[green]Output: {output_root}[/green]")
    else:
        console.print("[green]Output: in-place (*_functions beside each binary)[/green]")
    console.print(f"[green]Workers: {workers}[/green]")
    console.print(f"[green]Conda env: {conda_env or '<current-python>'}[/green]")
    console.print(f"[green]IDA path: {ida_path}[/green]")
    console.print(f"[green]Log file: {LOG_PATH}[/green]")
    if resume:
        console.print("[yellow]Resume mode: skip existing .i64 and *_functions outputs[/yellow]")
    if start_from_step2:
        console.print("[yellow]Start from Step 2: skip .i64 generation[/yellow]")
    console.print()

    logger.info("Input path: %s", db_path)
    logger.info("Output root: %s", output_root or "<in-place>")

    # Step 1: generate .i64 files
    if not start_from_step2:
        console.print("[bold blue]Step 1: Scanning for binaries to generate .i64...[/bold blue]")
        cleanup_ida_temp_files(db_path, logger)

        i64_tasks = []
        skipped_i64 = 0

        for root, dirs, files in os.walk(db_path):
            dirs[:] = [
                directory
                for directory in dirs
                if not directory.startswith(".")
                and directory != "__pycache__"
                and not directory.endswith("_functions")
            ]

            for file_name in files:
                if not is_candidate_binary(file_name):
                    continue

                binary_path = os.path.join(root, file_name)
                i64_path = binary_path + ".i64"

                if file_exists_and_not_empty(i64_path):
                    skipped_i64 += 1
                    continue

                i64_tasks.append(binary_path)

        i64_tasks.sort()

        if skipped_i64 > 0:
            console.print(f"[yellow]Skipping {skipped_i64} binaries with existing .i64 files[/yellow]")

        if i64_tasks:
            console.print(f"[bold blue]Found {len(i64_tasks)} binaries missing .i64[/bold blue]")
            console.print()
            logger.info("Step 1 pending .i64 generation: %s", len(i64_tasks))

            i64_success = 0
            i64_failed = 0
            i64_failed_files = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[{task.completed}/{task.total}]"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("[cyan]Generating .i64 files", total=len(i64_tasks))

                with ProcessPoolExecutor(max_workers=workers) as executor:
                    future_to_binary = {
                        executor.submit(generate_i64, binary_path, ida_path): binary_path
                        for binary_path in i64_tasks
                    }

                    for future in as_completed(future_to_binary):
                        binary_path = future_to_binary[future]
                        try:
                            success, _, command, returncode, stdout, stderr = future.result()
                            if success:
                                i64_success += 1
                            else:
                                i64_failed += 1
                                i64_failed_files.append(binary_path)
                                console.print(f"[red]✗ Failed .i64: {os.path.basename(binary_path)}[/red]")
                                logger.error(separator)
                                logger.error("Failed to generate .i64: %s", binary_path)
                                logger.error("Command: %s", " ".join(command))
                                logger.error("Return code: %s", returncode)
                                if stdout:
                                    logger.error("--- stdout ---\n%s", trim_output(stdout))
                                if stderr:
                                    logger.error("--- stderr ---\n%s", trim_output(stderr))
                        except Exception as exc:
                            i64_failed += 1
                            i64_failed_files.append(binary_path)
                            console.print(
                                f"[red]✗ Exception: {os.path.basename(binary_path)} - {str(exc)[:100]}[/red]"
                            )
                            logger.error("Exception during .i64 generation for %s: %s", binary_path, exc)
                            logger.error("Traceback:\n%s", traceback.format_exc())
                        finally:
                            progress.update(task_id, advance=1)

            console.print()
            console.print("[bold cyan]Step 1 Summary[/bold cyan]")
            console.print(f"[bold green]✓ .i64 generated: {i64_success}[/bold green]")
            if i64_failed > 0:
                console.print(f"[bold red]✗ .i64 failures: {i64_failed}[/bold red]")
            console.print()

            logger.info(
                "Step 1 summary: generated=%s, failed=%s, skipped_existing=%s",
                i64_success,
                i64_failed,
                skipped_i64,
            )

            if i64_failed > 0:
                logger.info("Step 1 failed files:")
                for failed_file in i64_failed_files:
                    logger.info("  - %s", failed_file)
        else:
            console.print("[yellow]No binaries need .i64 generation[/yellow]")
            console.print()
            logger.info("Step 1: no binaries required .i64 generation")
    else:
        console.print("[bold blue]Step 1: Skipped .i64 generation[/bold blue]")
        console.print()

    # Step 2: export .i64 files to ASM
    console.print("[bold blue]Step 2: Scanning for .i64 files to export ASM...[/bold blue]")
    export_tasks = []
    skipped_exports = 0

    for root, dirs, files in os.walk(db_path):
        dirs[:] = [
            directory
            for directory in dirs
            if not directory.startswith(".")
            and directory != "__pycache__"
            and not directory.endswith("_functions")
        ]

        for file_name in files:
            if not file_name.endswith(".i64"):
                continue

            i64_path = os.path.join(root, file_name)
            output_dir = build_output_dir(i64_path, db_path, output_root)

            if resume and directory_contains_asm(output_dir):
                skipped_exports += 1
                continue

            export_tasks.append((i64_path, output_dir))

    export_tasks.sort(key=lambda item: item[0])

    if resume and skipped_exports > 0:
        console.print(f"[yellow]Skipping {skipped_exports} binaries with existing ASM output[/yellow]")

    if not export_tasks:
        console.print("[yellow]No .i64 files to export, task completed[/yellow]")
        logger.info("Step 2: nothing to export. skipped_existing=%s", skipped_exports)
        return True

    console.print(f"[bold blue]Found {len(export_tasks)} .i64 files to export[/bold blue]")
    console.print()
    logger.info("Step 2 pending ASM exports: %s", len(export_tasks))

    success_count = 0
    failed_count = 0
    failed_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[{task.completed}/{task.total}]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Exporting functions to ASM", total=len(export_tasks))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(
                    export_single_i64,
                    i64_path,
                    output_dir,
                    conda_env,
                    save_database,
                ): (i64_path, output_dir)
                for i64_path, output_dir in export_tasks
            }

            for future in as_completed(future_to_task):
                i64_path, output_dir = future_to_task[future]
                try:
                    success, _, _, command, returncode, stdout, stderr = future.result()
                    if success:
                        success_count += 1
                        if returncode != 0:
                            logger.warning(separator)
                            logger.warning(
                                "ASM export for %s produced output but returned non-zero status %s",
                                i64_path,
                                returncode,
                            )
                            logger.warning("Output dir: %s", output_dir)
                            logger.warning("Command: %s", " ".join(command))
                            if stdout:
                                logger.warning("--- stdout ---\n%s", trim_output(stdout))
                            if stderr:
                                logger.warning("--- stderr ---\n%s", trim_output(stderr))
                    else:
                        failed_count += 1
                        failed_files.append(i64_path)
                        console.print(f"[red]✗ Failed ASM export: {os.path.basename(i64_path)}[/red]")
                        logger.error(separator)
                        logger.error("Failed ASM export for: %s", i64_path)
                        logger.error("Output dir: %s", output_dir)
                        logger.error("Command: %s", " ".join(command))
                        logger.error("Return code: %s", returncode)
                        if stdout:
                            logger.error("--- stdout ---\n%s", trim_output(stdout))
                        if stderr:
                            logger.error("--- stderr ---\n%s", trim_output(stderr))
                except Exception as exc:
                    failed_count += 1
                    failed_files.append(i64_path)
                    console.print(
                        f"[red]✗ Exception: {os.path.basename(i64_path)} - {str(exc)[:100]}[/red]"
                    )
                    logger.error("Exception during ASM export for %s: %s", i64_path, exc)
                    logger.error("Traceback:\n%s", traceback.format_exc())
                finally:
                    progress.update(task_id, advance=1)

    console.print()
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]bin2asm Summary[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[bold green]✓ ASM directories generated: {success_count}[/bold green]")
    if failed_count > 0:
        console.print(f"[bold red]✗ Export failures: {failed_count}[/bold red]")
        if len(failed_files) <= 10:
            for failed_file in failed_files:
                console.print(f"  - {os.path.basename(failed_file)}")
        else:
            for failed_file in failed_files[:5]:
                console.print(f"  - {os.path.basename(failed_file)}")
            console.print(f"  ... and {len(failed_files) - 5} more")
    else:
        console.print("[bold green]All .i64 files exported successfully![/bold green]")
    console.print()
    console.print(f"[cyan]Detailed logs: {LOG_PATH}[/cyan]")

    logger.info(separator)
    logger.info("bin2asm summary:")
    logger.info("  ASM export success: %s", success_count)
    logger.info("  ASM export failed:  %s", failed_count)
    if failed_files:
        logger.info("Step 2 failed files:")
        for failed_file in failed_files:
            logger.info("  - %s", failed_file)
    logger.info("bin2asm completed at %s", datetime.now())
    logger.info(separator)

    if success_count == 0 and failed_count > 0:
        raise typer.Exit(code=1)

    return True


if __name__ == "__main__":
    app()
