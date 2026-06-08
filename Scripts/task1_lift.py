#!/usr/bin/env python3
"""
Task 1: Lift binary files to LLVM IR using IDA Pro (Two-step process)

Step 1: Use IDA Pro to generate .i64 database files from binary files
Step 2: Use ida2llvm.py to convert .i64 files to LLVM IR

Both steps use parallel processing with separate progress bars for efficiency.
"""
import os
import typer
import multiprocessing
import logging
import subprocess
import traceback
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import console, ensure_directory, file_exists_and_not_empty

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = "/home/damaoooo/Downloads/regraphv2/Binaries"
IDA_PATH = "/home/damaoooo/ida-pro-9.3"  # Update this path to your IDA Pro installation
LOG_PATH = os.path.join(SCRIPT_DIR, "lift_task1_log.txt")
LIFT_BACKENDS = {"ida", "ghidra"}

app = typer.Typer()


def is_input_binary_file(file_name: str) -> bool:
    """Return True for dataset binaries and False for generated/source side files."""
    if file_name.startswith("."):
        return False
    ignored_suffixes = (
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
        ".re",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".txt",
        ".log",
        ".md",
        ".py",
        ".sh",
        ".json",
        ".jsonl",
        ".pkl",
    )
    return not file_name.endswith(ignored_suffixes)


def python_runner(conda_env: str) -> list[str]:
    """Build a Python command without hardcoding a conda environment."""
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"]
    return [sys.executable]


def generate_i64(binary_path: str, ida_path: str) -> tuple:
    """
    Generate the .i64 file for a given binary file using IDA Pro
    
    :param binary_path: Path to the binary file
    :return: (success, binary_path) tuple
    """
    command_line = [
        os.path.join(ida_path, "idat"),
        "-A", 
        "-B",
        binary_path
    ]
    success = False
    try:
        result = subprocess.run(command_line, capture_output=True, text=True)
        if result.returncode == 0:
            success = True
            # Check if .i64 file was actually created
            i64_path = binary_path + ".i64"
            if not os.path.exists(i64_path):
                success = False
        else:
            # Log error to file
            with open(LOG_PATH, "a") as log_file:
                log_file.write("=" * 60 + "\n")
                log_file.write(f"Error generating .i64 for: {binary_path}\n")
                log_file.write(f"Return code: {result.returncode}\n")
                if result.stderr:
                    log_file.write("--- stderr ---\n")
                    log_file.write(result.stderr[:500])
                    log_file.write("\n")
    except subprocess.TimeoutExpired:
        with open(LOG_PATH, "a") as log_file:
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timeout generating .i64 for: {binary_path}\n")
    except Exception as e:
        with open(LOG_PATH, "a") as log_file:
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Exception generating .i64 for: {binary_path}\n")
            log_file.write(f"Error: {str(e)}\n")
    
    return (success, binary_path)


def lift_single_file(
    input_binary: str,
    output_llvm: str,
    backend: str,
    conda_env: str,
    ghidra_home: str,
    analyze_headless: str,
    ghidra_target: str,
    ghidra_max_cpu: int,
    ghidra_decompile_timeout: int,
    ghidra_analysis_timeout: int,
    ghidra_no_analysis: bool,
    allow_partial: bool,
) -> tuple:
    """
    Lift a single binary file to LLVM IR.
    
    :param input_binary: Path to the binary file
    :param output_llvm: Path to the output LLVM IR file
    :return: (success, input_binary) tuple
    """
    if backend == "ida":
        lifter_path = os.path.join(SCRIPT_DIR, "ida2llvm.py")
        cmd = python_runner(conda_env) + [
            lifter_path,
            "-f",
            input_binary,
            "-o",
            output_llvm,
            "-v",
        ]
    elif backend == "ghidra":
        lifter_path = os.path.join(SCRIPT_DIR, "pcode2llvm.py")
        cmd = python_runner(conda_env) + [
            lifter_path,
            "-f",
            input_binary,
            "-o",
            output_llvm,
            "-v",
            "--target",
            ghidra_target,
            "--max-cpu",
            str(max(ghidra_max_cpu, 1)),
            "--decompile-timeout",
            str(max(ghidra_decompile_timeout, 1)),
            "--analysis-timeout",
            str(max(ghidra_analysis_timeout, 0)),
        ]
        if ghidra_home:
            cmd.extend(["--ghidra-home", ghidra_home])
        if analyze_headless:
            cmd.extend(["--analyze-headless", analyze_headless])
        if ghidra_no_analysis:
            cmd.append("--no-analysis")
        if allow_partial:
            cmd.append("--allow-partial")
    else:
        return (False, input_binary)

    result = subprocess.run(cmd, capture_output=True, text=True)
    success = result.returncode == 0
    if result.stdout or result.stderr:
        with open(LOG_PATH, "a") as log_file:
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Command output for: {input_binary}\n")
            log_file.write(f"Return code: {result.returncode}\n")
            if result.stdout:
                log_file.write("--- stdout ---\n")
                log_file.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    log_file.write("\n")
            if result.stderr:
                log_file.write("--- stderr ---\n")
                log_file.write(result.stderr)
                if not result.stderr.endswith("\n"):
                    log_file.write("\n")
    return (success, input_binary)

@app.command()
def main(
    input_path: str = typer.Option("", help="Input directory (defaults to DataProcess-1)"),
    output: str = typer.Option(..., help="Output directory"),
    workers: int = typer.Option(multiprocessing.cpu_count(), help="Number of worker processes"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
    backend: str = typer.Option(
        "ida",
        "--backend",
        help="Lifter backend: ida uses ida2llvm.py on .i64 files; ghidra uses pcode2llvm.py directly on binaries",
    ),
    conda_env: str = typer.Option(
        os.environ.get("REGRAPH_CONDA_ENV", ""),
        "--conda-env",
        help="Conda environment for lifter subprocesses; empty uses the current Python interpreter",
    ),
    ida_path: str = typer.Option(IDA_PATH, "--ida-path", help="IDA Pro installation path for the ida backend"),
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
    ghidra_max_cpu: int = typer.Option(
        1,
        "--ghidra-max-cpu",
        help="Ghidra CPUs per worker. Keep this low when task1 workers already provide parallelism",
    ),
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
        help="Skip Ghidra auto-analysis. Only use this for debugging already-analyzed projects",
    ),
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial",
        help="Allow pcode2llvm.py to succeed when some functions are skipped. Off by default for reproducibility",
    ),
    start_from_step2: bool = typer.Option(
        False,
        help="Start directly from Step 2 (scan .i64 and lift), skip Step 1 .i64 generation",
    ),
):
    """
    Lift binary files to LLVM IR using IDA or Ghidra.
    
    IDA backend:
    Step 1: Generate .i64 files from binary files using IDA Pro
    Step 2: Convert .i64 files to LLVM IR format using ida2llvm.py

    Ghidra backend:
    Step 1: Skipped
    Step 2: Convert binaries directly to LLVM IR using pcode2llvm.py
    
    This command scans the input directory for binaries and mirrors the input
    directory structure under the output directory.
    
    Features:
    - Two-stage processing with separate progress bars
    - Parallel processing (configurable number of workers) for both stages
    - Resume capability (skip already processed files in both stages)
    - Automatic directory creation for output
    - Detailed logging of all failures to lift_task1_log.txt
    - Automatic cleanup of failed IDA temporary files in IDA mode
    """
    backend = backend.lower()
    if backend not in LIFT_BACKENDS:
        console.print(f"[red]Error: backend must be one of {sorted(LIFT_BACKENDS)}, got {backend!r}.[/red]")
        raise typer.Exit(code=1)
    ghidra_target = ghidra_target.lower()
    if ghidra_target not in ("host", "ghidra"):
        console.print("[red]Error: --ghidra-target must be one of: host, ghidra.[/red]")
        raise typer.Exit(code=1)
    if backend == "ida" and not os.path.exists(os.path.join(ida_path, "idat")) and not start_from_step2:
        console.print(f"[red]Error: IDA idat not found under {ida_path}. Use --backend ghidra for artifact runs.[/red]")
        raise typer.Exit(code=1)
    
    # Setup logging
    logging.basicConfig(
        filename=LOG_PATH,
        filemode='a',  # Append mode
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Log session start
    separator = "="*60
    logger.info(separator)
    logger.info(f"Task 1 started at {datetime.now()}")
    logger.info(
        "Workers: %s, Resume: %s, StartFromStep2: %s, Backend: %s, CondaEnv: %s",
        workers,
        resume,
        start_from_step2,
        backend,
        conda_env or "<current-python>",
    )
    logger.info(separator)
    
    # Determine input path
    if input_path:
        db = os.path.basename(input_path.rstrip("/"))
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
    
    console.print(f"[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]Task 1: Binary to LLVM IR Lifting[/bold cyan]")
    console.print(f"[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[green]Input:  {db_path}[/green]")
    console.print(f"[green]Output: {output_path}[/green]")
    console.print(f"[green]Backend: {backend}[/green]")
    console.print(f"[green]Workers: {workers}[/green]")
    console.print(f"[green]Lifter Python: {conda_env or 'current interpreter'}[/green]")
    console.print(f"[green]Log file: {LOG_PATH}[/green]")
    if resume:
        console.print(f"[yellow]Resume mode: skipping existing lifted files[/yellow]")
    if start_from_step2:
        console.print(f"[yellow]Start from Step 2: skipping .i64 generation[/yellow]")
    console.print()
    
    logger.info(f"Input path: {db_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Backend: {backend}")

    # Step 1: Generate .i64 files from binaries (optional)
    if backend == "ghidra":
        console.print("[bold blue]Step 1: Skipped .i64 generation (Ghidra backend lifts binaries directly)[/bold blue]")
        console.print()
    elif not start_from_step2:
        console.print("[bold blue]Step 1: Scanning for binary files to generate .i64...[/bold blue]")
        
        # Clean any failed IDA files first
        files_to_remove = []
        for root, dirs, files in os.walk(db_path):
            for file in files:
                if file.endswith((".idb", ".id0", ".id1", ".id2", ".til", ".nam", ".asm")):
                    files_to_remove.append(os.path.join(root, file))
        for f in files_to_remove:
            try:
                os.remove(f)
                logger.info(f"Removed failed IDA file: {f}")
            except Exception as e:
                logger.warning(f"Could not remove {f}: {str(e)}")
        
        # Scan for binary files (non-IDA files) that need .i64 generation
        i64_tasks = []
        i64_skipped_count = 0
        
        for root, dirs, files in os.walk(db_path):
            for file in files:
                if not is_input_binary_file(file):
                    continue
                    
                file_path = os.path.join(root, file)
                i64_path = file_path + ".i64"
                
                # Always skip if .i64 already exists
                if file_exists_and_not_empty(i64_path):
                    i64_skipped_count += 1
                    continue
                    
                i64_tasks.append(file_path)
        
        if i64_skipped_count > 0:
            console.print(f"[yellow]Skipping {i64_skipped_count} binaries with existing .i64 files[/yellow]")
        
        total_i64_tasks = len(i64_tasks)
        if total_i64_tasks > 0:
            console.print(f"[bold blue]Found {total_i64_tasks} binary files to process[/bold blue]")
            console.print()
            
            logger.info(f"Found {total_i64_tasks} binary files for .i64 generation")
            if i64_skipped_count > 0:
                logger.info(f"Skipping {i64_skipped_count} binaries with existing .i64 files")
            
            # Generate .i64 files in parallel
            i64_success_count = 0
            i64_failed_count = 0
            i64_failed_files = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[{task.completed}/{task.total}]"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                i64_task = progress.add_task(
                    "[cyan]Generating .i64 files using IDA Pro", 
                    total=total_i64_tasks
                )
                
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    # Submit all i64 generation tasks
                    future_to_binary = {
                        executor.submit(generate_i64, binary_path, ida_path): binary_path
                        for binary_path in i64_tasks
                    }
                
                    # Process completed tasks as they finish
                    for future in as_completed(future_to_binary):
                        binary_path = future_to_binary[future]
                        try:
                            success, _ = future.result()
                            if success:
                                i64_success_count += 1
                                logger.debug(f"Successfully generated .i64: {binary_path}")
                            else:
                                i64_failed_count += 1
                                i64_failed_files.append(binary_path)
                                console.print(f"[red]✗ Failed: {os.path.basename(binary_path)}[/red]")
                                logger.error(f"Failed to generate .i64: {binary_path}")
                        except Exception as exc:
                            i64_failed_count += 1
                            i64_failed_files.append(binary_path)
                            error_msg = f"Exception for {binary_path}: {str(exc)}"
                            console.print(f"[red]✗ Exception: {os.path.basename(binary_path)} - {str(exc)[:100]}[/red]")
                            logger.error(error_msg)
                            logger.error(f"Traceback:\n{traceback.format_exc()}")
                        finally:
                            progress.update(i64_task, advance=1)
            
            console.print()
            console.print(f"[bold cyan]Step 1 Summary:[/bold cyan]")
            console.print(f"[bold green]✓ .i64 files generated: {i64_success_count}[/bold green]")
            if i64_failed_count > 0:
                console.print(f"[bold red]✗ Failed: {i64_failed_count}[/bold red]")
            console.print()
            
            logger.info(f"Step 1 Summary: Success={i64_success_count}, Failed={i64_failed_count}")
        else:
            console.print(f"[yellow]No binary files need .i64 generation (skipped: {i64_skipped_count})[/yellow]")
            console.print()
    else:
        console.print("[bold blue]Step 1: Skipped .i64 generation[/bold blue]")
        console.print()

    if backend == "ida":
        console.print("[bold blue]Step 2: Scanning for .i64 files to lift to LLVM IR...[/bold blue]")
    else:
        console.print("[bold blue]Step 2: Scanning for binaries to lift to LLVM IR with Ghidra...[/bold blue]")
    lift_tasks = []
    skipped_count = 0
    
    for root, dirs, files in os.walk(db_path):
        for file in files:
            if backend == "ida":
                if not file.endswith(".i64"):
                    continue
                output_name = file.replace(".i64", "") + ".ll"
            else:
                if not is_input_binary_file(file):
                    continue
                output_name = file + ".ll"
                
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, db_path)
            output_dir = os.path.join(output_path, relative_path)
            ensure_directory(output_dir)
            output_file_path = os.path.join(output_dir, output_name)
            
            # Check if file already exists and we're resuming
            if resume and file_exists_and_not_empty(output_file_path):
                skipped_count += 1
                continue
                
            lift_tasks.append((file_path, output_file_path))

    if resume and skipped_count > 0:
        console.print(f"[yellow]Skipping {skipped_count} already lifted files[/yellow]")

    total_tasks = len(lift_tasks)
    if total_tasks == 0:
        console.print("[yellow]No files to lift, task completed[/yellow]")
        return

    console.print(f"[bold blue]Found {total_tasks} files to lift[/bold blue]")
    console.print()
    
    logger.info(f"Found {total_tasks} files to lift")
    if resume and skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already lifted files")

    # Execute lifting in parallel
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
        console=console
    ) as progress:
        lift_task = progress.add_task(
            "[cyan]Lifting binary files to LLVM IR", 
            total=total_tasks
        )
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    lift_single_file,
                    input_bin,
                    output_llvm,
                    backend,
                    conda_env,
                    ghidra_home,
                    analyze_headless,
                    ghidra_target,
                    ghidra_max_cpu,
                    ghidra_decompile_timeout,
                    ghidra_analysis_timeout,
                    ghidra_no_analysis,
                    allow_partial,
                ): (input_bin, output_llvm)
                for input_bin, output_llvm in lift_tasks
            }
        
            # Process completed tasks as they finish
            for future in as_completed(future_to_task):
                input_bin, output_llvm = future_to_task[future]
                try:
                    success, _ = future.result()
                    if success:
                        success_count += 1
                        logger.debug(f"Successfully lifted: {input_bin}")
                    else:
                        failed_count += 1
                        failed_files.append(input_bin)
                        console.print(f"[red]✗ Failed: {os.path.basename(input_bin)}[/red]")
                        logger.error(f"Failed to lift: {input_bin}")
                except Exception as exc:
                    failed_count += 1
                    failed_files.append(input_bin)
                    error_msg = f"Exception for {input_bin}: {str(exc)}"
                    console.print(f"[red]✗ Exception: {os.path.basename(input_bin)} - {str(exc)[:100]}[/red]")
                    # Log detailed traceback to file
                    logger.error(error_msg)
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                finally:
                    progress.update(lift_task, advance=1)

    console.print()
    console.print(f"[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]Task 1 Final Summary[/bold cyan]")
    console.print(f"[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[bold green]✓ LLVM IR files generated: {success_count}[/bold green]")
    if failed_count > 0:
        console.print(f"[bold red]✗ Lift failures: {failed_count}[/bold red]")
        if len(failed_files) <= 10:
            for f in failed_files:
                console.print(f"  - {os.path.basename(f)}")
        else:
            for f in failed_files[:5]:
                console.print(f"  - {os.path.basename(f)}")
            console.print(f"  ... and {len(failed_files) - 5} more")
    else:
        console.print(f"[bold green]All files lifted successfully![/bold green]")
    console.print()
    
    # Log summary
    logger.info(separator)
    logger.info(f"Task 1 Final Summary:")
    logger.info(f"  LLVM IR Success: {success_count}")
    logger.info(f"  Lift Failed:  {failed_count}")
    if failed_count > 0:
        logger.info(f"Failed files:")
        for f in failed_files:
            logger.info(f"    - {f}")
    logger.info(f"Task 1 completed at {datetime.now()}")
    logger.info(separator)
    
    console.print(f"[cyan]Detailed logs: {LOG_PATH}[/cyan]")
    
    if failed_count > 0:
        console.print("[yellow]To retry failed files, run with the same output directory[/yellow]")
        raise typer.Exit(code=1)
    return True

if __name__ == "__main__":
    app()
