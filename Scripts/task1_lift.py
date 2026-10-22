#!/usr/bin/env python3
"""
Task 1: Lift binary files to LLVM IR using IDA Pro (Two-step process)

Step 1: Use IDA Pro to generate .i64 database files from binary files
Step 2: Use ida2llvm.py to convert .i64 files to LLVM IR

Both steps use parallel processing with separate progress bars for efficiency.
"""
import os
import hashlib
import time
import json
from collections import Counter
import typer
import logging
import sys
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import console, ensure_directory, file_exists_and_not_empty
import artifact_state
from job_runner import default_workers, log_tail, run_bounded, run_logged

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = "/home/damaoooo/Downloads/regraphv2/Binaries"


def default_ida_path() -> str:
    """IDADIR > ~/.idapro/ida-config.json（py-activate-idalib.py 写入）> ~/ida-pro-9.4"""
    env_path = os.environ.get("IDADIR")
    if env_path:
        return env_path
    try:
        with open(os.path.expanduser("~/.idapro/ida-config.json")) as handle:
            config_path = json.load(handle)["Paths"]["ida-install-dir"]
        if config_path:
            return config_path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return os.path.expanduser("~/ida-pro-9.4")


IDA_PATH = default_ida_path()
LOG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "lift_task1_log.txt")
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


# IDA 运行中解包出的数据库分片；存在说明上一次 IDA 未正常结束
IDA_UNPACKED_SUFFIXES = (".id0", ".id1", ".id2", ".nam", ".til")


# 与 ida2llvm.MEMORY_CONTRACTS 一致
MEMORY_CONTRACTS = {"source": "source-layout", "host": "host-compat-unconverted"}


def ida2llvm_lifter_id() -> str:
    """与 ida2llvm.LIFTER_ID 相同：lifter 源码 sha256 的前 16 位。"""
    return artifact_state.lifter_id(os.path.join(SCRIPT_DIR, "ida2llvm.py"))


def lift_output_reusable(input_path: str, output_path: str, lifter_id: str, target_mode: str = "source",
                         ida_id: str | None = None) -> bool:
    """
    --resume 只复用由当前 lifter、当前 IDA 安装、同一 target 模式、针对当前输入成功完成，且磁盘上的
    .ll 正是 sidecar 记录的那个文件（sha256/size 一致）的输出。
    """
    state = artifact_state.read_json(output_path + artifact_state.LIFT_SUFFIX)
    ok, _ = artifact_state.check_output(state, output_path)
    if not ok:
        return False
    try:
        st = os.stat(input_path)
    except OSError:
        return False
    recorded = state.get("input", {})
    return (
        state.get("lifter_id") == lifter_id
        and state.get("target_mode") == target_mode
        and state.get("memory_contract") == MEMORY_CONTRACTS.get(target_mode)
        and (ida_id is None or state.get("ida_install_id") == ida_id)
        and recorded.get("size") == st.st_size
        and recorded.get("mtime_ns") == st.st_mtime_ns
    )


def job_log_path(log_dir: str, kind: str, path: str) -> str:
    digest = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]
    return os.path.join(log_dir, f"{kind}_{os.path.basename(path)}_{digest}.log")


def python_runner(conda_env: str) -> list[str]:
    """Build a Python command without hardcoding a conda environment."""
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"]
    return [sys.executable]


def generate_i64(binary_path: str, ida_path: str, timeout_s: float = 0, log_dir: str = LOG_DIR) -> tuple:
    """
    Generate the .i64 file for a given binary file using IDA Pro.
    Output goes to a per-job log; on timeout the whole IDA process group is killed.

    :return: (success, binary_path, message) tuple
    """
    command_line = [os.path.join(ida_path, "idat"), "-A", "-B", binary_path]
    stale = [binary_path + suffix for suffix in IDA_UNPACKED_SUFFIXES if os.path.exists(binary_path + suffix)]
    if stale:
        # 不自动删除：这些文件可能属于正在运行的另一个 IDA，或需要人工检查
        return (False, binary_path, f"stale unpacked IDA database files {stale}")
    # -B 会在 binary 旁生成 .asm；只删除本次运行新建的那个
    asm_path = binary_path + ".asm"
    asm_existed = os.path.exists(asm_path)
    result = run_logged(command_line, job_log_path(log_dir, "i64", binary_path), timeout_s)
    if not asm_existed and os.path.exists(asm_path):
        os.remove(asm_path)
    if result.timed_out:
        return (False, binary_path, f"timed out after {timeout_s}s (log {result.log_path})")
    if not result.ok:
        return (False, binary_path, f"rc={result.returncode} {result.error} (log {result.log_path})")
    if not os.path.exists(binary_path + ".i64"):
        return (False, binary_path, f"idat succeeded but no .i64 (log {result.log_path})")
    return (True, binary_path, "")


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
    target_mode: str = "source",
    timeout_s: float = 0,
    log_dir: str = LOG_DIR,
) -> tuple:
    """
    Lift a single binary file to LLVM IR in a separate process group with a timeout.

    :return: (success, input_binary, message) tuple
    """
    if backend == "ida":
        lifter_path = os.path.join(SCRIPT_DIR, "ida2llvm.py")
        cmd = python_runner(conda_env) + [
            lifter_path, "-f", input_binary, "-o", output_llvm, "--target", target_mode, "-v",
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
        return (False, input_binary, f"unknown backend {backend}")

    started = time.time()
    result = run_logged(cmd, job_log_path(log_dir, "lift", input_binary), timeout_s)
    if backend == "ida" and not result.ok:
        # 子进程被杀或崩溃时 lifter 来不及写 sidecar：代写失败状态，下游不会把旧 .ll 当成本次结果
        status = "timeout" if result.timed_out else "crashed"
        artifact_state.write_failed_lift_state(output_llvm, input_binary, status,
                                               log_tail(result.log_path, 2000), started)
    if result.timed_out:
        return (False, input_binary, f"timed out after {timeout_s}s (log {result.log_path})")
    if not result.ok:
        tail = log_tail(result.log_path, 600).strip().splitlines()
        return (False, input_binary, f"rc={result.returncode} {result.error} {tail[-1] if tail else ''} (log {result.log_path})")
    return (True, input_binary, "")

@app.command()
def main(
    input_path: str = typer.Option("", help="Input directory (defaults to DataProcess-1)"),
    output: str = typer.Option(..., help="Output directory"),
    workers: int = typer.Option(
        0, help="Concurrent jobs; 0 = min(CPUs/2, MemAvailable / --mem-per-job-mb). Measure before raising it"),
    resume: bool = typer.Option(False, help="Resume: skip outputs whose .lift.json is ok and matches the current lifter, IDA install, target mode, input and output hash (Ghidra: existing non-empty files)"),
    target_mode: str = typer.Option(
        "source", "--target-mode",
        help="ida2llvm output target: source (source triple/data layout) or host (legacy compatibility, no memory-semantics conversion)"),
    lift_timeout: int = typer.Option(6 * 3600, "--lift-timeout", help="Per-binary lift timeout in seconds (0 = none)"),
    i64_timeout: int = typer.Option(3600, "--i64-timeout", help="Per-binary IDA analysis timeout in seconds (0 = none)"),
    mem_per_job_mb: int = typer.Option(
        4096, "--mem-per-job-mb", help="Estimated peak memory per job; new jobs wait while MemAvailable is below it"),
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
    if backend == "ida":
        # ida2llvm.py 子进程通过 idalib 加载 IDA，统一指向同一个安装目录
        os.environ["IDADIR"] = ida_path
    
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
    if target_mode not in MEMORY_CONTRACTS:
        console.print(f"[red]Error: --target-mode must be one of {sorted(MEMORY_CONTRACTS)}[/red]")
        raise typer.Exit(code=1)
    if workers <= 0:
        workers = default_workers(mem_per_job_mb)
    job_log_dir = os.path.join(LOG_DIR, "task1_jobs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    ida_id = artifact_state.ida_install_id(ida_path) if backend == "ida" else None
    
    console.print(f"[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]Task 1: Binary to LLVM IR Lifting[/bold cyan]")
    console.print(f"[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[green]Input:  {db_path}[/green]")
    console.print(f"[green]Output: {output_path}[/green]")
    console.print(f"[green]Backend: {backend}[/green]")
    console.print(f"[green]Workers: {workers} (mem/job {mem_per_job_mb} MB, lift timeout {lift_timeout}s)[/green]")
    if backend == "ida":
        console.print(f"[green]Target mode: {target_mode} ({MEMORY_CONTRACTS[target_mode]})[/green]")
    console.print(f"[green]Per-job logs: {job_log_dir}[/green]")
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
        
        # 不再递归删除 .idb/.id0/.asm 等文件：可能属于其他运行或其他数据（如 ASM 基线）。
        # 残留的解包分片由 generate_i64 按 binary 检测并跳过、记录到日志。

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
                
                def i64_done(binary_path, outcome):
                    if isinstance(outcome, Exception):
                        outcome = (False, binary_path, f"exception {outcome}")
                    success, _, message = outcome
                    if success:
                        i64_state["ok"] += 1
                    else:
                        i64_state["failed"] += 1
                        i64_failed_files.append(binary_path)
                        console.print(f"[red]✗ Failed: {os.path.basename(binary_path)} {message[:160]}[/red]")
                        logger.error(f"Failed to generate .i64: {binary_path}: {message}")
                    progress.update(i64_task, advance=1)

                i64_state = {"ok": 0, "failed": 0}
                run_bounded(
                    i64_tasks,
                    lambda binary_path: generate_i64(binary_path, ida_path, i64_timeout, job_log_dir),
                    workers, mem_per_job_mb, i64_done,
                )
                i64_success_count, i64_failed_count = i64_state["ok"], i64_state["failed"]
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
    lifter_id = ida2llvm_lifter_id() if backend == "ida" else None
    stale_reasons = Counter()
    
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
            if resume:
                if backend == "ida":
                    # 需要 sidecar 证明输出完整、来自当前 lifter 和当前输入
                    reusable = lift_output_reusable(file_path, output_file_path, lifter_id, target_mode, ida_id)
                    if not reusable and os.path.exists(output_file_path):
                        stale_reasons["existing output not reusable"] += 1
                else:
                    reusable = file_exists_and_not_empty(output_file_path)
                if reusable:
                    skipped_count += 1
                    continue
                
            lift_tasks.append((file_path, output_file_path))

    if resume and skipped_count > 0:
        console.print(f"[yellow]Skipping {skipped_count} already lifted files[/yellow]")
    for reason, count in stale_reasons.items():
        console.print(f"[yellow]Re-lifting {count} files: {reason} (state missing/failed, hash or identity mismatch)[/yellow]")

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
        
        def lift_done(task, outcome):
            input_bin, _ = task
            if isinstance(outcome, Exception):
                outcome = (False, input_bin, f"exception {outcome}")
            success, _, message = outcome
            if success:
                lift_state["ok"] += 1
            else:
                lift_state["failed"] += 1
                failed_files.append(input_bin)
                console.print(f"[red]✗ Failed: {os.path.basename(input_bin)} {message[:160]}[/red]")
                logger.error(f"Failed to lift: {input_bin}: {message}")
            progress.update(lift_task, advance=1)

        lift_state = {"ok": 0, "failed": 0}
        run_bounded(
            lift_tasks,
            lambda task: lift_single_file(
                task[0], task[1], backend, conda_env, ghidra_home, analyze_headless, ghidra_target,
                ghidra_max_cpu, ghidra_decompile_timeout, ghidra_analysis_timeout, ghidra_no_analysis,
                allow_partial, target_mode, lift_timeout, job_log_dir,
            ),
            workers, mem_per_job_mb, lift_done,
        )
        success_count, failed_count = lift_state["ok"], lift_state["failed"]

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
