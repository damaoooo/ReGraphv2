import os
import shutil
import subprocess
import logging
import tempfile
import hashlib
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, List
from Utils.utils import *
import pickle
import sqlite3
import multiprocessing
from DataProcess.dataset_utils import FileFilter, find_llvm_files
import typer

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

# Global logger for multiprocessing
logger = logging.getLogger(__name__)

# Graph artifacts live next to the extracted functions, never inside them: Task 3 publishes
# `<binary>_functions/` as an exact file set and replaces it when that set changes.
FUNCTIONS_SUFFIX = '_functions'
ARTIFACTS_SUFFIX = '_graphs'
ARTIFACT_ENDINGS = ('_purified.ll', '_instrumented.ll', '.dot')
CACHE_FORMAT = 3
# Workers start with spawn: the parent already runs rich's progress-refresh thread when the pool starts, and
# forking a multi-threaded process can leave a lock held forever in the child.
MP_CONTEXT = multiprocessing.get_context('spawn')


def artifact_dir_for(llvm_ir_path: str) -> str:
    """`.../<binary>_functions/<hash>.ll` -> `.../<binary>_graphs`; a file elsewhere keeps its own directory."""
    function_dir = os.path.dirname(os.path.abspath(llvm_ir_path))
    if not function_dir.endswith(FUNCTIONS_SUFFIX):
        return function_dir
    return function_dir[:-len(FUNCTIONS_SUFFIX)] + ARTIFACTS_SUFFIX


def _file_sha256(path: str) -> str:
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@lru_cache(maxsize=None)
def _tool_signature() -> str:
    """Passes and opt that produce the artifacts; a rebuild or upgrade invalidates earlier results."""
    opt_version = subprocess.run(["opt", "--version"], text=True, capture_output=True, check=True).stdout
    parts = [opt_version] + [_file_sha256(path) for path in (DEFAULT_PURIFY_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_CFG_SO_PATH)]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


def result_signature(llvm_ir_path: str) -> str:
    """Bind a result row to the input function's content (paths are reused when functions are re-extracted)."""
    return hashlib.sha256((_tool_signature() + _file_sha256(llvm_ir_path)).encode()).hexdigest()


class PassExecutionError(RuntimeError):
    """A pass failed on input opt accepts: an output, plugin or environment failure, never a rejected function."""


def _pass_failed(step: str, input_path: str, detail: str, returncode: Optional[int] = None) -> None:
    """Classify a failed pass run: a function opt itself rejects is skipped; anything else stops the run.

    Unwritable or full output storage and broken plugins also make opt fail, so only input that the verifier
    runs to completion on and rejects counts as a rejected function; the check runs on the failure path only.
    A pass or verifier killed by a signal (e.g. by the OOM killer) has shown nothing about the input.
    """
    if returncode is not None and returncode < 0:
        raise PassExecutionError(f"{step} on {input_path} was killed by signal {-returncode}: {detail}")
    verifier = subprocess.run(["opt", "-disable-output", "-passes=verify", os.path.abspath(input_path)],
                              text=True, capture_output=True)
    if verifier.returncode < 0:
        raise PassExecutionError(f"verifying {input_path} after a {step} failure was killed by signal "
                                 f"{-verifier.returncode}: {detail}")
    if verifier.returncode == 0:
        raise PassExecutionError(f"{step} failed on valid IR {input_path}: {detail}")
    logger.error(f"{step} rejected {input_path}: {detail}; verifier: {verifier.stderr.strip()[:500]}")


def _run_opt(cmd: List[str], cwd: str, step: str, input_path: str) -> Optional[subprocess.CompletedProcess]:
    p = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd)
    if p.returncode != 0:
        _pass_failed(step, input_path, f"return code {p.returncode}, stderr: {p.stderr.strip()[:2000]}", p.returncode)
        return None
    return p


def _opt_initial_purify(llvm_ir_path: str, artifact_dir: str) -> Optional[str]:
    """Strip all metadata from LLVM IR using custom pass"""
    purified_file = os.path.join(artifact_dir, os.path.splitext(os.path.basename(llvm_ir_path))[0] + '_purified.ll')
    cmd = [
        "opt",
        "-S",
        "-load-pass-plugin=" + DEFAULT_PURIFY_SO_PATH,
        "--passes=strip-all-metadata",
        "-non-global-value-max-name-size=16384",
        # relative to the artifact directory, so the ModuleID opt records does not depend on where the data lives
        os.path.relpath(os.path.abspath(llvm_ir_path), artifact_dir),
        "-o", os.path.basename(purified_file)
    ]
    if _run_opt(cmd, artifact_dir, "metadata stripping", llvm_ir_path) is None:
        return None
    if not os.path.exists(purified_file):
        _pass_failed("metadata stripping", llvm_ir_path, f"{purified_file} was not created")
        return None
    return purified_file


def _opt_generate_ddg(purified_llvm_ir_path: str) -> Optional[Tuple[str, str]]:
    """Generate DDG using opt tool on purified IR"""
    instrumented_file = os.path.splitext(purified_llvm_ir_path)[0] + '_instrumented.ll'
    new_path = os.path.dirname(purified_llvm_ir_path)
    cmd = [
        "opt",
        "-load-pass-plugin=" + DEFAULT_DDG_SO_PATH,
        "-passes=dot-id-graph",
        "-non-global-value-max-name-size=16384",
        os.path.basename(purified_llvm_ir_path),
        "-S",
        "-o", os.path.basename(instrumented_file)
    ]
    p = _run_opt(cmd, new_path, "DDG generation", purified_llvm_ir_path)
    if p is None:
        return None
    dot_file = next((line.split("'")[1] for line in p.stderr.splitlines() if "Writing ID-tagged graph to" in line), None)
    dot_file = dot_file and os.path.normpath(os.path.join(new_path, dot_file))
    if not (dot_file and os.path.exists(dot_file) and os.path.exists(instrumented_file)):
        _pass_failed("DDG generation", purified_llvm_ir_path, f"outputs missing; stderr: {p.stderr.strip()[:2000]}")
        return None
    return instrumented_file, dot_file


def _opt_generate_cfg(purified_llvm_ir_path: str) -> Optional[str]:
    """Generate CFG using opt tool on purified IR"""
    new_path = os.path.dirname(purified_llvm_ir_path)
    cmd = [
        "opt",
        "-load-pass-plugin=" + DEFAULT_CFG_SO_PATH,
        "-passes=dot-my-cfg",
        "-non-global-value-max-name-size=16384",
        os.path.basename(purified_llvm_ir_path),
        "-o", "/dev/null"
    ]
    p = _run_opt(cmd, new_path, "CFG generation", purified_llvm_ir_path)
    if p is None:
        return None
    dot_file = next((line.split("'")[1] for line in p.stderr.splitlines() if "Write CFG to " in line), None)
    dot_file = dot_file and os.path.normpath(os.path.join(new_path, dot_file))  # the pass reports './cfg_<hash>.dot'
    if not (dot_file and os.path.exists(dot_file)):
        _pass_failed("CFG generation", purified_llvm_ir_path, f"dot file missing; stderr: {p.stderr.strip()[:2000]}")
        return None
    return dot_file


# A small function every working toolchain handles: two blocks for the CFG, a data dependence for the DDG
_PROBE_IR = ('define i32 @probe(i32 %x) {\nentry:\n  %y = add i32 %x, 1\n  %c = icmp sgt i32 %y, 0\n'
             '  br i1 %c, label %pos, label %neg\npos:\n  ret i32 %y\nneg:\n  ret i32 0\n}\n')


class _ErrorCollector(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def check_toolchain():
    """Raise before any result changes if opt or a pass plugin cannot process a known-good function.

    A broken toolchain makes every function fail alike; without this check that would look like a dataset
    of invalid functions, and their results would be deleted.
    """
    missing = [path for path in (DEFAULT_PURIFY_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_CFG_SO_PATH)
               if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(f"LLVMTools pass plugins are missing: {missing}. Build them with "
                           "`cmake -S LLVMTools -B LLVMTools/build -DCMAKE_BUILD_TYPE=Release` and "
                           "`cmake --build LLVMTools/build -j 4`")
    collector = _ErrorCollector()
    logger.addHandler(collector)
    try:
        with tempfile.TemporaryDirectory(prefix='graph-toolchain-check-') as workspace:
            probe = Path(workspace, f'probe{FUNCTIONS_SUFFIX}', 'probe.ll')
            probe.parent.mkdir()
            probe.write_text(_PROBE_IR)
            try:
                result = process_one_file(str(probe))
            except Exception as error:  # e.g. PassExecutionError, or opt missing from PATH
                raise RuntimeError(f"opt ({shutil.which('opt')}) cannot run the graph passes on a known-good function: {error}") from error
    finally:
        logger.removeHandler(collector)
    if result is None:
        detail = collector.messages[0][:2000] if collector.messages else 'no diagnostics'
        raise RuntimeError(f"opt ({shutil.which('opt')}) cannot run the graph passes on a known-good function: {detail}")


def process_one_file(llvm_ir_path: str) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Process a single LLVM IR file to generate purified IR, DDG, and CFG.

    Args:
        llvm_ir_path: Path to the LLVM IR file.

    Returns:
        Tuple (input, purified IR, instrumented IR, CFG dot, DDG dot) or None if any step fails;
        a failed function leaves no artifacts behind.
    """
    artifact_dir = artifact_dir_for(llvm_ir_path)
    os.makedirs(artifact_dir, exist_ok=True)
    if not os.access(artifact_dir, os.W_OK | os.X_OK):
        # An unwritable output directory is an environment problem, not an invalid function: fail the run
        raise PermissionError(f"cannot write graph artifacts to {artifact_dir}")
    produced = []
    result = None
    try:
        purified_llvm_ir_path = _opt_initial_purify(llvm_ir_path, artifact_dir)
        if purified_llvm_ir_path is None:
            return None
        produced.append(purified_llvm_ir_path)

        ddg_result = _opt_generate_ddg(purified_llvm_ir_path)
        if ddg_result is None:
            return None

        instrumented_file, ddg_dot_file = ddg_result
        produced += [instrumented_file, ddg_dot_file]

        cfg_dot_file = _opt_generate_cfg(purified_llvm_ir_path)
        if cfg_dot_file is None:
            return None

        result = (llvm_ir_path, purified_llvm_ir_path, instrumented_file, cfg_dot_file, ddg_dot_file)
        return result
    finally:
        if result is None:
            for path in produced:
                if os.path.exists(path):
                    os.remove(path)


def process_chunk(file_paths: List[str]) -> Tuple[List[tuple], List[str]]:
    """Result rows of the functions that produced graphs, and the paths of those that failed."""
    results = []
    failed = []
    for file_path in file_paths:
        # Hash the input before processing so the row describes the content the artifacts came from
        signature = result_signature(file_path)
        result = process_one_file(file_path)
        if result is not None:
            results.append(result + (signature,))
        else:
            failed.append(file_path)
    return results, failed


def init_database(output_db_path: str):
    """Initialize the SQLite database for storing results."""
    with sqlite3.connect(output_db_path,) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_path TEXT UNIQUE,
            purify_path TEXT,
            instrumented_path TEXT,
            cfg_dot TEXT,
            ddg_dot TEXT,
            signature TEXT
            )
        """)
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(results)")}
        if 'signature' not in columns:
            # Rows from before signatures were recorded cannot be verified and will be regenerated
            cursor.execute("ALTER TABLE results ADD COLUMN signature TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_path ON results(input_path)")
        conn.commit()


def write_results(connection: sqlite3.Connection, results: List[tuple], failed: List[str]):
    connection.executemany("""
        INSERT OR REPLACE INTO results (input_path, purify_path, instrumented_path, cfg_dot, ddg_dot, signature)
        VALUES (?, ?, ?, ?, ?, ?)
    """, results)
    # A function that fails now must not keep a result row from an earlier input or toolchain
    connection.executemany("DELETE FROM results WHERE input_path = ?", [(path,) for path in failed])
    connection.commit()


def _function_dirs_digest(input_dir: str) -> str:
    """Changes whenever any file of any function directory is added, removed or rewritten.

    Task 3 republishes a function directory as new files, so name, size and modification time of every
    file cover a re-extraction even when function_map.csv and the file count stay the same.
    """
    digest = hashlib.sha256()
    for function_dir in sorted(find_llvm_files(input_dir, pattern=f"**/*{FUNCTIONS_SUFFIX}/")):
        digest.update(function_dir.encode())
        for entry in sorted(os.scandir(function_dir), key=lambda entry: entry.name):
            stat = entry.stat()
            digest.update(f'{entry.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def _list_valid_files(input_dir: str, use_cache: bool, jobs: int) -> List[str]:
    cache_path = os.path.join(input_dir, 'filecache.pkl')
    key = _function_dirs_digest(input_dir)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and cached.get('format') == CACHE_FORMAT and cached.get('key') == key:
            console.print(f"[green]Loaded {len(cached['files'])} valid files from cache.[/green]")
            return cached['files']
        console.print("[yellow]File cache is stale (functions were re-extracted); rebuilding it.[/yellow]")

    console.print("[yellow]Start to build the file and save to cache...[/yellow]")
    llvm_files = find_llvm_files(input_dir, pattern=f"**/*{FUNCTIONS_SUFFIX}/*.ll")
    llvm_files = [os.path.abspath(f) for f in llvm_files]

    # Remove graph artifacts left inside function directories by older versions
    llvm_files = [f for f in llvm_files if not f.endswith(ARTIFACT_ENDINGS)]
    file_filter = FileFilter(num_processes=jobs)
    valid_files = sorted(file_filter.filter_files_parallel(llvm_files))
    if use_cache:
        with open(cache_path, 'wb') as f:
            pickle.dump({'format': CACHE_FORMAT, 'key': key, 'files': valid_files}, f)
        console.print(f"[green]Saved {len(valid_files)} valid files to cache.[/green]")
    return valid_files


def _completed_inputs(db_path: str, valid_files: List[str]) -> set:
    """Inputs whose row matches the current input content and toolchain and whose artifacts all exist."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot, signature FROM results").fetchall()
    rows = {row[0]: row for row in rows}
    completed = set()
    for path in valid_files:
        row = rows.get(path)
        if row is None or not all(row[1:5]) or not all(os.path.exists(p) for p in row[1:5]):
            continue
        if row[5] == result_signature(path):
            completed.add(path)
    return completed


def _prune_and_cleanup(input_dir: str, db_path: str, valid_files: List[str], cleanup: bool):
    """Drop rows of functions that are gone or filtered out, then (cleanup) delete unreferenced artifacts."""
    with sqlite3.connect(db_path) as conn:
        stored = [row[0] for row in conn.execute("SELECT input_path FROM results")]
        valid = set(valid_files)
        stale = [(path,) for path in stored if path not in valid]
        conn.executemany("DELETE FROM results WHERE input_path = ?", stale)
        conn.commit()
        referenced = {os.path.normpath(p) for row in conn.execute(
            "SELECT purify_path, instrumented_path, cfg_dot, ddg_dot FROM results") for p in row if p}
    if stale:
        console.print(f"[yellow]Removed {len(stale)} results of functions no longer present.[/yellow]")
    if not cleanup:
        return

    removed = 0
    for root, dirs, files in os.walk(input_dir):
        # Only GraphBuilder's own artifacts: leftovers in *_graphs (interrupted runs) and, from older
        # versions, artifacts written inside *_functions; artifacts of recorded results are kept
        if not root.endswith((ARTIFACTS_SUFFIX, FUNCTIONS_SUFFIX)):
            continue
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(ARTIFACT_ENDINGS) and path not in referenced:
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    continue
    console.print(f"[green]Cleaned up {removed} unreferenced graph artifacts[/green]")


def pipeline(input_dir: str, cleanup: bool = True, use_cache: bool = True, jobs: Optional[int] = None) -> bool:
    """Generate graphs for the functions under input_dir; False when the run is incomplete.

    Failure policy: a function whose IR opt rejects (fails verification) is skipped and reported, and its old
    row is removed. Any other pass failure, e.g. an unwritable or full output, a broken plugin, or a missing
    output, stops the run and leaves that chunk's rows as they were. The run also fails when the toolchain
    check fails, when a worker fails, or when every attempted function was rejected while none was already
    up to date.
    """
    jobs = available_cpu_count() if jobs is None else jobs
    if jobs < 1:
        raise ValueError("jobs must be positive")
    input_dir = os.path.abspath(input_dir)
    db_path = os.path.join(input_dir, 'results.db')
    # Before anything prunes, cleans or signs results: a broken toolchain must not delete valid ones
    try:
        check_toolchain()
    except RuntimeError as error:
        console.print(f"[red]Toolchain check failed: {error}[/red]")
        return False
    valid_files = _list_valid_files(input_dir, use_cache, jobs)

    console.print(f"[yellow]Filtering finished Files...[/yellow]")
    init_database(db_path)
    _prune_and_cleanup(input_dir, db_path, valid_files, cleanup)

    # Skip only results that still match the input and toolchain and whose artifacts exist
    completed = _completed_inputs(db_path, valid_files)
    valid_files = [f for f in valid_files if f not in completed]
    console.print(f"[green]Filtered {len(valid_files)} files to process ({len(completed)} up to date).[/green]")
    # Use multiprocessing to process files in parallel
    console.print("[cyan]Starting file processing in parallel...[/cyan]")

    chunk_size = 100
    chunks = [valid_files[i:i + chunk_size] for i in range(0, len(valid_files), chunk_size)]
    generated = failed_count = 0
    failed_examples = []

    with closing(sqlite3.connect(db_path)) as connection, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress, MP_CONTEXT.Pool(processes=jobs) as pool:
        task = progress.add_task("[cyan]Processing files...", total=len(valid_files))
        try:
            # Results come back to this process, which alone writes the database
            for results, failed in pool.imap_unordered(process_chunk, chunks):
                write_results(connection, results, failed)
                generated += len(results)
                failed_count += len(failed)
                failed_examples += failed[:3 - len(failed_examples)]
                progress.update(task, advance=len(results) + len(failed))
        except Exception as error:  # a worker raised: not a per-function failure, the run is incomplete
            console.print(f"[red]Graph generation incomplete after {generated} functions: {error!r}[/red]")
            return False

    console.print(f"[blue]Generated {generated}, failed {failed_count} functions. Results saved to {db_path}[/blue]")
    if failed_count:
        console.print(f"[yellow]{failed_count} functions were rejected by the passes and skipped, e.g. {failed_examples}[/yellow]")
    if failed_count and not generated and not completed:
        # Permanently invalid functions are retried on every run, so only "nothing at all" is a failure
        console.print("[red]Every attempted function failed and none was already up to date.[/red]")
        return False
    console.print("[green]Pipeline completed successfully![/green]")
    return True

app = typer.Typer()

@app.command()
def main(
    input_dir: str = typer.Argument(..., help="Input directory containing LLVM IR files"),
    cleanup: bool = typer.Option(True, "--cleanup/--no-cleanup", help="Delete graph artifacts no result refers to (interrupted runs, older layouts)"),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Use cached file list if the extracted functions are unchanged"),
    jobs: int = typer.Option(available_cpu_count(), "--jobs", "-j", min=1,
                             help="Worker processes for filtering and graph generation (default: CPUs allocated to this process)")
):
    """
    Process LLVM IR files to generate purified IR, DDG, and CFG files.

    This tool processes LLVM IR files in parallel, generating:
    - Purified LLVM IR (metadata stripped)
    - Data Dependence Graph (DDG) in DOT format
    - Control Flow Graph (CFG) in DOT format

    Artifacts are written to `<binary>_graphs/` next to `<binary>_functions/`; results are stored in
    `results.db`, and a rerun only regenerates functions whose input, passes or opt changed.

    Exit status 1 when the toolchain check fails (opt or a pass plugin cannot process a known-good
    function), a pass fails on IR that opt accepts (unwritable/full output, plugin failure), a worker
    fails, or every attempted function was rejected with none already up to date. Functions whose IR opt
    rejects are otherwise skipped and listed.
    """
    if not pipeline(input_dir, cleanup, use_cache, jobs):
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
