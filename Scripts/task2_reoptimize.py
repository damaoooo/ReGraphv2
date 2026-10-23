#!/usr/bin/env python3
"""
Task 2: Re-optimize LLVM IR files using clang
"""
import sys
import os
import re
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import typer
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from utils import (
    available_cpu_count,
    console,
    run_command,
    ensure_directory,
    file_exists_and_not_empty,
    normalize_clang_opt_level,
)
from pipeline_cache import (
    build_input_signature,
    create_temporary_output_path,
    get_tool_identity,
    is_cached_output_valid,
    iter_completed_tasks,
    write_completion_record,
)

app = typer.Typer()
TASK2_STATE_DIRNAME = ".task2_reoptimize_state"
TASK2_CLANG_REOPT_FLAGS = (
    "-fno-inline",
    "-fno-inline-functions",
    "-fno-pic",
    "-fno-pie",
)
TASK2_CONFIG_TOKEN = "noinline_noinlinefunc_nopic_nopie"
TASK2_CANONICALIZE_PASSES_BY_OPT_LEVEL = {
    "-Oc": (
        "sroa",
        "mem2reg",
        "simplifycfg",
        "early-cse",
        "sccp",
        "correlated-propagation",
        "jump-threading",
        "simplifycfg",
        "reassociate",
        "gvn",
        "dce",
        "bdce",
        "adce",
        "simplifycfg",
    ),
}
TASK2_CANONICALIZE_CONFIG_TOKENS = {
    "-Oc": "canonicalize_v1_noattrs_noinline_noglobal",
}
ILLEGAL_FUNCTION_BITCAST_RE = re.compile(
    r"^\s*(?P<name>%\"[^\"]+\"|%[-A-Za-z0-9$._]+)\s*=\s*bitcast\s+"
    r"i(?:1|8|16|32|64|128)\s+.+\s+to\s+.+\([^)]*\)\s*$"
)
ARCH_TO_CLANG_FLAG = {
    "m32": "-m32",
    "m64": "-m64",
}
ARCH_PATH_PATTERNS = (
    (
        re.compile(
            r"(^|[\\/_.-])(x86[-_]?64|x64|amd64|arm[-_]?64|aarch64|mips[-_]?64)(?=$|[\\/_.-])"
        ),
        "m64",
    ),
    (
        re.compile(
            r"(^|[\\/_.-])(x86[-_]?32|i386|i686|x86|arm[-_]?32|mips[-_]?32)(?=$|[\\/_.-])"
        ),
        "m32",
    ),
)


def normalize_arch_mode(arch: str) -> str:
    value = arch.strip().lower()
    aliases = {
        "32": "m32",
        "64": "m64",
        "x86": "m32",
        "i386": "m32",
        "i686": "m32",
        "x64": "m64",
        "x86_64": "m64",
        "amd64": "m64",
    }
    value = aliases.get(value, value)
    if value not in ("auto", "m32", "m64"):
        raise typer.BadParameter("arch must be one of: auto, m32, m64")
    return value


def _detect_arch_from_text(text: str) -> str | None:
    normalized = text.lower()
    for pattern, arch in ARCH_PATH_PATTERNS:
        if pattern.search(normalized):
            return arch
    return None


def detect_arch_from_path(file_path: str) -> str | None:
    return _detect_arch_from_text(os.path.basename(file_path)) or _detect_arch_from_text(file_path)


def detect_arch_from_ir_header(file_path: str) -> str | None:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            header = handle.read(65536).lower()
    except OSError:
        return None

    if re.search(r'target\s+datalayout\s*=\s*"[^"]*[-_]p:32:32', header):
        return "m32"
    if re.search(r'target\s+datalayout\s*=\s*"[^"]*[-_]p:64:64', header):
        return "m64"

    triple_match = re.search(r'target\s+triple\s*=\s*"([^"]+)"', header)
    if not triple_match:
        return None

    triple = triple_match.group(1)
    if re.search(r"(x86_64|amd64|aarch64|arm64|mips64)", triple):
        return "m64"
    if re.search(r"(^|[-_])(i386|i486|i586|i686|arm|thumb|mips)([-_]|$)", triple):
        return "m32"
    return None


def resolve_arch(file_path: str, arch_mode: str) -> str | None:
    if arch_mode != "auto":
        return arch_mode
    return detect_arch_from_path(file_path) or detect_arch_from_ir_header(file_path)


def opt_level_state_token(opt_level: str, arch: str) -> str:
    opt_token = opt_level.lstrip("-").replace(os.sep, "_")
    config_token = TASK2_CANONICALIZE_CONFIG_TOKENS.get(opt_level, TASK2_CONFIG_TOKEN)
    return f"{opt_token}_{arch}_{config_token}"


def canonicalize_passes_for_opt_level(opt_level: str) -> tuple[str, ...] | None:
    return TASK2_CANONICALIZE_PASSES_BY_OPT_LEVEL.get(opt_level)


def build_cache_signature(file_path: str, opt_level: str, arch: str) -> dict:
    """Identify the source IR, optimization settings, and LLVM tool version."""
    passes = canonicalize_passes_for_opt_level(opt_level)
    return build_input_signature(file_path, {
        'opt': opt_level, 'arch': arch,
        'passes': list(passes) if passes else None,
        'flags': list(TASK2_CLANG_REOPT_FLAGS),
        'tool': get_tool_identity('opt' if passes else 'clang'),
    })


def sanitize_illegal_function_bitcasts(file_path: str, output_path: str) -> tuple[str | None, int]:
    """Drop invalid lifted stores that bitcast integers into function values."""
    invalid_values = set()
    output_lines = []
    removed = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.lstrip()
            # SSA local names can be reused in different functions.
            if stripped.startswith("define ") or stripped.startswith("}"):
                invalid_values.clear()
            match = ILLEGAL_FUNCTION_BITCAST_RE.match(line)
            if match:
                invalid_values.add(match.group("name"))
                removed += 1
                continue

            if invalid_values and stripped.startswith("store "):
                if any(f" {name}," in line or f"\t{name}," in line for name in invalid_values):
                    removed += 1
                    continue

            output_lines.append(line)

    if removed == 0:
        return None, 0

    sanitized_path = os.path.splitext(output_path)[0] + ".sanitized.ll"
    ensure_directory(os.path.dirname(sanitized_path))
    with open(sanitized_path, "w", encoding="utf-8") as handle:
        handle.write("".join(output_lines))
    return sanitized_path, removed


def marker_path_for_output(input_root: str, output_path: str, opt_level: str, arch: str) -> str:
    relative_output_path = os.path.relpath(output_path, input_root)
    return os.path.join(
        input_root,
        TASK2_STATE_DIRNAME,
        opt_level_state_token(opt_level, arch),
        relative_output_path + ".done",
    )


def reoptimize_file(file_path: str, output_path: str, opt_level: str, arch: str, marker_path: str):
    """Re-optimize into a temporary file, then publish and record successful output."""
    expected_signature = build_cache_signature(file_path, opt_level, arch)
    temporary_output_path = create_temporary_output_path(output_path)
    Path(marker_path).unlink(missing_ok=True)
    metadata = {}
    try:
        success, stdout, stderr = run_reoptimization_command(file_path, temporary_output_path, opt_level, arch, metadata)
        if success:
            os.replace(temporary_output_path, output_path)
            write_completion_record(marker_path, expected_signature, output_path, metadata)
        else:
            Path(output_path).unlink(missing_ok=True)
        return success, stdout, stderr
    finally:
        Path(temporary_output_path).unlink(missing_ok=True)
        Path(os.path.splitext(temporary_output_path)[0] + '.sanitized.ll').unlink(missing_ok=True)


def run_reoptimization_command(file_path: str, output_path: str, opt_level: str, arch: str, metadata: dict):
    """Run clang/opt and the legacy sanitization retry; do not publish cache state."""
    arch_flag = ARCH_TO_CLANG_FLAG[arch]
    canonicalize_passes = canonicalize_passes_for_opt_level(opt_level)
    sanitized_input = ""
    sanitized_removed = 0
    if canonicalize_passes is not None:
        command = [
            "opt",
            f"-passes={','.join(canonicalize_passes)}",
            file_path,
            "-o",
            output_path,
        ]
        tool = "opt"
        command_description = f"Canonicalizing {file_path} with {opt_level}"
    else:
        command = [
            "clang",
            arch_flag,
            opt_level,
            "-c",
            "-emit-llvm",
            *TASK2_CLANG_REOPT_FLAGS,
            file_path,
            "-o",
            output_path,
        ]
        tool = "clang"
        command_description = f"Re-optimizing {file_path} with {opt_level}"

    success, stdout, stderr = run_command(command, command_description)
    if (
        not success
        and canonicalize_passes is not None
        and "invalid cast opcode for cast from" in (stderr or stdout)
    ):
        sanitized_path, sanitized_removed = sanitize_illegal_function_bitcasts(file_path, output_path)
        if sanitized_path is not None:
            sanitized_input = sanitized_path
            metadata['sanitized_removed_lines'] = sanitized_removed
            console.print(f'[yellow]Warning: removed {sanitized_removed} invalid IR lines from {file_path}; this is not semantics-preserving canonicalization.[/yellow]')
            retry_command = [
                "opt",
                f"-passes={','.join(canonicalize_passes)}",
                sanitized_input,
                "-o",
                output_path,
            ]
            success, stdout, stderr = run_command(
                retry_command,
                f"Canonicalizing sanitized {file_path} with {opt_level}",
            )

    if success and not file_exists_and_not_empty(output_path):
        return False, stdout, f"{tool} finished successfully but output is missing: {output_path}"

    return success, stdout, stderr

def reoptimize_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    return reoptimize_file(*args)

@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .ll files"),
    workers: int = typer.Option(available_cpu_count(), min=1, help="Number of worker processes (default: CPUs allocated to this process)"),
    resume: bool = typer.Option(False, help="Resume from previous run, skip existing files"),
    opt_level: str = typer.Option(
        "O3",
        "--opt-level",
        help="Task 2 optimization level, e.g. O0, O1, O2, O3, Os, Og, Oz, or Oc",
    ),
    arch: str = typer.Option(
        "auto",
        "--arch",
        help="Target bitness for clang: auto, m32, or m64. auto infers from dataset-style file names.",
    ),
):
    """Re-optimize LLVM IR files using clang"""

    normalized_input_path = os.path.abspath(input_path)
    if workers < 1:
        raise typer.BadParameter('workers must be positive')
    normalized_opt_level = normalize_clang_opt_level(opt_level)
    normalized_arch = normalize_arch_mode(arch)

    if not os.path.exists(normalized_input_path):
        console.print(f"[red]Error: Input path {normalized_input_path} does not exist.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Processing directory: {normalized_input_path}[/green]")
    console.print(f"[green]Task 2 opt level: {normalized_opt_level}[/green]")
    console.print(f"[green]Task 2 arch mode: {normalized_arch}[/green]")
    canonicalize_passes = canonicalize_passes_for_opt_level(normalized_opt_level)
    if canonicalize_passes is not None:
        console.print(f"[green]Task 2 canonicalize passes: {','.join(canonicalize_passes)}[/green]")
    else:
        console.print(f"[green]Task 2 clang flags: {'/'.join(ARCH_TO_CLANG_FLAG.values())} {' '.join(TASK2_CLANG_REOPT_FLAGS)}[/green]")

    # Prepare re-optimization commands
    console.print("[bold blue]Preparing re-optimization tasks...[/bold blue]")
    reopt_commands = []
    skipped_reopt = 0
    rerun_existing = 0
    arch_counts = {"m32": 0, "m64": 0}
    unknown_arch_files = []

    for root, dirs, files in os.walk(normalized_input_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and not d.endswith('_functions')]
        if TASK2_STATE_DIRNAME in dirs:
            dirs.remove(TASK2_STATE_DIRNAME)

        # if dir ends with _functions, skip it
        if os.path.basename(root).endswith("_functions"):
            continue

        for file in files:
            if file.endswith(".ll") and not file.endswith('.sanitized.ll') and not file.startswith('.'):
                file_path = os.path.join(root, file)
                resolved_arch = resolve_arch(file_path, normalized_arch)
                if resolved_arch is None:
                    unknown_arch_files.append(file_path)
                    continue

                arch_counts[resolved_arch] += 1
                output_file_path = os.path.splitext(file_path)[0] + ".bc"
                marker_path = marker_path_for_output(
                    normalized_input_path,
                    output_file_path,
                    normalized_opt_level,
                    resolved_arch,
                )

                # Check if file already exists and we're resuming
                if resume and is_cached_output_valid(marker_path, build_cache_signature(file_path, normalized_opt_level, resolved_arch), output_file_path):
                    skipped_reopt += 1
                    continue

                if resume and file_exists_and_not_empty(output_file_path):
                    rerun_existing += 1

                cmd_arg = [
                    file_path,
                    output_file_path,
                    normalized_opt_level,
                    resolved_arch,
                    marker_path,
                ]
                reopt_commands.append(cmd_arg)

    if unknown_arch_files:
        console.print(
            f"[red]Error: could not infer 32/64-bit clang mode for {len(unknown_arch_files)} .ll files.[/red]"
        )
        for file_path in unknown_arch_files[:10]:
            console.print(f"[red]  - {file_path}[/red]")
        console.print("[red]Rename files with x86/x64/arm32/arm64/mips32/mips64, or pass --arch m32/--arch m64.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Task 2 resolved arches: m32={arch_counts['m32']}, m64={arch_counts['m64']}[/green]")

    if resume and skipped_reopt > 0:
        console.print(f"[yellow]Skipping {skipped_reopt} already re-optimized files[/yellow]")
    if resume and rerun_existing > 0:
        console.print(
            f"[yellow]Re-running {rerun_existing} existing .bc files because opt level/clang flags changed or no resume marker was found[/yellow]"
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
            for future, cmd in iter_completed_tasks(executor, reoptimize_file_wrapper, reopt_commands, workers * 2):
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
    if failed_count:
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
