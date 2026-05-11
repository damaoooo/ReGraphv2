#!/usr/bin/env python3
"""
Task 2: Re-optimize LLVM IR files using clang
"""
import sys
import os
import re
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
TASK2_CLANG_REOPT_FLAGS = (
    "-fno-inline",
    "-fno-inline-functions",
    "-fno-pic",
    "-fno-pie",
)
TASK2_CONFIG_TOKEN = "noinline_noinlinefunc_nopic_nopie"
TASK2_CANONICALIZE_OPT_LEVEL = "-Oc"
TASK2_CANONICALIZE_PASSES = (
    "sroa",
    "mem2reg",
    "instcombine",
    "simplifycfg",
    "early-cse",
    "sccp",
    "correlated-propagation",
    "jump-threading",
    "simplifycfg",
    "reassociate",
    "instcombine",
    "gvn",
    "dce",
    "bdce",
    "adce",
    "simplifycfg",
    "instcombine",
)
TASK2_CANONICALIZE_CONFIG_TOKEN = "canonicalize_v1_noattrs_noinline_noglobal"
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
    if triple == "x86_64-unknown-linux-gnu":
        return None
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
    config_token = (
        TASK2_CANONICALIZE_CONFIG_TOKEN
        if opt_level == TASK2_CANONICALIZE_OPT_LEVEL
        else TASK2_CONFIG_TOKEN
    )
    return f"{opt_token}_{arch}_{config_token}"


def marker_path_for_output(input_root: str, output_path: str, opt_level: str, arch: str) -> str:
    relative_output_path = os.path.relpath(output_path, input_root)
    return os.path.join(
        input_root,
        TASK2_STATE_DIRNAME,
        opt_level_state_token(opt_level, arch),
        relative_output_path + ".done",
    )


def reoptimize_file(file_path: str, output_path: str, opt_level: str, arch: str, marker_path: str):
    """Re-optimize a single LLVM IR file using clang or the Oc canonicalizer."""
    if os.path.exists(marker_path):
        os.remove(marker_path)

    arch_flag = ARCH_TO_CLANG_FLAG[arch]
    if opt_level == TASK2_CANONICALIZE_OPT_LEVEL:
        command = [
            "opt",
            f"-passes={','.join(TASK2_CANONICALIZE_PASSES)}",
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

    if success and not file_exists_and_not_empty(output_path):
        return False, stdout, f"clang finished successfully but output is missing: {output_path}"

    if success:
        ensure_directory(os.path.dirname(marker_path))
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(f"opt_level={opt_level}\n")
            handle.write(f"tool={tool}\n")
            handle.write(f"arch={arch}\n")
            handle.write(f"arch_flag={arch_flag}\n")
            if opt_level == TASK2_CANONICALIZE_OPT_LEVEL:
                handle.write(f"passes={','.join(TASK2_CANONICALIZE_PASSES)}\n")
            else:
                handle.write(f"flags={' '.join(TASK2_CLANG_REOPT_FLAGS)}\n")

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
        help="Task 2 optimization level, e.g. O0, O1, O2, O3, Os, Og, Oz, or Oc canonicalization",
    ),
    arch: str = typer.Option(
        "auto",
        "--arch",
        help="Target bitness for clang: auto, m32, or m64. auto infers from dataset-style file names.",
    ),
):
    """Re-optimize LLVM IR files using clang"""

    normalized_input_path = os.path.abspath(input_path)
    normalized_opt_level = normalize_clang_opt_level(opt_level)
    normalized_arch = normalize_arch_mode(arch)

    if not os.path.exists(normalized_input_path):
        console.print(f"[red]Error: Input path {normalized_input_path} does not exist.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Processing directory: {normalized_input_path}[/green]")
    console.print(f"[green]Task 2 opt level: {normalized_opt_level}[/green]")
    console.print(f"[green]Task 2 arch mode: {normalized_arch}[/green]")
    if normalized_opt_level == TASK2_CANONICALIZE_OPT_LEVEL:
        console.print(f"[green]Task 2 canonicalize passes: {','.join(TASK2_CANONICALIZE_PASSES)}[/green]")
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
        if TASK2_STATE_DIRNAME in dirs:
            dirs.remove(TASK2_STATE_DIRNAME)

        # if dir ends with _functions, skip it
        if os.path.basename(root).endswith("_functions"):
            continue

        for file in files:
            if file.endswith(".ll"):
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
                if resume and file_exists_and_not_empty(output_file_path) and os.path.exists(marker_path):
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
