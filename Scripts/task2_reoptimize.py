#!/usr/bin/env python3
"""
Task 2: Re-optimize LLVM IR files using clang
"""
import sys
import os
import re
import hashlib
import json
import time
from collections import Counter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from utils import (
    console,
    clang_target_args,
    run_command,
    ensure_directory,
    file_exists_and_not_empty,
    normalize_clang_opt_level,
)
import artifact_state
from job_runner import default_workers, run_bounded

app = typer.Typer()
TASK2_STATE_DIRNAME = ".task2_reoptimize_state"
TASK2_CLANG_REOPT_FLAGS = (
    "-fno-inline",
    "-fno-inline-functions",
    "-fno-pic",
    "-fno-pie",
)
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


def canonicalize_passes_for_opt_level(opt_level: str) -> tuple[str, ...] | None:
    return TASK2_CANONICALIZE_PASSES_BY_OPT_LEVEL.get(opt_level)


def sanitize_illegal_function_bitcasts(file_path: str, output_path: str) -> tuple[str | None, int]:
    """Drop invalid lifted stores that bitcast integers into function values.

    SSA names are function-local, so the set of invalid values is reset at every
    ``define``; a store in another function that reuses the same name is kept.
    Lines are streamed to a temporary file instead of being held in memory.
    """
    invalid_values = set()
    removed = 0
    sanitized_path = os.path.splitext(output_path)[0] + ".sanitized.ll"
    ensure_directory(os.path.dirname(sanitized_path))
    tmp_path = f"{sanitized_path}.tmp.{os.getpid()}"

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle, \
                open(tmp_path, "w", encoding="utf-8") as out:
            for line in handle:
                stripped = line.lstrip()
                if stripped.startswith("define "):
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

                out.write(line)
        if removed == 0:
            return None, 0
        os.replace(tmp_path, sanitized_path)
        return sanitized_path, removed
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def find_illegal_function_bitcasts(file_path: str, limit: int = 20) -> list[tuple[str, str]]:
    """(function, line) pairs where the lifter emitted an integer -> function-value bitcast."""
    found = []
    current = None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("define "):
                current = line.split("@", 1)[-1].split("(", 1)[0]
            elif ILLEGAL_FUNCTION_BITCAST_RE.match(line):
                found.append((current, line.strip()[:200]))
                if len(found) >= limit:
                    break
    return found


def tool_version(tool: str) -> str:
    success, stdout, _ = run_command([tool, "--version"], timeout=60)
    lines = [line.strip() for line in stdout.splitlines() if "version" in line.lower()]
    return lines[0] if success and lines else "unknown"


def task2_config(opt_level: str, allow_host_retarget: bool, unsafe_drop: bool) -> dict:
    passes = canonicalize_passes_for_opt_level(opt_level)
    config = {"opt_level": opt_level, "allow_host_retarget": allow_host_retarget,
              "unsafe_drop_illegal_bitcast_stores": unsafe_drop}
    if passes is not None:
        config.update(tool="opt", passes=list(passes), tool_version=tool_version("opt"))
    else:
        config.update(tool="clang", flags=list(TASK2_CLANG_REOPT_FLAGS), tool_version=tool_version("clang"))
    return config


def config_token(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def check_task2_input(file_path: str, allow_host_compat: bool, allow_legacy_inputs: bool) -> tuple[bool, str, dict]:
    """
    A task1 .ll is used only if its .lift.json says ok, the file is the one the state describes and
    the memory contract is the source layout. Legacy (stateless) and host-compat inputs need an
    explicit flag and are recorded as such in the output state.
    """
    state = artifact_state.read_json(file_path + artifact_state.LIFT_SUFFIX)
    if state is None:
        if allow_legacy_inputs:
            return True, "legacy", {"state": "legacy-unverified"}
        return False, "missing .lift.json (pass --allow-legacy-inputs to use unverified legacy IR)", {}
    contract = None if allow_host_compat else "source-layout"
    ok, reason, state = artifact_state.check_lift_output(file_path, contract)
    if ok and state.get("memory_contract") not in ("source-layout", "host-compat-unconverted"):
        ok, reason = False, f"unknown memory_contract {state.get('memory_contract')}"
    if not ok:
        return False, reason, {}
    return True, "verified", {
        "state": "verified",
        "lifter_id": state.get("lifter_id"),
        "memory_contract": state.get("memory_contract"),
        "target": state.get("target"),
        "lift_state_sha256": artifact_state.sha256_file(file_path + artifact_state.LIFT_SUFFIX),
    }


def reoptimize_file(file_path: str, output_path: str, opt_level: str, arch: str | None, input_info: dict,
                    config: dict, timeout: int = 3600):
    """
    Re-optimize one LLVM IR file into <output>.tmp, then publish <output> and <output>.reopt.json.
    On failure an existing <output> is kept but its state records the failure, so downstream
    stages reject it.
    """
    started = time.time()
    artifact_state.cleanup_stale_tmp(output_path)
    tmp_output = artifact_state.tmp_path_for(output_path)
    state_path = output_path + artifact_state.REOPT_SUFFIX
    canonicalize_passes = canonicalize_passes_for_opt_level(opt_level)
    state = {
        "schema": artifact_state.SCHEMA,
        "stage": "reopt",
        "status": "error",
        "input": {**artifact_state.file_identity(file_path), **input_info},
        "config": config,
        "config_token": config_token(config),
        "approx": [],
    }
    stdout = stderr = ""
    try:
        if canonicalize_passes is not None:
            # opt keeps the module's own triple/data layout
            command = ["opt", f"-passes={','.join(canonicalize_passes)}", file_path, "-o", tmp_output]
            description = f"Canonicalizing {file_path} with {opt_level}"
        else:
            try:
                target_args, target_info = clang_target_args(
                    file_path, ARCH_TO_CLANG_FLAG.get(arch) if arch else None, config["allow_host_retarget"])
            except ValueError as exc:
                raise RuntimeError(str(exc))
            state["target"] = target_info
            command = ["clang", *target_args, opt_level, "-c", "-emit-llvm", *TASK2_CLANG_REOPT_FLAGS,
                       file_path, "-o", tmp_output]
            description = f"Re-optimizing {file_path} with {opt_level}"

        success, stdout, stderr = run_command(command, description, timeout=timeout)
        if not success and canonicalize_passes is not None and "invalid cast opcode for cast from" in (stderr or stdout):
            offenders = find_illegal_function_bitcasts(file_path)
            state["illegal_function_bitcasts"] = [{"function": f, "line": line} for f, line in offenders]
            if not config["unsafe_drop_illegal_bitcast_stores"]:
                raise RuntimeError(
                    f"opt rejected integer->function bitcasts in {len({f for f, _ in offenders})} function(s) "
                    f"(e.g. {offenders[:1]}); deleting the stores is not semantics-preserving, so this input "
                    "fails (see --unsafe-drop-illegal-bitcast-stores)")
            sanitized_path, removed = sanitize_illegal_function_bitcasts(file_path, output_path)
            if sanitized_path is not None:
                state["approx"].append("dropped_illegal_bitcast_stores")
                state["dropped_lines"] = removed
                command = ["opt", f"-passes={','.join(canonicalize_passes)}", sanitized_path, "-o", tmp_output]
                success, stdout, stderr = run_command(command, f"Canonicalizing sanitized {file_path}", timeout=timeout)
        if not success:
            raise RuntimeError(f"{command[0]} failed: {(stderr or stdout)[-1500:]}")
        if not file_exists_and_not_empty(tmp_output):
            raise RuntimeError(f"{command[0]} finished successfully but produced no output")
        state["output"] = {"path": os.path.abspath(output_path), "size": os.path.getsize(tmp_output),
                           "sha256": artifact_state.sha256_file(tmp_output)}
        state["status"] = artifact_state.STATUS_OK
        os.replace(tmp_output, output_path)
        artifact_state.fsync_dir(os.path.dirname(os.path.abspath(output_path)))
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)[:2000]
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        state["elapsed_s"] = round(time.time() - started, 3)
        artifact_state.write_json_atomic(state_path, state)
    return state["status"] == artifact_state.STATUS_OK, stdout, state.get("error", "")


def reopt_output_reusable(file_path: str, output_path: str, token: str) -> bool:
    ok, _, state = artifact_state.check_reopt_output(output_path)
    if not ok or state.get("config_token") != token:
        return False
    return state["input"].get("sha256") == artifact_state.sha256_file(file_path)


@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .ll files"),
    workers: int = typer.Option(0, help="Concurrent jobs; 0 = min(CPUs/2, MemAvailable / --mem-per-job-mb)"),
    resume: bool = typer.Option(False, help="Skip outputs whose .reopt.json is ok for the same input hash and configuration"),
    opt_level: str = typer.Option(
        "O3",
        "--opt-level",
        help="Task 2 optimization level, e.g. O0, O1, O2, O3, Os, Og, Oz, or Oc",
    ),
    arch: str = typer.Option(
        "auto",
        "--arch",
        help="Bitness used only with --allow-host-retarget: auto, m32, or m64.",
    ),
    allow_host_retarget: bool = typer.Option(
        False, "--allow-host-retarget",
        help="Legacy clang -O? path: compile with -m32/-m64 for the host, overriding module triples (recorded)."),
    allow_host_compat: bool = typer.Option(
        False, "--allow-host-compat",
        help="Accept task1 outputs lifted in host-compat mode (no memory-semantics conversion)."),
    allow_legacy_inputs: bool = typer.Option(
        False, "--allow-legacy-inputs",
        help="Accept .ll files without a .lift.json; outputs are recorded as legacy-unverified."),
    unsafe_drop_illegal_bitcast_stores: bool = typer.Option(
        False, "--unsafe-drop-illegal-bitcast-stores",
        help="UNSAFE legacy fallback: delete integer->function bitcasts and their stores and retry (not "
             "semantics-preserving; recorded as an approximation)."),
    timeout: int = typer.Option(3600, "--timeout", help="Per-file timeout in seconds"),
    mem_per_job_mb: int = typer.Option(2048, "--mem-per-job-mb", help="Estimated peak memory per job"),
):
    """Re-optimize LLVM IR files that task1 published successfully."""

    normalized_input_path = os.path.abspath(input_path)
    normalized_opt_level = normalize_clang_opt_level(opt_level)
    normalized_arch = normalize_arch_mode(arch)

    if not os.path.exists(normalized_input_path):
        console.print(f"[red]Error: Input path {normalized_input_path} does not exist.[/red]")
        raise typer.Exit(code=1)
    if workers <= 0:
        workers = default_workers(mem_per_job_mb)

    config = task2_config(normalized_opt_level, allow_host_retarget, unsafe_drop_illegal_bitcast_stores)
    token = config_token(config)
    console.print(f"[green]Processing directory: {normalized_input_path}[/green]")
    console.print(f"[green]Task 2 opt level: {normalized_opt_level} ({config['tool']}: {config['tool_version']})[/green]")
    if config["tool"] == "opt":
        console.print(f"[green]Task 2 canonicalize passes: {','.join(config['passes'])}[/green]")
    else:
        mode = "-m32/-m64 (host retarget)" if allow_host_retarget else "--target=<module triple>"
        console.print(f"[green]Task 2 clang target: {mode} {' '.join(TASK2_CLANG_REOPT_FLAGS)}[/green]")
    if allow_host_retarget:
        console.print("[yellow]Warning: --allow-host-retarget overrides module triples with the host target[/yellow]")
    if unsafe_drop_illegal_bitcast_stores:
        console.print("[yellow]Warning: unsafe bitcast-store deletion enabled; affected outputs are marked approx[/yellow]")

    console.print("[bold blue]Preparing re-optimization tasks...[/bold blue]")
    reopt_commands = []
    skipped_reopt = 0
    rejected = Counter()
    rejected_examples = {}
    unknown_arch_files = []

    for root, dirs, files in os.walk(normalized_input_path):
        if TASK2_STATE_DIRNAME in dirs:
            dirs.remove(TASK2_STATE_DIRNAME)
        if os.path.basename(root).endswith("_functions"):
            continue

        for file in files:
            if not file.endswith(".ll") or file.endswith(".sanitized.ll") or ".tmp." in file:
                continue
            file_path = os.path.join(root, file)
            usable, reason, input_info = check_task2_input(file_path, allow_host_compat, allow_legacy_inputs)
            if not usable:
                rejected[reason] += 1
                rejected_examples.setdefault(reason, file_path)
                continue
            resolved_arch = resolve_arch(file_path, normalized_arch) if allow_host_retarget else None
            if allow_host_retarget and resolved_arch is None:
                unknown_arch_files.append(file_path)
                continue
            output_file_path = os.path.splitext(file_path)[0] + ".bc"
            if resume and reopt_output_reusable(file_path, output_file_path, token):
                skipped_reopt += 1
                continue
            reopt_commands.append([file_path, output_file_path, normalized_opt_level, resolved_arch,
                                   input_info, config, timeout])

    for reason, count in rejected.items():
        console.print(f"[red]Rejected {count} inputs: {reason} (e.g. {rejected_examples[reason]})[/red]")
    if unknown_arch_files:
        console.print(f"[red]Error: could not infer -m32/-m64 for {len(unknown_arch_files)} files under --allow-host-retarget.[/red]")
        raise typer.Exit(code=1)
    if resume and skipped_reopt > 0:
        console.print(f"[yellow]Skipping {skipped_reopt} outputs already valid for this input and configuration[/yellow]")

    console.print(
        f"[bold blue]Starting Task 2: Re-optimizing {len(reopt_commands)} LLVM IR files with {normalized_opt_level}[/bold blue]"
    )
    if len(reopt_commands) == 0:
        console.print("[yellow]No files to re-optimize, task completed[/yellow]")
        if rejected:
            raise typer.Exit(code=1)
        return

    counts = {"ok": 0, "failed": 0}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        reopt_task = progress.add_task("Re-optimizing LLVM IR files", total=len(reopt_commands))

        def done(cmd, outcome):
            if isinstance(outcome, Exception):
                outcome = (False, "", str(outcome))
            success, _, error = outcome
            if success:
                counts["ok"] += 1
            else:
                counts["failed"] += 1
                console.print(f"[red]Failed to re-optimize file: {cmd[0]}: {error[:300]}[/red]")
            progress.update(reopt_task, advance=1)

        run_bounded(reopt_commands, lambda cmd: reoptimize_file(*cmd), workers, mem_per_job_mb, done)

    console.print(f"[bold green]Task 2 completed! Success: {counts['ok']}, Failed: {counts['failed']}, "
                  f"Rejected inputs: {sum(rejected.values())}[/bold green]")
    if counts["failed"] or rejected:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
