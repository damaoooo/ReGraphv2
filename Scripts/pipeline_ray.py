#!/usr/bin/env python3
"""
Ray-based multi-node pipeline runner.

This script keeps the original task scripts untouched and reuses the same
per-file command semantics:
1. Generate .i64 files with IDA
2. Lift .i64 files to .ll with ida2llvm.py
3. Re-optimize .ll files to .bc with clang
4. Split .bc files into per-function artifacts (optional)
5. Recompile optimized .bc files to .re artifacts (optional)

The driver only does scheduling and progress reporting. Real work is
distributed by Ray as file batches, so it can run on a Ray cluster.
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
import socket
import uuid
from itertools import islice
from typing import Dict, Iterator, List, Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    console,
    directory_exists_and_not_empty,
    ensure_directory,
    file_exists_and_not_empty,
    normalize_clang_opt_level,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
BINARY_PATH = os.path.join(REPO_ROOT, "Binaries")
IDA2LLVM_SCRIPT = os.path.join(SCRIPT_DIR, "ida2llvm.py")
SPLIT_LLVM_IR_SCRIPT = os.path.join(SCRIPT_DIR, "split_llvm_ir.sh")

IDA_TEMP_SUFFIXES = (".idb", ".id0", ".id1", ".id2", ".til", ".nam", ".asm")
IGNORED_INPUT_SUFFIXES = IDA_TEMP_SUFFIXES + (".i64", ".ll", ".bc", ".c", ".cpp", ".h", ".hpp")
IGNORED_TEXT_SUFFIXES = (".txt", ".log", ".md", ".py", ".sh")


def opt_level_state_token(opt_level: str) -> str:
    return opt_level.lstrip("-").replace(os.sep, "_")


def clip_text(text: str, limit: int = 2000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def chunked(items: List[Dict[str, str]], batch_size: int) -> Iterator[List[Dict[str, str]]]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def normalize_relative_dir(root: str, base: str) -> str:
    relative_dir = os.path.relpath(root, base)
    return "" if relative_dir == "." else relative_dir


def first_non_empty(*values: Optional[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def find_executable(candidates: List[str]) -> str:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return ""


def command_exists(command: str) -> bool:
    return os.path.isfile(command) or shutil.which(command) is not None


def resolve_ida_bin(ida_bin: str, ida_path: str) -> str:
    if ida_bin:
        return ida_bin
    if ida_path:
        for name in ("idat64", "idat"):
            candidate = os.path.join(ida_path, name)
            if os.path.exists(candidate):
                return candidate
    return first_non_empty(
        os.environ.get("PIPELINE_RAY_IDA_BIN", ""),
        find_executable(["idat64", "idat"]),
        "idat",
    )


def resolve_lift_python(ida2llvm_python: str) -> str:
    return first_non_empty(
        ida2llvm_python,
        os.environ.get("PIPELINE_RAY_IDA2LLVM_PYTHON", ""),
        sys.executable,
        "python",
    )


def build_lift_command(item: Dict[str, str], config: Dict[str, object], output_path: str) -> List[str]:
    conda_env = str(config.get("conda_env", "") or "")
    if conda_env:
        return [
            str(config["conda_bin"]),
            "run",
            "-n",
            conda_env,
            str(config["ida2llvm_python"]),
            IDA2LLVM_SCRIPT,
            "-f",
            item["input"],
            "-o",
            output_path,
            "-v",
        ]

    return [
        str(config["ida2llvm_python"]),
        IDA2LLVM_SCRIPT,
        "-f",
        item["input"],
        "-o",
        output_path,
        "-v",
    ]


def validate_runtime_inputs(
    start_from: int,
    task1_start_from_step2: bool,
    enable_recompile: bool,
    skip_extract: bool,
    ida_bin: str,
    ida2llvm_python: str,
    conda_env: str,
    conda_bin: str,
    clang_bin: str,
    bash_bin: str,
) -> None:
    need_task1_step1 = start_from <= 1 and not task1_start_from_step2
    need_task1_step2 = start_from <= 1
    need_task2 = start_from <= 2
    need_task3 = start_from <= 3 and not skip_extract
    need_task4 = enable_recompile and start_from <= 4

    if need_task1_step1 and not command_exists(ida_bin):
        console.print(f"[red]Error: IDA executable not found: {ida_bin}[/red]")
        console.print("[yellow]Pass --ida-bin explicitly or put idat/idat64 on PATH.[/yellow]")
        raise typer.Exit(code=1)

    if need_task1_step2 and conda_env:
        if not command_exists(conda_bin):
            console.print(f"[red]Error: conda executable not found: {conda_bin}[/red]")
            console.print("[yellow]Either install conda on every node or stop using --conda-env and pass --ida2llvm-python.[/yellow]")
            raise typer.Exit(code=1)
    elif need_task1_step2 and not command_exists(ida2llvm_python):
        console.print(f"[red]Error: ida2llvm Python executable not found: {ida2llvm_python}[/red]")
        console.print("[yellow]Pass --ida2llvm-python explicitly so every worker can find the same interpreter.[/yellow]")
        raise typer.Exit(code=1)

    if need_task2 and not command_exists(clang_bin):
        console.print(f"[red]Error: clang executable not found: {clang_bin}[/red]")
        raise typer.Exit(code=1)

    if need_task4 and not command_exists(clang_bin):
        console.print(f"[red]Error: clang executable not found: {clang_bin}[/red]")
        raise typer.Exit(code=1)

    if need_task3 and not command_exists(bash_bin):
        console.print(f"[red]Error: shell executable not found: {bash_bin}[/red]")
        raise typer.Exit(code=1)


def marker_path_for_item(stage: str, item: Dict[str, str], config: Dict[str, object]) -> str:
    state_root = str(config["state_root"])
    if stage == "task1_step1":
        relative_path = os.path.relpath(item["input"], str(config["db_path"]))
    else:
        relative_path = os.path.relpath(item["output"], str(config["final_output_path"]))
    if stage == "task2":
        return os.path.join(
            state_root,
            stage,
            opt_level_state_token(str(config["task2_opt_level"])),
            relative_path + ".done",
        )
    return os.path.join(state_root, stage, relative_path + ".done")


def marker_exists(stage: str, item: Dict[str, str], config: Dict[str, object]) -> bool:
    return os.path.exists(marker_path_for_item(stage, item, config))


def write_marker(stage: str, item: Dict[str, str], config: Dict[str, object]) -> None:
    path = marker_path_for_item(stage, item, config)
    ensure_directory(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("done\n")


def remove_marker(stage: str, item: Dict[str, str], config: Dict[str, object]) -> None:
    path = marker_path_for_item(stage, item, config)
    if os.path.exists(path):
        os.remove(path)


def failure_relative_path_for_item(stage: str, item: Dict[str, str], config: Dict[str, object]) -> str:
    if stage == "task1_step1":
        return os.path.relpath(item["input"], str(config["db_path"]))
    if item.get("output"):
        return os.path.relpath(item["output"], str(config["final_output_path"]))
    return os.path.relpath(item["input"], str(config["final_output_path"]))


def live_failure_log_path_for_item(stage: str, item: Dict[str, str], config: Dict[str, object]) -> str:
    live_failure_root = str(config["live_failure_root"])
    relative_path = failure_relative_path_for_item(stage, item, config)
    return os.path.join(live_failure_root, stage, relative_path + ".log")


def clear_live_failure_log(stage: str, item: Dict[str, str], config: Dict[str, object]) -> None:
    path = live_failure_log_path_for_item(stage, item, config)
    if os.path.exists(path):
        os.remove(path)


def write_live_failure_log(result: Dict[str, object], config: Dict[str, object]) -> None:
    stage = str(result["stage"])
    item = {
        "input": str(result["input"]),
        "output": str(result.get("output", "")),
    }
    path = live_failure_log_path_for_item(stage, item, config)
    ensure_directory(os.path.dirname(path))

    hostname = socket.gethostname()
    pid = os.getpid()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"Stage: {stage}\n")
        handle.write(f"Host: {hostname}\n")
        handle.write(f"PID: {pid}\n")
        handle.write(f"Input: {result['input']}\n")
        if result.get("output"):
            handle.write(f"Output: {result['output']}\n")
        if result.get("command"):
            handle.write(f"Command: {result['command']}\n")
        handle.write(f"Return code: {result.get('returncode', -1)}\n")
        if result.get("error"):
            handle.write("Error:\n")
            handle.write(f"{result['error']}\n")


def temp_file_path(final_path: str) -> str:
    return f"{final_path}.raytmp.{uuid.uuid4().hex}"


def temp_dir_path(final_path: str) -> str:
    return f"{final_path}.raytmp.{uuid.uuid4().hex}"


def remove_path_if_exists(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def prune_hidden_dirs(dirs: List[str]) -> None:
    """In-place filter for os.walk dirs to skip dot-prefixed directories."""
    dirs[:] = [name for name in dirs if not name.startswith(".")]


def run_process(command: List[str], timeout: Optional[int] = None) -> Dict[str, object]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"Command timed out after {timeout} seconds\n{exc.stderr or ''}",
        }
    except Exception as exc:  # pragma: no cover - defensive guard
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def build_result(
    stage: str,
    item: Dict[str, str],
    command: List[str],
    process_result: Dict[str, object],
    extra_error: str = "",
) -> Dict[str, object]:
    success = bool(process_result["success"])
    error_parts = []
    if not success:
        if extra_error:
            error_parts.append(extra_error)
        stdout = clip_text(str(process_result.get("stdout", "")))
        stderr = clip_text(str(process_result.get("stderr", "")))
        if stdout:
            error_parts.append(f"stdout:\n{stdout}")
        if stderr:
            error_parts.append(f"stderr:\n{stderr}")

    return {
        "stage": stage,
        "input": item["input"],
        "output": item.get("output", ""),
        "success": success,
        "returncode": int(process_result.get("returncode", -1)),
        "command": " ".join(command),
        "error": "\n".join(error_parts).strip(),
    }


def process_stage_item(stage: str, item: Dict[str, str], config: Dict[str, object]) -> Dict[str, object]:
    if stage == "task1_step1":
        binary_path = item["input"]
        i64_path = binary_path + ".i64"
        if bool(config.get("resume")) and file_exists_and_not_empty(i64_path) and not marker_exists(stage, item, config):
            remove_path_if_exists(i64_path)
        command = [str(config["ida_bin"]), "-A", "-B", binary_path]
        process_result = run_process(command)
        extra_error = ""
        if process_result["success"] and not file_exists_and_not_empty(i64_path):
            process_result["success"] = False
            extra_error = f"Expected output not found: {i64_path}"
        if process_result["success"]:
            write_marker(stage, item, config)
            clear_live_failure_log(stage, item, config)
        else:
            remove_marker(stage, item, config)
        result = build_result(stage, item, command, process_result, extra_error=extra_error)
        if not result["success"]:
            write_live_failure_log(result, config)
        return result

    if stage == "task1_step2":
        final_output = item["output"]
        ensure_directory(os.path.dirname(final_output))
        tmp_output = temp_file_path(final_output)
        remove_path_if_exists(tmp_output)
        command = build_lift_command(item, config, tmp_output)
        process_result = run_process(command)
        extra_error = ""
        if process_result["success"]:
            if not file_exists_and_not_empty(tmp_output):
                process_result["success"] = False
                extra_error = f"Expected output not found: {tmp_output}"
            else:
                os.replace(tmp_output, final_output)
                write_marker(stage, item, config)
                clear_live_failure_log(stage, item, config)
        else:
            remove_marker(stage, item, config)
        if os.path.exists(tmp_output):
            remove_path_if_exists(tmp_output)
        result = build_result(stage, item, command, process_result, extra_error=extra_error)
        if not result["success"]:
            write_live_failure_log(result, config)
        return result

    if stage == "task2":
        final_output = item["output"]
        ensure_directory(os.path.dirname(final_output))
        tmp_output = temp_file_path(final_output)
        remove_path_if_exists(tmp_output)
        command = [
            str(config["clang_bin"]),
            "-m32",
            str(config["task2_opt_level"]),
            "-c",
            "-emit-llvm",
            "-fno-inline",
            item["input"],
            "-o",
            tmp_output,
        ]
        process_result = run_process(command, timeout=int(config["task2_timeout"]))
        extra_error = ""
        if process_result["success"]:
            if not file_exists_and_not_empty(tmp_output):
                process_result["success"] = False
                extra_error = f"Expected output not found: {tmp_output}"
            else:
                os.replace(tmp_output, final_output)
                write_marker(stage, item, config)
                clear_live_failure_log(stage, item, config)
        else:
            remove_marker(stage, item, config)
        if os.path.exists(tmp_output):
            remove_path_if_exists(tmp_output)
        result = build_result(stage, item, command, process_result, extra_error=extra_error)
        if not result["success"]:
            write_live_failure_log(result, config)
        return result

    if stage == "task3":
        final_output = item["output"]
        tmp_output = temp_dir_path(final_output)
        remove_path_if_exists(tmp_output)
        ensure_directory(os.path.dirname(final_output))
        ensure_directory(tmp_output)
        command = [str(config["bash_bin"]), SPLIT_LLVM_IR_SCRIPT, item["input"], tmp_output]
        process_result = run_process(command, timeout=int(config["task3_timeout"]))
        extra_error = ""
        if process_result["success"]:
            if not directory_exists_and_not_empty(tmp_output):
                process_result["success"] = False
                extra_error = f"Expected extracted output not found: {tmp_output}"
            else:
                if os.path.exists(final_output):
                    remove_path_if_exists(final_output)
                os.rename(tmp_output, final_output)
                write_marker(stage, item, config)
                clear_live_failure_log(stage, item, config)
        else:
            remove_marker(stage, item, config)
        if os.path.exists(tmp_output):
            remove_path_if_exists(tmp_output)
        result = build_result(stage, item, command, process_result, extra_error=extra_error)
        if not result["success"]:
            write_live_failure_log(result, config)
        return result

    if stage == "task4":
        final_output = item["output"]
        ensure_directory(os.path.dirname(final_output))
        tmp_output = temp_file_path(final_output)
        remove_path_if_exists(tmp_output)
        command = [
            str(config["clang_bin"]),
            "-c",
            "-fno-inline",
            item["input"],
            "-o",
            tmp_output,
        ]
        process_result = run_process(command, timeout=int(config["task4_timeout"]))
        extra_error = ""
        if process_result["success"]:
            if not file_exists_and_not_empty(tmp_output):
                process_result["success"] = False
                extra_error = f"Expected output not found: {tmp_output}"
            else:
                os.replace(tmp_output, final_output)
                write_marker(stage, item, config)
                clear_live_failure_log(stage, item, config)
        else:
            remove_marker(stage, item, config)
        if os.path.exists(tmp_output):
            remove_path_if_exists(tmp_output)
        result = build_result(stage, item, command, process_result, extra_error=extra_error)
        if not result["success"]:
            write_live_failure_log(result, config)
        return result

    raise ValueError(f"Unsupported stage: {stage}")


def process_batch(stage: str, batch: List[Dict[str, str]], config: Dict[str, object]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for item in batch:
        try:
            results.append(process_stage_item(stage, item, config))
        except Exception as exc:  # pragma: no cover - defensive guard
            result = {
                "stage": stage,
                "input": item["input"],
                "output": item.get("output", ""),
                "success": False,
                "returncode": -1,
                "command": "",
                "error": str(exc),
            }
            write_live_failure_log(result, config)
            results.append(
                result
            )
    return results


def clean_failed_ida_files(db_path: str) -> int:
    removed = 0
    for root, dirs, files in os.walk(db_path):
        prune_hidden_dirs(dirs)
        for file_name in files:
            if not file_name.endswith(IDA_TEMP_SUFFIXES):
                continue
            file_path = os.path.join(root, file_name)
            try:
                os.remove(file_path)
                removed += 1
            except OSError:
                continue
    return removed


def collect_i64_generation_tasks(
    db_path: str,
    resume: bool,
    config: Dict[str, object],
) -> tuple[List[Dict[str, str]], int]:
    tasks: List[Dict[str, str]] = []
    skipped = 0

    for root, dirs, files in os.walk(db_path):
        prune_hidden_dirs(dirs)
        for file_name in files:
            if file_name.endswith(IGNORED_INPUT_SUFFIXES):
                continue
            if file_name.startswith(".") or file_name.endswith(IGNORED_TEXT_SUFFIXES):
                continue

            file_path = os.path.join(root, file_name)
            i64_path = file_path + ".i64"
            item = {"input": file_path}
            if resume:
                if file_exists_and_not_empty(i64_path) and marker_exists("task1_step1", item, config):
                    skipped += 1
                    continue
            elif file_exists_and_not_empty(i64_path):
                skipped += 1
                continue

            tasks.append(item)

    tasks.sort(key=lambda item: item["input"])
    return tasks, skipped


def collect_lift_tasks(
    db_path: str,
    output_path: str,
    resume: bool,
    config: Dict[str, object],
) -> tuple[List[Dict[str, str]], int]:
    tasks: List[Dict[str, str]] = []
    skipped = 0

    for root, dirs, files in os.walk(db_path):
        prune_hidden_dirs(dirs)
        for file_name in files:
            if not file_name.endswith(".i64"):
                continue

            file_path = os.path.join(root, file_name)
            relative_dir = normalize_relative_dir(root, db_path)
            output_dir = os.path.join(output_path, relative_dir)
            output_file_path = os.path.join(output_dir, file_name.replace(".i64", "")) + ".ll"

            item = {"input": file_path, "output": output_file_path}
            if resume and file_exists_and_not_empty(output_file_path) and marker_exists("task1_step2", item, config):
                skipped += 1
                continue

            tasks.append(item)

    tasks.sort(key=lambda item: item["input"])
    return tasks, skipped


def collect_reoptimize_tasks(
    input_path: str,
    resume: bool,
    config: Dict[str, object],
) -> tuple[List[Dict[str, str]], int]:
    tasks: List[Dict[str, str]] = []
    skipped = 0

    for root, dirs, files in os.walk(input_path):
        prune_hidden_dirs(dirs)
        for file_name in files:
            if not file_name.endswith(".ll"):
                continue

            file_path = os.path.join(root, file_name)
            output_path = os.path.splitext(file_path)[0] + ".bc"

            item = {"input": file_path, "output": output_path}
            if resume and file_exists_and_not_empty(output_path) and marker_exists("task2", item, config):
                skipped += 1
                continue

            tasks.append(item)

    tasks.sort(key=lambda item: item["input"])
    return tasks, skipped


def collect_extract_tasks(
    input_path: str,
    resume: bool,
    config: Dict[str, object],
) -> tuple[List[Dict[str, str]], int]:
    tasks: List[Dict[str, str]] = []
    skipped = 0

    for root, dirs, files in os.walk(input_path):
        prune_hidden_dirs(dirs)
        for file_name in files:
            if not file_name.endswith(".bc"):
                continue

            file_path = os.path.join(root, file_name)
            output_path = os.path.splitext(file_path)[0] + "_functions"

            item = {"input": file_path, "output": output_path}
            if resume and directory_exists_and_not_empty(output_path) and marker_exists("task3", item, config):
                skipped += 1
                continue

            tasks.append(item)

    tasks.sort(key=lambda item: item["input"])
    return tasks, skipped


def collect_recompile_tasks(
    input_path: str,
    resume: bool,
    config: Dict[str, object],
) -> tuple[List[Dict[str, str]], int]:
    tasks: List[Dict[str, str]] = []
    skipped = 0

    for root, dirs, files in os.walk(input_path):
        prune_hidden_dirs(dirs)
        for file_name in files:
            if not file_name.endswith(".bc"):
                continue

            file_path = os.path.join(root, file_name)
            output_path = os.path.splitext(file_path)[0] + ".re"

            item = {"input": file_path, "output": output_path}
            if resume and file_exists_and_not_empty(output_path) and marker_exists("task4", item, config):
                skipped += 1
                continue

            tasks.append(item)

    tasks.sort(key=lambda item: item["input"])
    return tasks, skipped


def write_failure_report(log_path: str, stage: str, failures: List[Dict[str, object]]) -> None:
    if not failures:
        return

    ensure_directory(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write(f"Stage: {stage}\n")
        handle.write(f"Failure count: {len(failures)}\n")
        handle.write("=" * 80 + "\n")
        for failure in failures:
            handle.write(f"Input: {failure['input']}\n")
            if failure.get("output"):
                handle.write(f"Output: {failure['output']}\n")
            if failure.get("command"):
                handle.write(f"Command: {failure['command']}\n")
            handle.write(f"Return code: {failure.get('returncode', -1)}\n")
            if failure.get("error"):
                handle.write(f"{failure['error']}\n")
            handle.write("-" * 80 + "\n")


def run_stage_with_ray(
    ray_module,
    remote_process_batch,
    stage: str,
    items: List[Dict[str, str]],
    batch_size: int,
    config: Dict[str, object],
    description: str,
) -> Dict[str, object]:
    if not items:
        console.print(f"[yellow]{description}: no work to do[/yellow]")
        return {"success_count": 0, "failed_count": 0, "failures": []}

    batches = list(chunked(items, batch_size))
    pending_refs = []
    ref_to_batch_size: Dict[object, int] = {}
    ref_to_inputs: Dict[object, List[Dict[str, str]]] = {}

    for batch in batches:
        ref = remote_process_batch.remote(stage, batch, config)
        pending_refs.append(ref)
        ref_to_batch_size[ref] = len(batch)
        ref_to_inputs[ref] = batch

    success_count = 0
    failed_count = 0
    failures: List[Dict[str, object]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[{task.completed}/{task.total}]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress_task = progress.add_task(description, total=len(items))

        while pending_refs:
            done_refs, pending_refs = ray_module.wait(pending_refs, num_returns=1)
            done_ref = done_refs[0]

            try:
                batch_results = ray_module.get(done_ref)
            except Exception as exc:
                batch_failures = []
                for item in ref_to_inputs[done_ref]:
                    batch_failures.append(
                        {
                            "stage": stage,
                            "input": item["input"],
                            "output": item.get("output", ""),
                            "success": False,
                            "returncode": -1,
                            "command": "",
                            "error": f"Ray batch failure: {exc}",
                        }
                    )
                failures.extend(batch_failures)
                failed_count += len(batch_failures)
                progress.update(progress_task, advance=ref_to_batch_size[done_ref])
                continue

            for result in batch_results:
                if result["success"]:
                    success_count += 1
                else:
                    failed_count += 1
                    failures.append(result)

            progress.update(progress_task, advance=len(batch_results))

    return {
        "success_count": success_count,
        "failed_count": failed_count,
        "failures": failures,
    }


def import_ray():
    try:
        import ray  # type: ignore

        return ray
    except ImportError as exc:
        console.print("[red]Ray is not installed in the current Python environment.[/red]")
        console.print("[yellow]Install it first, for example: pip install ray[/yellow]")
        raise typer.Exit(code=1) from exc


def resolve_ray_address(
    ray_mode: str,
    ray_address: str,
    ray_port: int,
) -> str:
    if ray_mode == "local":
        return "local"

    if ray_mode == "external":
        resolved = first_non_empty(ray_address, os.environ.get("RAY_ADDRESS", ""))
        if not resolved:
            console.print("[red]Error: external mode requires --ray-address or RAY_ADDRESS.[/red]")
            raise typer.Exit(code=1)
        return resolved

    if ray_mode == "slurm":
        head_ip = first_non_empty(
            os.environ.get("SLURM_RAY_HEAD_IP", ""),
            os.environ.get("SLURM_LAUNCH_NODE_IPADDR", ""),
        )
        resolved = first_non_empty(ray_address, os.environ.get("RAY_ADDRESS", ""), head_ip)
        if resolved and ":" not in resolved:
            resolved = f"{resolved}:{ray_port}"
        if not resolved:
            console.print("[red]Error: slurm mode requires Ray head address.[/red]")
            console.print("[yellow]Set one of: --ray-address, RAY_ADDRESS, or SLURM_RAY_HEAD_IP.[/yellow]")
            raise typer.Exit(code=1)
        return resolved

    # auto mode
    return first_non_empty(ray_address, os.environ.get("RAY_ADDRESS", ""), "auto")


def main(
    input_path: str = typer.Option("", help="Input directory. Defaults to Binaries/DataProcess-1"),
    output: str = typer.Option(..., help="Output root directory"),
    resume: bool = typer.Option(False, help="Resume from previous run"),
    start_from: int = typer.Option(1, help="Start from task number: 1, 2, 3, or 4"),
    task1_start_from_step2: bool = typer.Option(
        False,
        help="Task 1: skip .i64 generation and start directly from lifting existing .i64 files",
    ),
    enable_recompile: bool = typer.Option(
        False,
        help="Enable optional Task 4 recompile (.bc -> .re). Disabled by default",
    ),
    skip_extract: bool = typer.Option(
        False,
        help="Skip Task 3 function extraction and keep only lifted/re-optimized/recompiled whole-file artifacts",
    ),
    ray_mode: str = typer.Option(
        "auto",
        help="Ray bootstrap mode: auto | local | external | slurm",
    ),
    ray_address: str = typer.Option(
        "",
        help="Ray head address, e.g. 10.0.0.10:6379. In slurm mode this can be omitted if env vars are set",
    ),
    ray_port: int = typer.Option(6379, help="Ray head port used for slurm address fallback"),
    ray_with_runtime_env: bool = typer.Option(
        False,
        help="Send runtime_env working_dir to Ray workers. Keep disabled on shared-filesystem Slurm clusters",
    ),
    batch_size: int = typer.Option(8, help="How many files each Ray task processes sequentially"),
    ray_task_cpus: float = typer.Option(1.0, help="CPU resources requested by each Ray task"),
    ida_bin: str = typer.Option(
        "",
        help="IDA executable for Step 1, e.g. /opt/ida/idat64. Defaults to PIPELINE_RAY_IDA_BIN or idat64/idat on PATH",
    ),
    ida_path: str = typer.Option(
        "",
        help="Optional IDA installation directory. Used only to resolve idat64/idat when --ida-bin is not provided",
    ),
    ida2llvm_python: str = typer.Option(
        "",
        help="Python executable used to run ida2llvm.py. Defaults to PIPELINE_RAY_IDA2LLVM_PYTHON or the current Python",
    ),
    conda_env: str = typer.Option(
        "",
        help="Optional Conda env for ida2llvm.py. Leave empty in Docker or when workers already run the right Python env",
    ),
    conda_bin: str = typer.Option("conda", help="Conda executable used only when --conda-env is set"),
    clang_bin: str = typer.Option("clang", help="clang executable used in task 2"),
    opt_level: str = typer.Option(
        "O3",
        "--opt-level",
        help="clang optimization level for Task 2, e.g. O0, O1, O2, O3, Os, Oz",
    ),
    bash_bin: str = typer.Option("bash", help="Shell executable used for task 3"),
    task2_timeout: int = typer.Option(3600, help="Timeout in seconds for each clang task"),
    task3_timeout: int = typer.Option(3600, help="Timeout in seconds for each split task"),
    task4_timeout: int = typer.Option(3600, help="Timeout in seconds for each recompile task"),
):
    """Run the full pipeline on a Ray cluster with minimal changes to the original code."""

    if start_from not in (1, 2, 3, 4):
        console.print("[red]Error: --start-from must be 1, 2, 3, or 4.[/red]")
        raise typer.Exit(code=1)
    if batch_size <= 0:
        console.print("[red]Error: --batch-size must be greater than 0.[/red]")
        raise typer.Exit(code=1)
    if ray_task_cpus <= 0:
        console.print("[red]Error: --ray-task-cpus must be greater than 0.[/red]")
        raise typer.Exit(code=1)
    if ray_port <= 0:
        console.print("[red]Error: --ray-port must be greater than 0.[/red]")
        raise typer.Exit(code=1)
    if ray_mode not in ("auto", "local", "external", "slurm"):
        console.print("[red]Error: --ray-mode must be one of auto, local, external, slurm.[/red]")
        raise typer.Exit(code=1)

    normalized_opt_level = normalize_clang_opt_level(opt_level)

    if input_path:
        db_path = os.path.abspath(input_path)
        db = os.path.basename(db_path.rstrip(os.sep))
    else:
        db = "DataProcess-1"
        db_path = os.path.join(BINARY_PATH, db)

    if not os.path.exists(db_path):
        console.print(f"[red]Error: input path does not exist: {db_path}[/red]")
        raise typer.Exit(code=1)

    resolved_ida_bin = resolve_ida_bin(ida_bin=ida_bin, ida_path=ida_path)
    resolved_ida2llvm_python = resolve_lift_python(ida2llvm_python=ida2llvm_python)
    validate_runtime_inputs(
        start_from=start_from,
        task1_start_from_step2=task1_start_from_step2,
        enable_recompile=enable_recompile,
        skip_extract=skip_extract,
        ida_bin=resolved_ida_bin,
        ida2llvm_python=resolved_ida2llvm_python,
        conda_env=conda_env,
        conda_bin=conda_bin,
        clang_bin=clang_bin,
        bash_bin=bash_bin,
    )

    output_root = os.path.abspath(output)
    final_output_path = os.path.join(output_root, db)
    ensure_directory(output_root)
    ensure_directory(final_output_path)

    log_path = os.path.join(output_root, f"{db}_pipeline_ray_failures.log")

    resolved_ray_address = resolve_ray_address(
        ray_mode=ray_mode,
        ray_address=ray_address,
        ray_port=ray_port,
    )

    console.print(f"[bold green]Ray pipeline starting from task {start_from}[/bold green]")
    console.print(f"[green]Input: {db_path}[/green]")
    console.print(f"[green]Output: {final_output_path}[/green]")
    console.print(f"[green]Ray mode: {ray_mode}[/green]")
    console.print(f"[green]Ray address: {resolved_ray_address}[/green]")
    console.print(f"[green]Batch size: {batch_size}[/green]")
    console.print(f"[green]CPUs per Ray task: {ray_task_cpus}[/green]")
    console.print(f"[green]IDA bin: {resolved_ida_bin}[/green]")
    if conda_env:
        console.print(f"[green]ida2llvm via: {conda_bin} run -n {conda_env} {resolved_ida2llvm_python}[/green]")
    else:
        console.print(f"[green]ida2llvm via: {resolved_ida2llvm_python}[/green]")
    console.print(f"[green]clang: {clang_bin}[/green]")
    console.print(f"[green]Task 2 opt level: {normalized_opt_level}[/green]")
    if resume:
        console.print("[yellow]Resume mode enabled[/yellow]")
    if task1_start_from_step2:
        console.print("[yellow]Task 1 starts from Step 2[/yellow]")
    if skip_extract:
        console.print("[yellow]Task 3 extraction will be skipped[/yellow]")
    if enable_recompile:
        console.print("[yellow]Task 4 recompile enabled (.bc -> .re)[/yellow]")

    ray_module = import_ray()
    init_kwargs: Dict[str, object] = {}
    if resolved_ray_address and resolved_ray_address != "local":
        init_kwargs["address"] = resolved_ray_address
    if ray_with_runtime_env:
        init_kwargs["runtime_env"] = {"working_dir": SCRIPT_DIR}
    ray_module.init(**init_kwargs)

    remote_process_batch = ray_module.remote(num_cpus=ray_task_cpus)(process_batch)
    config: Dict[str, object] = {
        "ida_bin": resolved_ida_bin,
        "ida2llvm_python": resolved_ida2llvm_python,
        "conda_env": conda_env,
        "conda_bin": conda_bin,
        "clang_bin": clang_bin,
        "task2_opt_level": normalized_opt_level,
        "bash_bin": bash_bin,
        "task2_timeout": task2_timeout,
        "task3_timeout": task3_timeout,
        "task4_timeout": task4_timeout,
        "db_path": db_path,
        "final_output_path": final_output_path,
        "live_failure_root": os.path.join(output_root, ".pipeline_ray_live_failures", db),
        "state_root": os.path.join(output_root, ".pipeline_ray_state", db),
        "resume": resume,
    }

    all_failures: List[Dict[str, object]] = []

    try:
        if start_from <= 1:
            console.print("[bold blue]============================================================[/bold blue]")
            console.print("[bold blue]TASK 1: Binary to LLVM IR[/bold blue]")
            console.print("[bold blue]============================================================[/bold blue]")

            if not task1_start_from_step2:
                removed = clean_failed_ida_files(db_path)
                if removed > 0:
                    console.print(f"[yellow]Removed {removed} stale IDA temporary files[/yellow]")

                i64_tasks, i64_skipped = collect_i64_generation_tasks(
                    db_path=db_path,
                    resume=resume,
                    config=config,
                )
                if i64_skipped > 0:
                    console.print(f"[yellow]Skipping {i64_skipped} binaries with existing .i64 files[/yellow]")

                i64_summary = run_stage_with_ray(
                    ray_module=ray_module,
                    remote_process_batch=remote_process_batch,
                    stage="task1_step1",
                    items=i64_tasks,
                    batch_size=batch_size,
                    config=config,
                    description="Generating .i64 files with IDA",
                )
                console.print(
                    f"[bold green]Task 1 Step 1 finished. Success: {i64_summary['success_count']}, "
                    f"Failed: {i64_summary['failed_count']}[/bold green]"
                )
                write_failure_report(log_path, "task1_step1", i64_summary["failures"])
                all_failures.extend(i64_summary["failures"])

            lift_tasks, lift_skipped = collect_lift_tasks(
                db_path=db_path,
                output_path=final_output_path,
                resume=resume,
                config=config,
            )
            if lift_skipped > 0:
                console.print(f"[yellow]Skipping {lift_skipped} already lifted .ll files[/yellow]")

            lift_summary = run_stage_with_ray(
                ray_module=ray_module,
                remote_process_batch=remote_process_batch,
                stage="task1_step2",
                items=lift_tasks,
                batch_size=batch_size,
                config=config,
                description="Lifting .i64 files to LLVM IR",
            )
            console.print(
                f"[bold green]Task 1 Step 2 finished. Success: {lift_summary['success_count']}, "
                f"Failed: {lift_summary['failed_count']}[/bold green]"
            )
            write_failure_report(log_path, "task1_step2", lift_summary["failures"])
            all_failures.extend(lift_summary["failures"])

        if start_from <= 2:
            console.print("[bold blue]============================================================[/bold blue]")
            console.print("[bold blue]TASK 2: Re-optimize LLVM IR[/bold blue]")
            console.print("[bold blue]============================================================[/bold blue]")

            reopt_tasks, reopt_skipped = collect_reoptimize_tasks(
                input_path=final_output_path,
                resume=resume,
                config=config,
            )
            if reopt_skipped > 0:
                console.print(f"[yellow]Skipping {reopt_skipped} already re-optimized .bc files[/yellow]")

            reopt_summary = run_stage_with_ray(
                ray_module=ray_module,
                remote_process_batch=remote_process_batch,
                stage="task2",
                items=reopt_tasks,
                batch_size=batch_size,
                config=config,
                description="Re-optimizing LLVM IR with clang",
            )
            console.print(
                f"[bold green]Task 2 finished. Success: {reopt_summary['success_count']}, "
                f"Failed: {reopt_summary['failed_count']}[/bold green]"
            )
            write_failure_report(log_path, "task2", reopt_summary["failures"])
            all_failures.extend(reopt_summary["failures"])

        if start_from <= 3 and not skip_extract:
            console.print("[bold blue]============================================================[/bold blue]")
            console.print("[bold blue]TASK 3: Extract Functions[/bold blue]")
            console.print("[bold blue]============================================================[/bold blue]")

            extract_tasks, extract_skipped = collect_extract_tasks(
                input_path=final_output_path,
                resume=resume,
                config=config,
            )
            if extract_skipped > 0:
                console.print(f"[yellow]Skipping {extract_skipped} already extracted function directories[/yellow]")

            extract_summary = run_stage_with_ray(
                ray_module=ray_module,
                remote_process_batch=remote_process_batch,
                stage="task3",
                items=extract_tasks,
                batch_size=batch_size,
                config=config,
                description="Extracting per-function artifacts",
            )
            console.print(
                f"[bold green]Task 3 finished. Success: {extract_summary['success_count']}, "
                f"Failed: {extract_summary['failed_count']}[/bold green]"
            )
            write_failure_report(log_path, "task3", extract_summary["failures"])
            all_failures.extend(extract_summary["failures"])
        elif start_from <= 3 and skip_extract:
            console.print("[yellow]Skipping Task 3: function extraction disabled by --skip-extract[/yellow]")

        if enable_recompile and start_from <= 4:
            console.print("[bold blue]============================================================[/bold blue]")
            console.print("[bold blue]TASK 4: Recompile optimized .bc to .re[/bold blue]")
            console.print("[bold blue]============================================================[/bold blue]")

            recompile_tasks, recompile_skipped = collect_recompile_tasks(
                input_path=final_output_path,
                resume=resume,
                config=config,
            )
            if recompile_skipped > 0:
                console.print(f"[yellow]Skipping {recompile_skipped} already recompiled .re files[/yellow]")

            recompile_summary = run_stage_with_ray(
                ray_module=ray_module,
                remote_process_batch=remote_process_batch,
                stage="task4",
                items=recompile_tasks,
                batch_size=batch_size,
                config=config,
                description="Recompiling optimized .bc files to .re",
            )
            console.print(
                f"[bold green]Task 4 finished. Success: {recompile_summary['success_count']}, "
                f"Failed: {recompile_summary['failed_count']}[/bold green]"
            )
            write_failure_report(log_path, "task4", recompile_summary["failures"])
            all_failures.extend(recompile_summary["failures"])
    finally:
        ray_module.shutdown()

    console.print("[bold green]============================================================[/bold green]")
    console.print("[bold green]PIPELINE FINISHED[/bold green]")
    console.print("[bold green]============================================================[/bold green]")
    console.print(f"[green]Final output: {final_output_path}[/green]")

    if all_failures:
        console.print(f"[yellow]Finished with {len(all_failures)} failed items[/yellow]")
        console.print(f"[yellow]Failure log: {log_path}[/yellow]")
        raise typer.Exit(code=1)

    console.print("[bold green]All stages completed successfully[/bold green]")


if __name__ == "__main__":
    typer.run(main)
