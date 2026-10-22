#!/usr/bin/env python3
"""
Task 3: Extract individual functions from LLVM IR files

Only task2 outputs whose .reopt.json is ok (and still match their recorded hashes) are used.
Each .bc is split into a fresh temporary directory which then replaces <x>_functions as a
whole, so function files of an older .bc are never reused. <x>_functions.extract.json
records every published function file (sha256, LLVM name) together with the lifter's
per-function record (status and approximation tags) for provenance-aware dataset building.
"""
import csv
import os
import shutil
import sys
import time
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import artifact_state
from job_runner import default_workers, run_bounded
from utils import console, run_command

# Configuration
EXTRACT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_llvm_ir.sh")

app = typer.Typer()


def check_task3_input(bc_path: str, allow_legacy_inputs: bool) -> tuple[bool, str, dict]:
    if not os.path.exists(bc_path + artifact_state.REOPT_SUFFIX):
        if allow_legacy_inputs:
            return True, "legacy", {"state": "legacy-unverified"}
        return False, "missing .reopt.json (pass --allow-legacy-inputs to use unverified legacy bitcode)", {}
    ok, reason, state = artifact_state.check_reopt_output(bc_path)
    if not ok:
        return False, reason, {}
    return True, "verified", {
        "state": state["input"].get("state"),
        "reopt_state_sha256": artifact_state.sha256_file(bc_path + artifact_state.REOPT_SUFFIX),
        "config_token": state.get("config_token"),
        "approx": state.get("approx", []),
        "lift_input": state["input"].get("path"),
        "lift_state_sha256": state["input"].get("lift_state_sha256"),
        "memory_contract": state["input"].get("memory_contract"),
    }


def lift_records_by_llvm_name(ll_path: str | None, expected_sha256: str | None) -> dict:
    """
    llvm name -> lifter record (status, approximation tags) from the task1 state of the source .ll.
    The .lift.json must be the one recorded when the .bc was produced; a different (rerun/failed)
    state is never attached to an older .bc.
    """
    if not ll_path or not expected_sha256:
        return {}
    state_path = ll_path + artifact_state.LIFT_SUFFIX
    if not os.path.isfile(state_path) or artifact_state.sha256_file(state_path) != expected_sha256:
        raise RuntimeError("upstream .lift.json differs from the one this .bc was produced from")
    lift_state = artifact_state.read_json(state_path) or {}
    records = {}
    for record in lift_state.get("functions", []):
        name = record.get("llvm_name")
        if name:
            records[name] = {key: record[key] for key in ("ea", "name", "status", "approx") if key in record}
    return records


def extract_functions(file_path: str, output_path: str, input_info: dict, timeout: int = 3600):
    """Extract every defined function of `file_path` and publish `output_path` with its state."""
    started = time.time()
    output_path = output_path.rstrip(os.sep)
    state_path = output_path + artifact_state.EXTRACT_SUFFIX
    artifact_state.cleanup_stale_tmp(output_path)
    artifact_state.cleanup_stale_tmp(output_path + ".old")
    tmp_dir = artifact_state.tmp_path_for(output_path)
    state = {
        "schema": artifact_state.SCHEMA,
        "stage": "extract",
        "status": "error",
        "input": {**artifact_state.file_identity(file_path), **input_info},
        "output_dir": os.path.abspath(output_path),
    }
    try:
        os.makedirs(tmp_dir)
        success, stdout, stderr = run_command([EXTRACT_SCRIPT, file_path, tmp_dir],
                                              f"Extracting functions from {file_path}", timeout=timeout)
        if not success:
            raise RuntimeError(f"split_llvm_ir.sh failed: {(stderr or stdout)[-1500:]}")
        map_path = os.path.join(tmp_dir, "function_map.csv")
        if not os.path.exists(map_path):
            raise RuntimeError(f"no defined functions found: {(stderr or stdout)[-300:]}")
        lift_records = lift_records_by_llvm_name(input_info.get("lift_input"), input_info.get("lift_state_sha256"))
        functions, record_status = [], Counter()
        with open(map_path, newline="") as handle:
            for row in csv.DictReader(handle):
                name, file_name = row["OriginalFunctionName"], row["HashedFileName"]
                record = lift_records.get(name)
                record_status[record["status"] if record else "no_lift_record"] += 1
                functions.append({
                    "llvm_name": name,
                    "file": file_name,
                    "sha256": artifact_state.sha256_file(os.path.join(tmp_dir, file_name)),
                    "lift": record,
                })
        state["functions"] = functions
        state["counts"] = {"functions": len(functions), **dict(record_status)}
        state["status"] = artifact_state.STATUS_OK
        old_dir = artifact_state.tmp_path_for(output_path + ".old")
        if os.path.exists(output_path):
            os.replace(output_path, old_dir)
        os.replace(tmp_dir, output_path)
        artifact_state.fsync_dir(os.path.dirname(os.path.abspath(output_path)))
        shutil.rmtree(old_dir, ignore_errors=True)
    except Exception as exc:
        # an existing function directory is left in place; the failed state makes consumers reject it
        state["status"] = "failed"
        state["error"] = str(exc)[:2000]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        state["elapsed_s"] = round(time.time() - started, 3)
        artifact_state.write_json_atomic(state_path, state)
    return state["status"] == artifact_state.STATUS_OK, "", state.get("error", "")


def extract_output_reusable(bc_path: str, output_path: str) -> bool:
    ok, _, state = artifact_state.check_extract_dir(output_path)
    return ok and state["input"].get("sha256") == artifact_state.sha256_file(bc_path)


@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing .bc files"),
    workers: int = typer.Option(0, help="Concurrent jobs; 0 = min(CPUs/2, MemAvailable / --mem-per-job-mb)"),
    resume: bool = typer.Option(False, help="Skip function sets whose .extract.json is ok for the current .bc"),
    allow_legacy_inputs: bool = typer.Option(
        False, "--allow-legacy-inputs",
        help="Accept .bc files without a .reopt.json; outputs are recorded as legacy-unverified."),
    timeout: int = typer.Option(3600, "--timeout", help="Per-file timeout in seconds"),
    mem_per_job_mb: int = typer.Option(1024, "--mem-per-job-mb", help="Estimated peak memory per job"),
):
    """Extract individual functions from task2 outputs."""
    if not os.path.exists(input_path):
        console.print(f"[red]Error: Input path {input_path} does not exist.[/red]")
        raise typer.Exit(code=1)
    if workers <= 0:
        workers = default_workers(mem_per_job_mb)

    console.print(f"[green]Processing directory: {input_path}[/green]")
    console.print("[bold blue]Preparing function extraction tasks...[/bold blue]")
    extract_commands = []
    skipped_extract = 0
    rejected = Counter()
    rejected_examples = {}

    for root, dirs, files in os.walk(input_path):
        dirs[:] = [d for d in dirs if ".tmp." not in d]
        for file in files:
            if not file.endswith(".bc") or ".tmp." in file:
                continue
            file_path = os.path.join(root, file)
            usable, reason, input_info = check_task3_input(file_path, allow_legacy_inputs)
            if not usable:
                rejected[reason] += 1
                rejected_examples.setdefault(reason, file_path)
                continue
            output_folder_path = os.path.splitext(file_path)[0] + "_functions"
            if resume and extract_output_reusable(file_path, output_folder_path):
                skipped_extract += 1
                continue
            extract_commands.append([file_path, output_folder_path, input_info, timeout])

    for reason, count in rejected.items():
        console.print(f"[red]Rejected {count} inputs: {reason} (e.g. {rejected_examples[reason]})[/red]")
    if resume and skipped_extract > 0:
        console.print(f"[yellow]Skipping {skipped_extract} already extracted function sets[/yellow]")

    console.print(f"[bold blue]Starting Task 3: Extracting functions from {len(extract_commands)} files[/bold blue]")
    if len(extract_commands) == 0:
        console.print("[yellow]No files to extract functions from, task completed[/yellow]")
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
        extract_task = progress.add_task("Extracting individual functions", total=len(extract_commands))

        def done(cmd, outcome):
            if isinstance(outcome, Exception):
                outcome = (False, "", str(outcome))
            success, _, error = outcome
            if success:
                counts["ok"] += 1
            else:
                counts["failed"] += 1
                console.print(f"[red]Failed to extract functions from file: {cmd[0]}: {error[:300]}[/red]")
            progress.update(extract_task, advance=1)

        run_bounded(extract_commands, lambda cmd: extract_functions(*cmd), workers, mem_per_job_mb, done)

    console.print(f"[bold green]Task 3 completed! Success: {counts['ok']}, Failed: {counts['failed']}, "
                  f"Rejected inputs: {sum(rejected.values())}[/bold green]")
    if counts["failed"] or rejected:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
