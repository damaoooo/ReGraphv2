#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import task2_reoptimize


def parse_command(value: str) -> list[str]:
    return shlex.split(value)


def run_json(command: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    stdout = proc.stdout.strip()
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            result = json.loads(last_line)
        except json.JSONDecodeError:
            result = {
                "ok": False,
                "error": "worker did not emit JSON on the final stdout line",
                "stdout_tail": stdout[-4000:],
            }
    else:
        result = {"ok": False, "error": "worker emitted no stdout"}

    result["returncode"] = proc.returncode
    if proc.stderr:
        result["stderr_tail"] = proc.stderr[-4000:]
    return result


def count_defined_functions(ll_path: Path) -> int:
    count = 0
    with ll_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("define "):
                count += 1
    return count


def relative_to_or_name(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def normalize_binary_path(dataset_path: Path, binary: Path | None) -> Path | None:
    if binary is None:
        return None
    if binary.is_absolute():
        return binary.resolve()
    return (dataset_path / binary).resolve()


def list_binary_functions(
    repo_root: Path,
    worker_script: Path,
    ida_python_cmd: list[str],
    binary: Path,
) -> dict[str, Any]:
    return run_json(
        [
            *ida_python_cmd,
            str(worker_script),
            "--repo-root",
            str(repo_root),
            "list",
            "--binary",
            str(binary),
        ],
        repo_root,
    )


def build_sample_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_script = args.repo_root / "Scripts" / "ida_latency_worker.py"
    rng = random.Random(args.seed)

    if args.binary is not None:
        binaries = [args.binary]
    else:
        binaries = sorted(path for path in args.dataset_path.rglob("*.i64") if path.is_file())
        rng.shuffle(binaries)
        if args.max_candidate_binaries > 0:
            binaries = binaries[: args.max_candidate_binaries]

    tasks: list[dict[str, Any]] = []
    inspected = 0
    listed_ok = 0
    list_failures: list[dict[str, Any]] = []
    listed_ida_open_seconds_total = 0.0
    listed_elapsed_seconds_total = 0.0

    for binary in binaries:
        selected_so_far = sum(len(task["functions"]) for task in tasks)
        if selected_so_far >= args.functions:
            break

        inspected += 1
        result = list_binary_functions(args.repo_root, worker_script, args.ida_python_cmd, binary)
        listed_elapsed_seconds_total += float(result.get("elapsed_s", 0.0))
        listed_ida_open_seconds_total += float(result.get("ida_open_seconds", 0.0))

        if not result.get("ok"):
            list_failures.append(
                {
                    "binary": str(binary),
                    "returncode": result.get("returncode"),
                    "error": result.get("error"),
                    "stderr_tail": result.get("stderr_tail", "")[-1000:],
                }
            )
            continue

        functions = result.get("functions", [])
        if not functions:
            continue

        listed_ok += 1
        rng.shuffle(functions)
        remaining = args.functions - selected_so_far
        take = min(args.functions_per_binary, remaining, len(functions))
        if take <= 0:
            continue

        relative_binary = relative_to_or_name(binary, args.dataset_path)
        selected_functions = functions[:take]
        if args.lift_chunk_size > 0:
            relative_stem = relative_binary.with_suffix("")
            for chunk_index, chunk_start in enumerate(range(0, len(selected_functions), args.lift_chunk_size)):
                chunk_functions = selected_functions[chunk_start : chunk_start + args.lift_chunk_size]
                relative_output = relative_stem.parent / f"{relative_stem.name}.chunk{chunk_index:04d}.ll"
                tasks.append(
                    {
                        "binary": str(binary),
                        "relative_binary": str(relative_binary),
                        "relative_output": str(relative_output),
                        "chunk_index": chunk_index,
                        "functions": chunk_functions,
                    }
                )
        else:
            tasks.append(
                {
                    "binary": str(binary),
                    "relative_binary": str(relative_binary),
                    "relative_output": str(relative_binary.with_suffix(".ll")),
                    "chunk_index": 0,
                    "functions": selected_functions,
                }
            )

    selected = sum(len(task["functions"]) for task in tasks)
    selected_binaries = len({task["binary"] for task in tasks})
    if selected < args.functions and not args.allow_fewer_functions:
        raise RuntimeError(
            f"Only collected {selected} functions from {inspected} binaries. "
            "Pass --allow-fewer-functions to keep the partial sample."
        )

    manifest_stats = {
        "dataset_path": str(args.dataset_path),
        "binary": str(args.binary) if args.binary is not None else None,
        "seed": args.seed,
        "target_functions": args.functions,
        "selected_functions": selected,
        "selected_binaries": selected_binaries,
        "selected_lift_tasks": len(tasks),
        "inspected_binaries": inspected,
        "listed_ok_binaries": listed_ok,
        "listed_elapsed_seconds_total": listed_elapsed_seconds_total,
        "listed_ida_open_seconds_total": listed_ida_open_seconds_total,
        "list_failures": list_failures[:20],
        "list_failure_count": len(list_failures),
        "functions_per_binary_cap": args.functions_per_binary,
        "lift_chunk_size": args.lift_chunk_size,
    }
    return tasks, manifest_stats


def load_or_create_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.manifest.exists() and not args.rebuild_manifest:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        return data["tasks"], data["stats"]

    tasks, stats = build_sample_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps({"tasks": tasks, "stats": stats}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tasks, stats


def lift_one(
    repo_root: Path,
    worker_script: Path,
    ida_python_cmd: list[str],
    task: dict[str, Any],
    output_root: Path,
    target_mode: str,
) -> dict[str, Any]:
    rel = Path(task.get("relative_output") or Path(task["relative_binary"]).with_suffix(".ll"))
    output = output_root / rel
    eas = ",".join(hex(int(function["ea"])) for function in task["functions"])
    result = run_json(
        [
            *ida_python_cmd,
            str(worker_script),
            "--repo-root",
            str(repo_root),
            "lift",
            "--binary",
            task["binary"],
            "--output",
            str(output),
            "--eas",
            eas,
            "--target-mode",
            target_mode,
        ],
        repo_root,
    )
    result["relative_binary"] = task["relative_binary"]
    result["requested_functions"] = len(task["functions"])
    return result


def prepare_lift_tasks(
    tasks: list[dict[str, Any]],
    output_root: Path,
    copy_binary_per_task: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not copy_binary_per_task:
        return tasks, {
            "copy_binary_per_task": False,
            "binary_copies": 0,
            "excluded_binary_copy_seconds": 0.0,
        }

    copy_root = output_root / "_binary_copies"
    copy_root.mkdir(parents=True, exist_ok=True)
    prepared_tasks: list[dict[str, Any]] = []
    started = time.perf_counter()
    copied_bytes = 0

    for task_index, task in enumerate(tasks):
        source_binary = Path(task["binary"])
        task_copy_dir = copy_root / f"task_{task_index:05d}"
        task_copy_dir.mkdir(parents=True, exist_ok=True)
        copied_binary = task_copy_dir / source_binary.name
        shutil.copy2(source_binary, copied_binary)
        copied_bytes += copied_binary.stat().st_size

        prepared_task = dict(task)
        prepared_task["original_binary"] = task["binary"]
        prepared_task["binary"] = str(copied_binary)
        prepared_tasks.append(prepared_task)

    return prepared_tasks, {
        "copy_binary_per_task": True,
        "binary_copies": len(prepared_tasks),
        "copied_bytes": copied_bytes,
        "excluded_binary_copy_seconds": time.perf_counter() - started,
        "copy_root": str(copy_root),
    }


def simulate_parallel_wall(durations: list[float], workers: int) -> float:
    if not durations:
        return 0.0
    worker_count = max(1, min(workers, len(durations)))
    heap = [0.0] * worker_count
    heapq.heapify(heap)
    for duration in sorted(durations, reverse=True):
        current = heapq.heappop(heap)
        heapq.heappush(heap, current + max(0.0, duration))
    return max(heap)


def per_100(seconds: float, functions: int) -> float | None:
    if functions <= 0:
        return None
    return seconds * 100.0 / functions


def benchmark_lift(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    workers: int,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_tasks, preparation_stats = prepare_lift_tasks(
        tasks=tasks,
        output_root=output_root,
        copy_binary_per_task=args.copy_binary_per_task,
    )

    worker_script = args.repo_root / "Scripts" / "ida_latency_worker.py"
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                lift_one,
                args.repo_root,
                worker_script,
                args.ida_python_cmd,
                task,
                output_root,
                args.target_mode,
            ): task
            for task in prepared_tasks
        }
        progress_interval = max(1, math.ceil(len(prepared_tasks) / 10))
        for completed, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            results.append(result)
            if completed % progress_interval == 0 or completed == len(prepared_tasks):
                ok = sum(1 for item in results if item.get("ok"))
                print(f"[lift w={workers}] {completed}/{len(prepared_tasks)} tasks, ok={ok}", flush=True)

    wall_seconds = time.perf_counter() - started
    ok_results = [result for result in results if result.get("ok")]
    requested_functions = sum(int(result.get("requested_functions", 0)) for result in results)
    defined_functions = sum(int(result.get("defined_functions", 0)) for result in ok_results)
    ida_open_seconds_total = sum(float(result.get("ida_open_seconds", 0.0)) for result in ok_results)
    controller_initialize_seconds_total = sum(
        float(result.get("controller_initialize_seconds", 0.0)) for result in ok_results
    )
    function_emit_seconds_total = sum(float(result.get("function_emit_seconds", 0.0)) for result in ok_results)
    steady_state_durations = [float(result.get("steady_state_lift_seconds", 0.0)) for result in ok_results]
    steady_state_lift_seconds_total = sum(steady_state_durations)
    simulated_steady_state_wall_seconds = simulate_parallel_wall(steady_state_durations, workers)

    return {
        "workers": workers,
        "output_root": str(output_root),
        "wall_seconds": wall_seconds,
        "task_count": len(prepared_tasks),
        "successful_tasks": len(ok_results),
        "failed_tasks": len(results) - len(ok_results),
        **preparation_stats,
        "requested_functions": requested_functions,
        "defined_functions": defined_functions,
        "ida_open_seconds_total": ida_open_seconds_total,
        "controller_initialize_seconds_total": controller_initialize_seconds_total,
        "function_emit_seconds_total": function_emit_seconds_total,
        "steady_state_lift_seconds_total": steady_state_lift_seconds_total,
        "simulated_steady_state_wall_seconds": simulated_steady_state_wall_seconds,
        "seconds_per_100_defined_functions": per_100(wall_seconds, defined_functions),
        "steady_state_work_seconds_per_100_defined_functions": per_100(
            steady_state_lift_seconds_total,
            defined_functions,
        ),
        "simulated_steady_state_seconds_per_100_defined_functions": per_100(
            simulated_steady_state_wall_seconds,
            defined_functions,
        ),
        "function_emit_seconds_per_100_defined_functions": per_100(
            function_emit_seconds_total,
            defined_functions,
        ),
        "failed_examples": [result for result in results if not result.get("ok")][:10],
    }


def reopt_one(
    input_root: Path,
    ll_path: Path,
    output_root: Path,
    opt_level: str,
    arch_mode: str,
) -> dict[str, Any]:
    relative = ll_path.relative_to(input_root)
    bc_path = (output_root / relative).with_suffix(".bc")
    marker_path = output_root / ".task2_reoptimize_state" / opt_level.lstrip("-") / (str(relative) + ".done")
    bc_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_arch = task2_reoptimize.resolve_arch(str(ll_path), arch_mode)
    if resolved_arch is None:
        return {
            "ok": False,
            "input": str(ll_path),
            "output": str(bc_path),
            "relative": str(relative),
            "defined_functions": count_defined_functions(ll_path),
            "elapsed_s": 0.0,
            "error": "could not infer architecture; pass --arch m32 or --arch m64",
        }

    start = time.perf_counter()
    success, stdout, stderr = task2_reoptimize.reoptimize_file(
        str(ll_path),
        str(bc_path),
        opt_level,
        resolved_arch,
        str(marker_path),
    )
    elapsed_s = time.perf_counter() - start
    return {
        "ok": bool(success),
        "input": str(ll_path),
        "output": str(bc_path),
        "relative": str(relative),
        "defined_functions": count_defined_functions(ll_path),
        "elapsed_s": elapsed_s,
        "arch": resolved_arch,
        "stdout_tail": stdout[-2000:] if stdout else "",
        "stderr_tail": stderr[-2000:] if stderr else "",
    }


def benchmark_reopt(
    input_root: Path,
    ll_paths: list[Path],
    workers: int,
    output_root: Path,
    opt_level: str,
    arch_mode: str,
) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(reopt_one, input_root, ll_path, output_root, opt_level, arch_mode): ll_path
            for ll_path in ll_paths
        }
        progress_interval = max(1, math.ceil(len(ll_paths) / 10)) if ll_paths else 1
        for completed, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            results.append(result)
            if completed % progress_interval == 0 or completed == len(ll_paths):
                ok = sum(1 for item in results if item.get("ok"))
                print(f"[reopt {opt_level} w={workers}] {completed}/{len(ll_paths)} files, ok={ok}", flush=True)

    wall_seconds = time.perf_counter() - started
    ok_results = [result for result in results if result.get("ok")]
    functions = sum(int(result.get("defined_functions", 0)) for result in ok_results)
    work_seconds_total = sum(float(result.get("elapsed_s", 0.0)) for result in ok_results)
    simulated_wall_seconds = simulate_parallel_wall(
        [float(result.get("elapsed_s", 0.0)) for result in ok_results],
        workers,
    )

    return {
        "workers": workers,
        "opt_level": opt_level,
        "input_files": len(ll_paths),
        "successful_files": len(ok_results),
        "failed_files": len(results) - len(ok_results),
        "defined_functions": functions,
        "wall_seconds": wall_seconds,
        "work_seconds_total": work_seconds_total,
        "simulated_wall_seconds": simulated_wall_seconds,
        "seconds_per_100_defined_functions": per_100(wall_seconds, functions),
        "work_seconds_per_100_defined_functions": per_100(work_seconds_total, functions),
        "simulated_seconds_per_100_defined_functions": per_100(simulated_wall_seconds, functions),
        "failed_examples": [result for result in results if not result.get("ok")][:10],
    }


def format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    rows = [
        "| Stage | Workers | Files/Tasks | Functions | Wall s/100 | Steady-state s/100 | Work s/100 | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for workers, data in summary.get("lift", {}).items():
        rows.append(
            "| Lifting | "
            f"{workers} | {data.get('task_count', 0)} | {data.get('defined_functions', 0)} | "
            f"{format_seconds(data.get('seconds_per_100_defined_functions'))} | "
            f"{format_seconds(data.get('simulated_steady_state_seconds_per_100_defined_functions'))} | "
            f"{format_seconds(data.get('steady_state_work_seconds_per_100_defined_functions'))} | "
            "steady-state excludes IDA open/auto-analysis |"
        )
    for workers, data in summary.get("reopt", {}).items():
        rows.append(
            "| Reoptimization | "
            f"{workers} | {data.get('input_files', 0)} | {data.get('defined_functions', 0)} | "
            f"{format_seconds(data.get('seconds_per_100_defined_functions'))} | "
            f"{format_seconds(data.get('simulated_seconds_per_100_defined_functions'))} | "
            f"{format_seconds(data.get('work_seconds_per_100_defined_functions'))} | "
            f"{data.get('opt_level', '')} |"
        )

    manifest = summary.get("manifest_stats", {})
    lines = [
        "# Pipeline Latency Benchmark",
        "",
        f"- Dataset: `{summary.get('dataset_path')}`",
        f"- Binary: `{manifest.get('binary')}`",
        f"- Target functions: `{manifest.get('target_functions')}`",
        f"- Selected functions: `{manifest.get('selected_functions')}`",
        f"- Selected binaries: `{manifest.get('selected_binaries')}`",
        f"- Lift tasks: `{manifest.get('selected_lift_tasks')}`",
        f"- Lift chunk size: `{manifest.get('lift_chunk_size')}`",
        f"- Copy binary per lift task: `{summary.get('copy_binary_per_task')}`",
        f"- Reoptimization level: `{summary.get('opt_level')}`",
        "",
        *rows,
        "",
        "For the lifting steady-state column, IDA database open and auto-analysis are excluded.",
        "The wall column includes subprocess startup and IDA database initialization.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark lifting and reoptimization latency for a sampled set of functions. "
            "The lifting stage reports both end-to-end wall time and steady-state time "
            "with IDA database open/auto-analysis excluded."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-path", type=Path, default=Path("Binaries/Dataset-1/validation"))
    parser.add_argument("--binary", type=Path, default=None, help="Optional single .i64 file to sample from.")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/pipeline_latency_benchmark"))
    parser.add_argument(
        "--ida-python-cmd",
        default=os.environ.get("REGRAPH_IDA_PYTHON_CMD", "python"),
        help="Python command that can import IDA modules. Example: 'conda run -n ReLL python'.",
    )
    parser.add_argument("--functions", type=int, default=100)
    parser.add_argument("--functions-per-binary", type=int, default=100)
    parser.add_argument(
        "--lift-chunk-size",
        type=int,
        default=0,
        help=(
            "Split sampled functions from each binary into this many functions per lift task. "
            "Use a positive value to make single-binary parallel lifting actually create multiple tasks. "
            "0 keeps one lift task per sampled binary."
        ),
    )
    parser.add_argument(
        "--copy-binary-per-task",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Copy the sampled .i64 database once per lift task before timing. "
            "This avoids concurrent IDA processes opening the same database. "
            "Default: enabled when --lift-chunk-size is positive, disabled otherwise."
        ),
    )
    parser.add_argument("--allow-fewer-functions", action="store_true")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-candidate-binaries", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--opt-level", default="Oc", help="O0/O1/O2/O3/Og/Os/Oz/Oc/Oc2.")
    parser.add_argument("--arch", default="auto", help="auto, m32, or m64.")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--target-mode", default="host")
    parser.add_argument("--skip-lift", action="store_true")
    parser.add_argument("--skip-reopt", action="store_true")
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.dataset_path = (
        (args.repo_root / args.dataset_path).resolve()
        if not args.dataset_path.is_absolute()
        else args.dataset_path.resolve()
    )
    args.binary = normalize_binary_path(args.dataset_path, args.binary)
    args.work_dir = (
        (args.repo_root / args.work_dir).resolve()
        if not args.work_dir.is_absolute()
        else args.work_dir.resolve()
    )
    args.manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else args.work_dir / "manifest.json"
    )
    args.markdown_output = (
        args.markdown_output.resolve()
        if args.markdown_output is not None
        else args.work_dir / "pipeline_latency.md"
    )
    args.json_output = (
        args.json_output.resolve()
        if args.json_output is not None
        else args.work_dir / "summary.json"
    )
    args.ida_python_cmd = parse_command(args.ida_python_cmd)
    args.opt_level = task2_reoptimize.normalize_clang_opt_level(args.opt_level)
    args.arch = task2_reoptimize.normalize_arch_mode(args.arch)
    if args.copy_binary_per_task is None:
        args.copy_binary_per_task = args.lift_chunk_size > 0
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if not args.dataset_path.is_dir():
        parser.error(f"dataset path does not exist: {args.dataset_path}")
    if args.binary is not None and not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary}")
    if args.functions < 1:
        parser.error("--functions must be positive")
    if args.functions_per_binary < 1:
        parser.error("--functions-per-binary must be positive")
    if args.lift_chunk_size < 0:
        parser.error("--lift-chunk-size must be non-negative")
    if any(workers < 1 for workers in args.workers):
        parser.error("--workers values must be positive")

    tasks, manifest_stats = load_or_create_manifest(args)
    print(json.dumps({"manifest": manifest_stats}, indent=2, ensure_ascii=False), flush=True)

    summary: dict[str, Any] = {
        "repo_root": str(args.repo_root),
        "dataset_path": str(args.dataset_path),
        "work_dir": str(args.work_dir),
        "manifest": str(args.manifest),
        "manifest_stats": manifest_stats,
        "opt_level": args.opt_level,
        "arch": args.arch,
        "target_mode": args.target_mode,
        "ida_python_cmd": args.ida_python_cmd,
        "copy_binary_per_task": args.copy_binary_per_task,
        "lift": {},
        "reopt": {},
    }

    if not args.skip_lift:
        for workers in args.workers:
            lift_root = args.work_dir / f"lift_w{workers}"
            summary["lift"][str(workers)] = benchmark_lift(args, tasks, workers, lift_root)

    reference_workers = str(args.workers[0])
    reference_lift_root = Path(
        summary["lift"].get(reference_workers, {}).get(
            "output_root",
            args.work_dir / f"lift_w{args.workers[0]}",
        )
    )
    ll_paths = sorted(path for path in reference_lift_root.rglob("*.ll") if path.is_file())

    if not args.skip_reopt:
        for workers in args.workers:
            reopt_root = args.work_dir / f"reopt_{args.opt_level.lstrip('-')}_from_w{reference_workers}_w{workers}"
            summary["reopt"][str(workers)] = benchmark_reopt(
                input_root=reference_lift_root,
                ll_paths=ll_paths,
                workers=workers,
                output_root=reopt_root,
                opt_level=args.opt_level,
                arch_mode=args.arch,
            )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, args.markdown_output)

    print(json.dumps({"summary_json": str(args.json_output), "summary_md": str(args.markdown_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
