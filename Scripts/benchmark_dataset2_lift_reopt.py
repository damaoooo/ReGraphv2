#!/usr/bin/env python3
"""
Benchmark the full Dataset-2 lift + reoptimization pipeline.

Stage 1 matches the current lift pipeline's per-file command:
    conda run -n ReLL python Scripts/ida2llvm.py -f <input.i64> -o <output.ll> -v

Stage 2 matches the current reoptimization pipeline's per-file command:
    clang -m32 -O3 -c -emit-llvm -fno-inline <input.ll> -o <output.bc>

The script creates a temporary workspace for generated .ll/.bc files, runs the
entire selected Dataset-2 set, records stage wall-clock time plus per-file
timings, writes a compact JSON summary, and deletes the temporary directory by
default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "Binaries" / "Dataset-2"
IDA2LLVM_PATH = SCRIPT_DIR / "ida2llvm.py"


def iter_i64_files(dataset_path: Path, selection_order: str) -> List[Path]:
    files = [path for path in dataset_path.rglob("*.i64") if path.is_file()]
    if selection_order == "path_asc":
        return sorted(files, key=lambda path: str(path.relative_to(dataset_path)))
    return sorted(
        files,
        key=lambda path: (path.stat().st_size, str(path.relative_to(dataset_path))),
    )


def count_defined_functions(ll_path: Path) -> int:
    count = 0
    with ll_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("define "):
                count += 1
    return count


def clip_text(text: str, limit: int = 1200) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def run_command(command: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def run_lift_task(
    input_i64: Path,
    dataset_path: Path,
    ll_root: Path,
    conda_env: str,
) -> Dict[str, object]:
    relative_path = input_i64.relative_to(dataset_path)
    ll_path = (ll_root / relative_path).with_suffix(".ll")
    ll_path.parent.mkdir(parents=True, exist_ok=True)

    lift_cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(IDA2LLVM_PATH),
        "-f",
        str(input_i64),
        "-o",
        str(ll_path),
        "-v",
    ]

    start = time.perf_counter()
    lift_result = run_command(lift_cmd, REPO_ROOT)
    lift_seconds = time.perf_counter() - start

    success = (
        lift_result.returncode == 0
        and ll_path.exists()
        and ll_path.stat().st_size > 0
    )

    function_count = count_defined_functions(ll_path) if success else 0
    return {
        "stage": "lift",
        "file": str(relative_path),
        "input": str(input_i64),
        "ll_path": str(ll_path),
        "success": success,
        "returncode": lift_result.returncode,
        "lift_seconds": lift_seconds,
        "functions": function_count,
        "stdout": clip_text(lift_result.stdout),
        "stderr": clip_text(lift_result.stderr),
    }


def run_reopt_task(
    ll_path: Path,
    bc_path: Path,
    relative_path: str,
    function_count: int,
) -> Dict[str, object]:
    bc_path.parent.mkdir(parents=True, exist_ok=True)
    reopt_cmd = [
        "clang",
        "-m32",
        "-O3",
        "-c",
        "-emit-llvm",
        "-fno-inline",
        str(ll_path),
        "-o",
        str(bc_path),
    ]

    start = time.perf_counter()
    reopt_result = run_command(reopt_cmd, REPO_ROOT)
    reopt_seconds = time.perf_counter() - start

    success = (
        reopt_result.returncode == 0
        and bc_path.exists()
        and bc_path.stat().st_size > 0
    )
    return {
        "stage": "reopt",
        "file": relative_path,
        "ll_path": str(ll_path),
        "bc_path": str(bc_path),
        "success": success,
        "returncode": reopt_result.returncode,
        "reopt_seconds": reopt_seconds,
        "functions": function_count,
        "stdout": clip_text(reopt_result.stdout),
        "stderr": clip_text(reopt_result.stderr),
    }


def finalize_metric(seconds_value: float, functions_value: int) -> float | None:
    if functions_value <= 0:
        return None
    return seconds_value * 100.0 / functions_value


def print_stage_progress(stage: str, completed: int, total: int, detail: str) -> None:
    print(f"[{stage} {completed}/{total}] {detail}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Directory containing Dataset-2 .i64 files. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel workers used for each stage.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional hard cap for debugging. 0 means process all .i64 files.",
    )
    parser.add_argument(
        "--selection-order",
        choices=("size_asc", "path_asc"),
        default="size_asc",
        help="Traversal order for the selected .i64 files.",
    )
    parser.add_argument(
        "--temp-root",
        default="",
        help="Optional parent directory for the temporary workspace.",
    )
    parser.add_argument(
        "--conda-env",
        default="ReLL",
        help="Conda environment used for ida2llvm.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary workspace instead of deleting it.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failure in a stage.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_files < 0:
        parser.error("--max-files must be non-negative")

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.is_dir():
        parser.error(f"Dataset path does not exist or is not a directory: {dataset_path}")

    all_i64_files = iter_i64_files(dataset_path, args.selection_order)
    if not all_i64_files:
        parser.error(f"No .i64 files found under {dataset_path}")

    selected_i64_files = all_i64_files[: args.max_files] if args.max_files else all_i64_files

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="dataset2-lift-reopt-",
            dir=args.temp_root or None,
        )
    )
    ll_root = temp_dir / "ll"
    bc_root = temp_dir / "bc"
    ll_root.mkdir(parents=True, exist_ok=True)
    bc_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "dataset_path": str(dataset_path),
        "selection_order": args.selection_order,
        "workers": args.workers,
        "selected_i64_files": len(selected_i64_files),
        "max_files_limit": args.max_files,
        "temp_dir": str(temp_dir),
        "lift_attempted_files": len(selected_i64_files),
        "lift_successful_files": 0,
        "lift_failed_files": 0,
        "reopt_attempted_files": 0,
        "reopt_successful_files": 0,
        "reopt_failed_files": 0,
        "fully_successful_files": 0,
        "functions_lifted_total": 0,
        "functions_reoptimized_total": 0,
        "lift_file_seconds_total": 0.0,
        "reopt_file_seconds_total": 0.0,
        "lift_wall_seconds_total": 0.0,
        "reopt_wall_seconds_total": 0.0,
        "combined_wall_seconds_total": 0.0,
        "lift_wall_seconds_per_100_functions": None,
        "reopt_wall_seconds_per_100_functions": None,
        "combined_wall_seconds_per_100_functions": None,
        "lift_file_seconds_per_100_functions": None,
        "reopt_file_seconds_per_100_functions": None,
        "failures": [],
        "cleanup_performed": False,
    }

    fatal_error: RuntimeError | None = None

    try:
        print(
            f"Starting full Dataset-2 benchmark: files={len(selected_i64_files)} "
            f"workers={args.workers} temp_dir={temp_dir}",
            flush=True,
        )

        lift_results: List[Dict[str, object]] = []
        lift_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_input = {
                executor.submit(run_lift_task, input_i64, dataset_path, ll_root, args.conda_env): input_i64
                for input_i64 in selected_i64_files
            }
            for completed, future in enumerate(as_completed(future_to_input), start=1):
                result = future.result()
                lift_results.append(result)
                summary["lift_file_seconds_total"] += float(result["lift_seconds"])
                if result["success"]:
                    summary["lift_successful_files"] += 1
                    summary["functions_lifted_total"] += int(result["functions"])
                    print_stage_progress(
                        "lift",
                        completed,
                        len(selected_i64_files),
                        f"ok file={result['file']} functions={result['functions']} "
                        f"time={result['lift_seconds']:.3f}s",
                    )
                else:
                    summary["lift_failed_files"] += 1
                    failure = {
                        "stage": "lift",
                        "file": result["file"],
                        "returncode": result["returncode"],
                        "seconds": result["lift_seconds"],
                        "stderr": result["stderr"],
                        "stdout": result["stdout"],
                    }
                    summary["failures"].append(failure)
                    print_stage_progress(
                        "lift",
                        completed,
                        len(selected_i64_files),
                        f"failed file={result['file']} time={result['lift_seconds']:.3f}s",
                    )
                    if args.stop_on_error:
                        fatal_error = RuntimeError(json.dumps(failure, indent=2))
                        break
        summary["lift_wall_seconds_total"] = time.perf_counter() - lift_start

        if fatal_error is None:
            reopt_inputs = [
                {
                    "ll_path": Path(result["ll_path"]),
                    "bc_path": (bc_root / Path(result["file"])).with_suffix(".bc"),
                    "relative_path": str(result["file"]),
                    "functions": int(result["functions"]),
                }
                for result in lift_results
                if result["success"]
            ]
            summary["reopt_attempted_files"] = len(reopt_inputs)

            reopt_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_reopt = {
                    executor.submit(
                        run_reopt_task,
                        item["ll_path"],
                        item["bc_path"],
                        item["relative_path"],
                        item["functions"],
                    ): item
                    for item in reopt_inputs
                }
                for completed, future in enumerate(as_completed(future_to_reopt), start=1):
                    result = future.result()
                    summary["reopt_file_seconds_total"] += float(result["reopt_seconds"])
                    if result["success"]:
                        summary["reopt_successful_files"] += 1
                        summary["fully_successful_files"] += 1
                        summary["functions_reoptimized_total"] += int(result["functions"])
                        print_stage_progress(
                            "reopt",
                            completed,
                            len(reopt_inputs),
                            f"ok file={result['file']} functions={result['functions']} "
                            f"time={result['reopt_seconds']:.3f}s",
                        )
                    else:
                        summary["reopt_failed_files"] += 1
                        failure = {
                            "stage": "reopt",
                            "file": result["file"],
                            "returncode": result["returncode"],
                            "functions": result["functions"],
                            "seconds": result["reopt_seconds"],
                            "stderr": result["stderr"],
                            "stdout": result["stdout"],
                        }
                        summary["failures"].append(failure)
                        print_stage_progress(
                            "reopt",
                            completed,
                            len(reopt_inputs),
                            f"failed file={result['file']} functions={result['functions']} "
                            f"time={result['reopt_seconds']:.3f}s",
                        )
                        if args.stop_on_error:
                            fatal_error = RuntimeError(json.dumps(failure, indent=2))
                            break
            summary["reopt_wall_seconds_total"] = time.perf_counter() - reopt_start

        summary["combined_wall_seconds_total"] = (
            float(summary["lift_wall_seconds_total"]) + float(summary["reopt_wall_seconds_total"])
        )
        summary["lift_wall_seconds_per_100_functions"] = finalize_metric(
            float(summary["lift_wall_seconds_total"]),
            int(summary["functions_lifted_total"]),
        )
        summary["reopt_wall_seconds_per_100_functions"] = finalize_metric(
            float(summary["reopt_wall_seconds_total"]),
            int(summary["functions_reoptimized_total"]),
        )
        summary["combined_wall_seconds_per_100_functions"] = finalize_metric(
            float(summary["combined_wall_seconds_total"]),
            int(summary["functions_reoptimized_total"]),
        )
        summary["lift_file_seconds_per_100_functions"] = finalize_metric(
            float(summary["lift_file_seconds_total"]),
            int(summary["functions_lifted_total"]),
        )
        summary["reopt_file_seconds_per_100_functions"] = finalize_metric(
            float(summary["reopt_file_seconds_total"]),
            int(summary["functions_reoptimized_total"]),
        )

    finally:
        if not args.keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)
            summary["cleanup_performed"] = True

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

    if fatal_error is not None:
        raise fatal_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
