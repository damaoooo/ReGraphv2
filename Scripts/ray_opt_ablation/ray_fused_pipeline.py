#!/usr/bin/env python3
"""Ray driver for the fused .ll -> final_set pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import ray
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


DEFAULT_REPO_ROOT = Path("/scratch/zhoul0e/ReGraphv2")
DEFAULT_DATASET_PATH = Path("/scratch/zhoul0e/Dataset-1")
DEFAULT_SMOKE_DATASET_PATH = Path("/scratch/zhoul0e/Dataset-smoketest")
DEFAULT_CACHE_ROOT = Path("/scratch/zhoul0e/regraph_cache")
SPLITS = ("train", "validation", "test")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def configure_cache(cache_root: str | Path | None = None) -> dict[str, str]:
    root = Path(cache_root or os.environ.get("REGRAPH_CACHE_ROOT", DEFAULT_CACHE_ROOT)).expanduser().resolve()
    os.environ["REGRAPH_CACHE_ROOT"] = str(root)
    cache_env = {
        "HF_HOME": root / "huggingface",
        "HF_DATASETS_CACHE": root / "huggingface" / "datasets",
        "HF_HUB_CACHE": root / "huggingface" / "hub",
        "HF_ASSETS_CACHE": root / "huggingface" / "assets",
        "TRANSFORMERS_CACHE": root / "huggingface" / "transformers",
        "XDG_CACHE_HOME": root / "xdg",
    }
    for name, path in cache_env.items():
        os.environ[name] = str(path)
        ensure_dir(path)
    os.environ["TMPDIR"] = str(root / "tmp")
    ensure_dir(Path(os.environ["TMPDIR"]))
    tempfile.tempdir = os.environ["TMPDIR"]
    ensure_dir(root)
    return {
        "REGRAPH_CACHE_ROOT": str(root),
        "TMPDIR": os.environ["TMPDIR"],
        **{name: os.environ[name] for name in cache_env},
    }


def configure_imports(repo_root: str, cache_root: str | None = None) -> None:
    repo = Path(repo_root)
    configure_cache(cache_root)
    for path in (repo, repo / "Scripts"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ.setdefault("REGRAPH_REPO_ROOT", str(repo))
    os.environ.setdefault(
        "REGRAPH_TOKENIZER_PATH",
        str(repo / "Tokenizer" / "output_tokenizer" / "llvm_ir_bpe.json"),
    )


def normalize_opt_level(repo_root: str, opt_level: str) -> str:
    configure_imports(repo_root)
    from utils import normalize_clang_opt_level

    return normalize_clang_opt_level(opt_level)


def opt_state_token(opt_level: str) -> str:
    return opt_level.lstrip("-").replace(os.sep, "_")


def run_command(
    command: list[str],
    cwd: str | Path | None = None,
    timeout: int = 0,
    env: dict[str, str] | None = None,
) -> tuple[bool, str, str, int]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout if timeout > 0 else None,
        )
        return result.returncode == 0, result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        return False, stdout, stderr + f"\nTimeout after {timeout}s", -1


class RunLogger:
    def __init__(self, output_root: Path, console: Console):
        self.output_root = output_root
        self.console = console
        self.log_dir = output_root / "logs"
        self.failure_dir = self.log_dir / "stage_failures"
        ensure_dir(self.failure_dir)
        self.run_log = (self.log_dir / "run.log").open("a", encoding="utf-8", buffering=1)
        self.events = (self.log_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        self.run_log.close()
        self.events.close()

    def info(self, message: str, screen: bool = True) -> None:
        line = f"[{now_ts()}] {message}"
        self.run_log.write(line + "\n")
        if screen:
            self.console.print(line)

    def event(self, event: str, **payload: Any) -> None:
        payload = {"ts": now_ts(), "event": event, **payload}
        self.events.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")

    def write_failures(self, stage: str, failures: list[dict[str, Any]]) -> None:
        if not failures:
            return
        path = self.failure_dir / f"{stage}.txt"
        with path.open("a", encoding="utf-8") as handle:
            for failure in failures:
                item = failure.get("item", {})
                label = item.get("source_ll") or item.get("output_bc") or item.get("dataset_path") or "<unknown>"
                handle.write(f"{label}\t{failure.get('error', '')}\n")


def prepare_output_root(output_root: Path, resume: bool, force_clean: bool) -> None:
    if force_clean and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)
    ensure_dir(output_root / "manifests")
    ensure_dir(output_root / ".ray_fused_state")


def split_roots(dataset_path: Path) -> list[tuple[str | None, Path]]:
    roots = [(split, dataset_path / split) for split in SPLITS if (dataset_path / split).is_dir()]
    if roots:
        return roots
    return [(None, dataset_path)]


def discover_source_ll_files(dataset_path: Path, output_root: Path, opt_level: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    bc_root = output_root / "bc"
    for split, root in split_roots(dataset_path):
        for source_ll in sorted(root.rglob("*.ll")):
            name = source_ll.name
            if name.endswith("_purified.ll") or name.endswith("_instrumented.ll"):
                continue
            if any(part.endswith("_functions") for part in source_ll.parts):
                continue
            relative = source_ll.relative_to(dataset_path)
            output_bc = (bc_root / relative).with_suffix(".bc")
            marker = output_root / ".ray_fused_state" / "task2_done" / relative.with_suffix(".bc.done")
            items.append(
                {
                    "split": split,
                    "source_ll": str(source_ll.resolve()),
                    "relative_path": str(relative),
                    "output_bc": str(output_bc.resolve()),
                    "marker": str(marker.resolve()),
                    "opt_level": opt_level,
                }
            )
    return items


def choose_chunk_size(total: int, cpus: int, override: int, default_min: int) -> int:
    if total <= 0:
        return 1
    if override > 0:
        return override
    target_chunks = max(cpus * 4, 1)
    return max(default_min, math.ceil(total / target_chunks))


def chunks(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


@ray.remote
def task2_chunk(chunk_id: str, items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    configure_imports(context["repo_root"], context.get("cache_root"))
    successes = []
    skipped = []
    failures = []
    timeout = int(context.get("command_timeout_seconds", 0))
    resume = bool(context.get("resume"))

    for item in items:
        output_bc = Path(item["output_bc"])
        marker = Path(item["marker"])
        if resume and output_bc.exists() and output_bc.stat().st_size > 0 and marker.exists():
            skipped.append(item)
            continue

        ensure_dir(output_bc.parent)
        ensure_dir(marker.parent)
        if marker.exists():
            marker.unlink()

        command = [
            "clang",
            "-m32",
            item["opt_level"],
            "-c",
            "-emit-llvm",
            "-fno-inline",
            "-fno-inline-functions",
            item["source_ll"],
            "-o",
            str(output_bc),
        ]
        success, stdout, stderr, returncode = run_command(command, timeout=timeout)
        if success and output_bc.exists() and output_bc.stat().st_size > 0:
            marker.write_text(json.dumps({"source_ll": item["source_ll"], "opt_level": item["opt_level"]}) + "\n")
            successes.append(item)
        else:
            if output_bc.exists():
                output_bc.unlink(missing_ok=True)
            failures.append(
                {
                    "item": item,
                    "error": (stderr or stdout or f"clang failed returncode={returncode}")[:4000],
                }
            )

    return {
        "chunk_id": chunk_id,
        "stage": "task2",
        "processed": len(items),
        "successes": successes,
        "skipped": skipped,
        "failures": failures,
        "elapsed_s": round(time.time() - start, 3),
    }


def run_ray_stage(
    stage: str,
    items: list[dict[str, Any]],
    remote_func: Any,
    chunk_size: int,
    context: dict[str, Any],
    output_root: Path,
    logger: RunLogger,
    progress: Progress,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    task_id = progress.add_task(stage, total=len(items))
    pending: dict[Any, dict[str, Any]] = {}

    for index, chunk in enumerate(chunks(items, chunk_size)):
        chunk_id = f"{stage}-{index:06d}"
        ref = remote_func.remote(chunk_id, chunk, context)
        pending[ref] = {"chunk_id": chunk_id, "items": chunk, "start": time.time()}
        logger.event("chunk_start", stage=stage, chunk_id=chunk_id, items=len(chunk))

    while pending:
        ready, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=30)
        if not ready:
            oldest = max((time.time() - meta["start"] for meta in pending.values()), default=0)
            logger.info(f"stage={stage} pending_chunks={len(pending)} oldest_age_s={oldest:.0f}")
            continue
        for ref in ready:
            meta = pending.pop(ref)
            try:
                result = ray.get(ref)
                result_successes = result.get("successes", [])
                result_failures = result.get("failures", [])
                result_skipped = result.get("skipped", [])
                successes.extend(result_successes)
                failures.extend(result_failures)
                skipped.extend(result_skipped)
                progress.update(task_id, advance=result.get("processed", len(meta["items"])))
                logger.event(
                    "chunk_complete",
                    stage=stage,
                    chunk_id=result.get("chunk_id"),
                    processed=result.get("processed"),
                    success=len(result_successes),
                    skipped=len(result_skipped),
                    failed=len(result_failures),
                    elapsed_s=result.get("elapsed_s"),
                )
            except Exception as exc:
                chunk_failures = [
                    {"item": item, "error": f"Ray task exception: {repr(exc)}\n{traceback.format_exc()}"}
                    for item in meta["items"]
                ]
                failures.extend(chunk_failures)
                progress.update(task_id, advance=len(meta["items"]))
                logger.event("chunk_exception", stage=stage, chunk_id=meta["chunk_id"], error=repr(exc))

    write_jsonl(output_root / "manifests" / f"{stage}_success.jsonl", successes)
    write_jsonl(output_root / "manifests" / f"{stage}_skipped.jsonl", skipped)
    write_jsonl(output_root / "manifests" / f"{stage}_failed.jsonl", failures)
    logger.write_failures(stage, failures)
    logger.info(
        f"stage={stage} complete success={len(successes)} skipped={len(skipped)} failed={len(failures)}"
    )
    return successes, failures, skipped


def run_subprocess_stage(stage: str, command: list[str], repo_root: str, output_root: Path, logger: RunLogger, timeout: int = 0) -> None:
    logger.info(f"stage={stage} command={' '.join(command)}")
    success, stdout, stderr, returncode = run_command(command, cwd=repo_root, timeout=timeout, env=os.environ.copy())
    log_path = output_root / "logs" / f"{stage}.log"
    ensure_dir(log_path.parent)
    log_path.write_text((stdout or "") + ("\n--- stderr ---\n" + stderr if stderr else ""), encoding="utf-8")
    if not success:
        logger.info(f"stage={stage} failed returncode={returncode} log={log_path}")
        raise RuntimeError(f"{stage} failed, see {log_path}")
    logger.info(f"stage={stage} complete log={log_path}")


def dataset_dir_complete(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def discover_hf_split_datasets(hf_root: Path) -> list[tuple[str, Path]]:
    results = []
    for split in SPLITS:
        path = hf_root / f"{split}_dataset"
        if dataset_dir_complete(path):
            results.append((split, path))
    if not results and dataset_dir_complete(hf_root):
        results.append(("dataset", hf_root))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray fused .ll to final_set pipeline")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-path", default="")
    parser.add_argument("--cache-root", default=os.environ.get("REGRAPH_CACHE_ROOT", str(DEFAULT_CACHE_ROOT)))
    parser.add_argument("--opt-level", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task2-chunk-size", type=int, default=0)
    parser.add_argument("--task3-chunk-size", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-parquet-files", type=int, default=5000)
    parser.add_argument("--target-shard-size-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--command-timeout-seconds", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = str(Path(args.repo_root).resolve())
    cache_root = str(Path(args.cache_root).expanduser().resolve())
    cache_env = configure_cache(cache_root)
    configure_imports(repo_root, cache_root)
    opt_level = normalize_opt_level(repo_root, args.opt_level)

    dataset_path = Path(args.dataset_path)
    if args.smoke:
        dataset_path = DEFAULT_SMOKE_DATASET_PATH
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset_path}")

    output_root = Path(args.output_path) if args.output_path else dataset_path.with_name(f"{dataset_path.name}-{opt_state_token(opt_level)}-fused")
    output_root = output_root.resolve()
    prepare_output_root(output_root, args.resume, args.force_clean)

    console = Console()
    logger = RunLogger(output_root, console)
    try:
        logger.info(f"repo_root={repo_root}")
        logger.info(f"dataset_path={dataset_path}")
        logger.info(f"output_root={output_root}")
        logger.info(f"opt_level={opt_level}")
        for name in sorted(cache_env):
            logger.info(f"{name}={cache_env[name]}")

        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            ray.init(address=ray_address, log_to_driver=False)
        else:
            ray.init(log_to_driver=False)
        cpus = int(ray.cluster_resources().get("CPU", 1))
        logger.info(f"ray_cluster_resources={ray.cluster_resources()}")
        logger.info(f"ray_cluster_cpus={cpus}")

        context = {
            "repo_root": repo_root,
            "cache_root": cache_root,
            "resume": args.resume,
            "command_timeout_seconds": args.command_timeout_seconds,
        }

        source_items = discover_source_ll_files(dataset_path, output_root, opt_level)
        if not source_items:
            raise SystemExit(f"No source .ll files found under {dataset_path}")

        task2_chunk_size = choose_chunk_size(len(source_items), cpus, args.task2_chunk_size, 50)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task2_success, task2_fail, task2_skip = run_ray_stage(
                "task2",
                source_items,
                task2_chunk,
                task2_chunk_size,
                context,
                output_root,
                logger,
                progress,
            )

        if task2_fail:
            logger.info("pipeline finished with task2 failures")
            return 1

        task3_output = output_root / "task3_fused"
        task3_command = [
            sys.executable,
            str(Path(repo_root) / "Scripts" / "task3_extract.py"),
            "--input-path",
            str(output_root / "bc"),
            "--output",
            str(task3_output),
            "--backend",
            "ray",
            "--max-seq-length",
            str(args.max_seq_length),
            "--max-parquet-files",
            str(args.max_parquet_files),
            "--target-shard-size-bytes",
            str(args.target_shard_size_bytes),
            "--command-timeout-seconds",
            str(args.command_timeout_seconds),
        ]
        if args.task3_chunk_size > 0:
            task3_command.extend(["--chunk-size", str(args.task3_chunk_size)])
        if args.resume:
            task3_command.append("--resume")
        run_subprocess_stage("task3_fused", task3_command, repo_root, output_root, logger)

        hf_root = output_root / "hf"
        dataprocess_command = [
            sys.executable,
            "-m",
            "DataProcess.cli",
            "parquet",
            "--input-parquet-dir",
            str(task3_output / "parquet"),
            "--output-dir",
            str(hf_root),
            "--cache-dir",
            os.environ["HF_DATASETS_CACHE"],
        ]
        run_subprocess_stage("dataprocess_hf", dataprocess_command, repo_root, output_root, logger)

        final_failures = []
        for split, dataset_dir in discover_hf_split_datasets(hf_root):
            final_dir = output_root / f"{split}_final_set"
            if args.resume and dataset_dir_complete(final_dir / "train_dataset_pool"):
                logger.info(f"stage=final split={split} skipped existing {final_dir}")
                continue
            final_command = [
                sys.executable,
                "-m",
                "Pretrain.split_train_validation",
                str(dataset_dir),
                "--base-path",
                str(output_root / "bc" / split),
                "--train-ratio",
                "1.0",
                "--output-dir",
                str(final_dir),
            ]
            try:
                run_subprocess_stage(f"final_{split}", final_command, repo_root, output_root, logger, timeout=7200)
            except Exception as exc:
                final_failures.append({"split": split, "error": str(exc)})

        write_jsonl(output_root / "manifests" / "final_failed.jsonl", final_failures)
        failed_splits = [
            split
            for split, _ in discover_hf_split_datasets(hf_root)
            if not dataset_dir_complete(output_root / f"{split}_final_set" / "train_dataset_pool")
        ]
        if final_failures or failed_splits:
            logger.info(f"pipeline finished with failures final_failures={len(final_failures)} failed_splits={failed_splits}")
            return 1

        logger.info("pipeline completed successfully")
        return 0
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
