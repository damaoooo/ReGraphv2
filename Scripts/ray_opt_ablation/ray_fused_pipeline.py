#!/usr/bin/env python3
"""Ray driver for the fused .ll -> final_set pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
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


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def progress_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, max(0, int(width * completed / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def failure_label(failure: dict[str, Any]) -> str:
    item = failure.get("item", {})
    return (
        item.get("source_ll")
        or item.get("output_bc")
        or item.get("dataset_path")
        or item.get("bc_path")
        or "<unknown>"
    )


def compact_error(error: Any, limit: int = 1000) -> str:
    text = str(error or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


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
                label = failure_label(failure)
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


def stable_hash_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def discover_source_ll_files(dataset_path: Path, output_root: Path, opt_level: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    bc_root = output_root / "bc"
    for split, root in split_roots(dataset_path):
        for source_ll in root.rglob("*.ll"):
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
    return sorted(items, key=lambda item: stable_hash_key(item["relative_path"]))


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
    all_chunks = chunks(items, chunk_size)
    total_chunks = len(all_chunks)
    stage_start = time.time()
    summary_interval_s = max(1, int(context.get("progress_summary_interval_s", 30)))
    wait_timeout_s = min(5, summary_interval_s)
    next_summary = stage_start + summary_interval_s

    def emit_summary(reason: str, force: bool = False) -> None:
        nonlocal next_summary
        now = time.time()
        if not force and now < next_summary:
            return
        completed_items = len(successes) + len(failures) + len(skipped)
        completed_chunks = total_chunks - len(pending)
        elapsed = now - stage_start
        rate = completed_items / elapsed if elapsed > 0 else 0.0
        remaining = max(0, len(items) - completed_items)
        eta = remaining / rate if rate > 0 else None
        oldest = max((now - meta["start"] for meta in pending.values()), default=0.0)
        pct = (100.0 * completed_items / len(items)) if items else 100.0
        logger.info(
            f"stage={stage} progress {progress_bar(completed_items, len(items))} "
            f"items={completed_items}/{len(items)} pct={pct:.1f}% "
            f"success={len(successes)} failed={len(failures)} skipped={len(skipped)} "
            f"chunks={completed_chunks}/{total_chunks} pending={len(pending)} "
            f"rate_items_s={rate:.2f} eta={format_duration(eta)} "
            f"oldest_pending={format_duration(oldest)} reason={reason}"
        )
        logger.event(
            "stage_progress",
            stage=stage,
            reason=reason,
            completed_items=completed_items,
            total_items=len(items),
            success=len(successes),
            failed=len(failures),
            skipped=len(skipped),
            completed_chunks=completed_chunks,
            total_chunks=total_chunks,
            pending_chunks=len(pending),
            rate_items_s=round(rate, 3),
            eta_s=round(eta, 3) if eta is not None else None,
            oldest_pending_s=round(oldest, 3),
        )
        next_summary = now + summary_interval_s

    logger.info(
        f"stage={stage} start items={len(items)} chunk_size={chunk_size} "
        f"chunks={total_chunks} progress_summary_interval_s={summary_interval_s}"
    )
    for index, chunk in enumerate(all_chunks):
        chunk_id = f"{stage}-{index:06d}"
        ref = remote_func.remote(chunk_id, chunk, context)
        pending[ref] = {"chunk_id": chunk_id, "items": chunk, "start": time.time()}
        logger.event("chunk_start", stage=stage, chunk_id=chunk_id, items=len(chunk))

    while pending:
        ready, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=wait_timeout_s)
        if not ready:
            emit_summary("wait_timeout")
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
            emit_summary("chunk_complete")

    write_jsonl(output_root / "manifests" / f"{stage}_success.jsonl", successes)
    write_jsonl(output_root / "manifests" / f"{stage}_skipped.jsonl", skipped)
    write_jsonl(output_root / "manifests" / f"{stage}_failed.jsonl", failures)
    logger.write_failures(stage, failures)
    emit_summary("complete", force=True)
    logger.info(
        f"stage={stage} complete success={len(successes)} skipped={len(skipped)} failed={len(failures)}"
    )
    if failures:
        failure_path = logger.failure_dir / f"{stage}.txt"
        max_screen = max(0, int(context.get("max_failures_to_screen", 200)))
        logger.info(
            f"stage={stage} failures recorded count={len(failures)} failure_log={failure_path} "
            f"manifest={output_root / 'manifests' / f'{stage}_failed.jsonl'}"
        )
        for index, failure in enumerate(failures, start=1):
            screen = index <= max_screen
            logger.info(
                f"stage={stage} failure[{index}/{len(failures)}] "
                f"item={failure_label(failure)} error={compact_error(failure.get('error'))}",
                screen=screen,
            )
        if len(failures) > max_screen:
            logger.info(
                f"stage={stage} failure screen output capped at {max_screen}; "
                f"full list is in {failure_path} and run.log"
            )
    return successes, failures, skipped


def run_subprocess_stage(stage: str, command: list[str], repo_root: str, output_root: Path, logger: RunLogger, timeout: int = 0) -> None:
    logger.info(f"stage={stage} command={' '.join(command)}")
    log_path = output_root / "logs" / f"{stage}.log"
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    returncode = 0
    timed_out = False

    def write_child_line(handle: Any, raw_line: str) -> None:
        for line in raw_line.replace("\r", "\n").splitlines():
            if not line.strip():
                continue
            handle.write(line + "\n")
            logger.run_log.write(f"[{now_ts()}] stage={stage} output: {line}\n")
            logger.console.print(line)

    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while process.poll() is None:
            if timeout > 0 and time.time() - start > timeout:
                timed_out = True
                process.kill()
                break
            for key, _ in selector.select(timeout=1):
                line = key.fileobj.readline()
                if line:
                    write_child_line(handle, line)
        for line in process.stdout:
            write_child_line(handle, line)
        returncode = process.wait()
        selector.close()

    success = returncode == 0 and not timed_out
    if not success:
        if timed_out:
            logger.info(f"stage={stage} failed timeout={format_duration(timeout)} log={log_path}")
        else:
            logger.info(f"stage={stage} failed returncode={returncode} log={log_path}")
        raise RuntimeError(f"{stage} failed, see {log_path}")
    logger.info(f"stage={stage} complete log={log_path}")


def dataset_dir_complete(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def directory_file_stats(path: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    if not path.exists():
        return files, total_bytes
    for entry in path.rglob("*"):
        if entry.is_file():
            files += 1
            total_bytes += entry.stat().st_size
    return files, total_bytes


def copy_directory_atomic(source: Path, target: Path, logger: RunLogger, label: str) -> None:
    files, total_bytes = directory_file_stats(source)
    tmp_target = target.with_name(f".{target.name}.copying-{os.getpid()}")
    remove_path(tmp_target)
    ensure_dir(tmp_target)
    copied_files = 0
    copied_bytes = 0
    start = time.time()
    logger.info(
        f"stage={label} copy_start source={source} target={target} "
        f"files={files} bytes={total_bytes}"
    )
    try:
        for root, dirs, filenames in os.walk(source):
            root_path = Path(root)
            rel_root = root_path.relative_to(source)
            target_root = tmp_target / rel_root
            ensure_dir(target_root)
            for dirname in dirs:
                ensure_dir(target_root / dirname)
            for filename in filenames:
                src_file = root_path / filename
                dst_file = target_root / filename
                shutil.copy2(src_file, dst_file)
                copied_files += 1
                copied_bytes += src_file.stat().st_size
                logger.info(
                    f"stage={label} copy_progress files={copied_files}/{files} "
                    f"bytes={copied_bytes}/{total_bytes} current={src_file.relative_to(source)}"
                )
        remove_path(target)
        os.replace(tmp_target, target)
    except Exception:
        remove_path(tmp_target)
        raise
    elapsed = time.time() - start
    logger.info(
        f"stage={label} copy_complete target={target} files={copied_files} "
        f"bytes={copied_bytes} elapsed={format_duration(elapsed)}"
    )


def refresh_directory_symlink(link_path: Path, target: Path, logger: RunLogger) -> None:
    if same_path(link_path, target):
        return
    if link_path.is_symlink():
        existing_target = Path(os.readlink(link_path))
        if not existing_target.is_absolute():
            existing_target = (link_path.parent / existing_target).resolve()
        if same_path(existing_target, target):
            return
    remove_path(link_path)
    ensure_dir(link_path.parent)
    os.symlink(str(target), str(link_path), target_is_directory=True)
    logger.info(f"final_set_link link={link_path} target={target}")


def discover_hf_split_datasets(hf_root: Path) -> list[tuple[str, Path]]:
    results = []
    for split in SPLITS:
        path = hf_root / f"{split}_dataset"
        if dataset_dir_complete(path):
            results.append((split, path))
    if not results and dataset_dir_complete(hf_root):
        results.append(("dataset", hf_root))
    return results


def parquet_splits_complete(parquet_root: Path) -> list[str]:
    return [
        split
        for split in SPLITS
        if (parquet_root / split).is_dir() and any((parquet_root / split).glob("*.parquet"))
    ]


def hf_splits_complete(hf_root: Path, splits: Iterable[str]) -> bool:
    return all(dataset_dir_complete(hf_root / f"{split}_dataset") for split in splits)


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
    parser.add_argument("--task2-chunk-size", type=int, default=1)
    parser.add_argument("--task3-chunk-size", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-parquet-files", type=int, default=100000)
    parser.add_argument("--target-shard-size-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--command-timeout-seconds", type=int, default=0)
    parser.add_argument("--progress-summary-interval-s", type=int, default=30)
    parser.add_argument("--max-failures-to-screen", type=int, default=200)
    parser.add_argument("--task3-chunk-manifest-mode", choices=("worker", "chunk"), default="worker")
    parser.add_argument("--task3-ray-max-in-flight-chunks", type=int, default=0)
    parser.add_argument("--keep-task3-chunk-manifests", action="store_true")
    parser.add_argument(
        "--force-task3-rebuild",
        action="store_true",
        help="Delete and rebuild task3_fused from existing .bc files before DataProcess.",
    )
    parser.add_argument(
        "--final-output-root",
        default=os.environ.get("REGRAPH_FINAL_OUTPUT_ROOT", ""),
        help="Root for *_final_set outputs. Defaults to --output-path.",
    )
    parser.add_argument(
        "--final-local-root",
        default=os.environ.get("REGRAPH_FINAL_LOCAL_ROOT", ""),
        help=(
            "Optional node-local staging root for final split_train_validation outputs. "
            "When omitted and --final-output-root differs from --output-path, "
            "$TMPDIR/final_sets/<output-name> is used."
        ),
    )
    parser.add_argument("--keep-final-local", action="store_true")
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
    final_output_root = Path(args.final_output_root).resolve() if args.final_output_root else output_root
    if args.final_local_root:
        final_local_root: Path | None = Path(args.final_local_root).resolve()
    elif not same_path(final_output_root, output_root):
        final_local_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())).resolve() / "final_sets" / output_root.name
    else:
        final_local_root = None
    ensure_dir(final_output_root)

    console = Console()
    logger = RunLogger(output_root, console)
    try:
        logger.info(f"repo_root={repo_root}")
        logger.info(f"dataset_path={dataset_path}")
        logger.info(f"output_root={output_root}")
        logger.info(f"final_output_root={final_output_root}")
        logger.info(f"final_local_root={final_local_root if final_local_root else 'disabled'} keep_final_local={args.keep_final_local}")
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
            "progress_summary_interval_s": args.progress_summary_interval_s,
            "max_failures_to_screen": args.max_failures_to_screen,
        }

        source_items = discover_source_ll_files(dataset_path, output_root, opt_level)
        if not source_items:
            raise SystemExit(f"No source .ll files found under {dataset_path}")

        task2_chunk_size = choose_chunk_size(len(source_items), cpus, args.task2_chunk_size, 50)
        logger.info(f"task2_chunk_size={task2_chunk_size}")
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
            usable_bc_count = sum(1 for _ in (output_root / "bc").rglob("*.bc"))
            logger.info(
                f"stage=task2 continuing_after_failures failed={len(task2_fail)} "
                f"usable_bc_files={usable_bc_count}"
            )
            if usable_bc_count == 0:
                logger.info("pipeline finished because task2 produced no usable .bc files")
                return 1

        task3_output = output_root / "task3_fused"
        hf_root = output_root / "hf"
        existing_parquet_splits = parquet_splits_complete(task3_output / "parquet")
        task3_ran = False
        if args.force_task3_rebuild and task3_output.exists():
            logger.info(f"stage=task3_fused force_rebuild removing {task3_output}")
            shutil.rmtree(task3_output)
            existing_parquet_splits = []
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
            "--progress-summary-interval-s",
            str(args.progress_summary_interval_s),
            "--chunk-manifest-mode",
            args.task3_chunk_manifest_mode,
            "--ray-max-in-flight-chunks",
            str(args.task3_ray_max_in_flight_chunks),
        ]
        if args.keep_task3_chunk_manifests:
            task3_command.append("--keep-chunk-manifests")
        if args.task3_chunk_size > 0:
            task3_command.extend(["--chunk-size", str(args.task3_chunk_size)])
        if args.resume and not args.force_task3_rebuild:
            task3_command.append("--resume")
        if args.resume and not args.force_task3_rebuild and set(existing_parquet_splits) == set(SPLITS):
            logger.info(
                f"stage=task3_fused skipped existing final parquet splits={existing_parquet_splits}; "
                "use --force-task3-rebuild to regenerate"
            )
        else:
            run_subprocess_stage("task3_fused", task3_command, repo_root, output_root, logger)
            task3_ran = True

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
        parquet_splits = parquet_splits_complete(task3_output / "parquet")
        if args.resume and not task3_ran and parquet_splits and hf_splits_complete(hf_root, parquet_splits):
            logger.info(f"stage=dataprocess_hf skipped existing HF splits={parquet_splits}")
        else:
            run_subprocess_stage("dataprocess_hf", dataprocess_command, repo_root, output_root, logger)

        final_failures = []
        for split, dataset_dir in discover_hf_split_datasets(hf_root):
            stage_name = f"final_{split}"
            final_link_dir = output_root / f"{split}_final_set"
            final_target_dir = final_output_root / f"{split}_final_set"
            final_work_dir = (final_local_root / f"{split}_final_set") if final_local_root else final_target_dir
            final_input_dir = dataset_dir
            local_input_dir: Path | None = None
            if args.resume and dataset_dir_complete(final_target_dir / "train_dataset_pool"):
                logger.info(f"stage=final split={split} skipped existing {final_target_dir}")
                if not same_path(final_link_dir, final_target_dir):
                    refresh_directory_symlink(final_link_dir, final_target_dir, logger)
                continue
            if not same_path(final_work_dir, final_target_dir):
                remove_path(final_work_dir)
                if final_link_dir.exists() or final_link_dir.is_symlink():
                    remove_path(final_link_dir)
            elif final_work_dir.exists():
                remove_path(final_work_dir)
            if final_local_root:
                local_input_dir = final_local_root / "_hf_inputs" / f"{split}_dataset"
                remove_path(local_input_dir)
                copy_directory_atomic(dataset_dir, local_input_dir, logger, f"{stage_name}_input")
                final_input_dir = local_input_dir
            final_command = [
                sys.executable,
                "-m",
                "Pretrain.split_train_validation",
                str(final_input_dir),
                "--base-path",
                str(output_root / "bc" / split),
                "--train-ratio",
                "1.0",
                "--output-dir",
                str(final_work_dir),
            ]
            try:
                logger.info(
                    f"stage={stage_name} output_target={final_target_dir} "
                    f"work_dir={final_work_dir}"
                )
                run_subprocess_stage(stage_name, final_command, repo_root, output_root, logger, timeout=args.command_timeout_seconds)
                if not dataset_dir_complete(final_work_dir / "train_dataset_pool"):
                    raise RuntimeError(f"{stage_name} did not produce a complete train_dataset_pool at {final_work_dir}")
                if not same_path(final_work_dir, final_target_dir):
                    copy_directory_atomic(final_work_dir, final_target_dir, logger, stage_name)
                    if not args.keep_final_local:
                        remove_path(final_work_dir)
                        logger.info(f"stage={stage_name} removed local final work dir {final_work_dir}")
                if local_input_dir and not args.keep_final_local:
                    remove_path(local_input_dir)
                    logger.info(f"stage={stage_name} removed local HF input dir {local_input_dir}")
                if not same_path(final_link_dir, final_target_dir):
                    refresh_directory_symlink(final_link_dir, final_target_dir, logger)
            except Exception as exc:
                final_failures.append(
                    {
                        "split": split,
                        "error": str(exc),
                        "input_dir": str(final_input_dir),
                        "work_dir": str(final_work_dir),
                        "target_dir": str(final_target_dir),
                    }
                )

        write_jsonl(output_root / "manifests" / "final_failed.jsonl", final_failures)
        failed_splits = [
            split
            for split, _ in discover_hf_split_datasets(hf_root)
            if not dataset_dir_complete(final_output_root / f"{split}_final_set" / "train_dataset_pool")
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
