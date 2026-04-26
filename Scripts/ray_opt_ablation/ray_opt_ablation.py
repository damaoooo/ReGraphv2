#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import ray
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


DEFAULT_REPO_ROOT = Path("/ibex/tmp/zhoul0e/regraphv2")
DEFAULT_DATASET_PATH = Path("/ibex/tmp/zhoul0e/Dataset-1")
DEFAULT_SMOKE_DATASET_PATH = Path("/ibex/tmp/zhoul0e/Dataset-smoketest")
DEFAULT_CACHE_ROOT = Path("/ibex/tmp/zhoul0e/regraph_cache")
SPLITS = ("train", "validation", "test")
GRAPH_TEMP_SUFFIXES = ("_purified.ll", "_instrumented.ll")


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
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(root / "tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = os.environ["TMPDIR"]
    root.mkdir(parents=True, exist_ok=True)
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
        "REGRAPH_SPLIT_LLVM_IR_SCRIPT", str(repo / "Scripts" / "split_llvm_ir.sh")
    )
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


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class RunLogger:
    def __init__(self, output_root: Path, console: Console):
        self.output_root = output_root
        self.log_dir = output_root / "logs"
        self.failure_dir = self.log_dir / "stage_failures"
        ensure_dir(self.failure_dir)
        self.console = console
        self.run_log = (self.log_dir / "run.log").open("a", encoding="utf-8", buffering=1)
        self.events = (self.log_dir / "events.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )

    def close(self) -> None:
        self.run_log.close()
        self.events.close()

    def info(self, message: str, screen: bool = True) -> None:
        line = f"[{now_ts()}] {message}"
        self.run_log.write(line + "\n")
        if screen:
            self.console.print(line)

    def event(self, event_type: str, **payload: Any) -> None:
        payload = {"ts": now_ts(), "event": event_type, **payload}
        self.events.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def write_failures(self, stage: str, failures: list[dict[str, Any]]) -> None:
        if not failures:
            return
        path = self.failure_dir / f"{stage}.txt"
        with path.open("a", encoding="utf-8") as handle:
            for failure in failures:
                item = failure.get("item", {})
                file_path = (
                    item.get("source_ll")
                    or item.get("output_bc")
                    or item.get("function_ll")
                    or item.get("parquet")
                    or item.get("dataset_path")
                    or "<unknown>"
                )
                handle.write(f"{file_path}\t{failure.get('error', '')}\n")


def split_counts_for(items: list[dict[str, Any]], status: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        split = item.get("split", "unknown")
        counts[split][status] += 1
    return {split: dict(values) for split, values in counts.items()}


def merge_split_counts(
    target: dict[str, dict[str, int]],
    source: dict[str, dict[str, int]],
) -> None:
    for split, values in source.items():
        for key, value in values.items():
            target.setdefault(split, {}).setdefault(key, 0)
            target[split][key] += value


def split_summary(counts: dict[str, dict[str, int]]) -> str:
    parts = []
    for split in SPLITS:
        values = counts.get(split, {})
        parts.append(
            f"{split}:s={values.get('success', 0)},"
            f"f={values.get('failed', 0)},k={values.get('skipped', 0)}"
        )
    return " | ".join(parts)


def item_label(item: dict[str, Any]) -> str:
    return (
        item.get("source_ll")
        or item.get("output_bc")
        or item.get("function_ll")
        or item.get("parquet")
        or item.get("dataset_path")
        or item.get("rel_path")
        or "<unknown>"
    )


def item_labels(items: list[dict[str, Any]], limit: int = 5) -> list[str]:
    labels = [item_label(item) for item in items[:limit]]
    if len(items) > limit:
        labels.append(f"...(+{len(items) - limit} more)")
    return labels


def pending_summary(pending: dict[Any, dict[str, Any]], limit: int = 5) -> str:
    if not pending:
        return "none"
    now = time.time()
    entries = sorted(
        pending.values(),
        key=lambda meta: meta.get("started_at", now),
    )[:limit]
    parts = []
    for meta in entries:
        age = int(now - meta.get("started_at", now))
        labels = ", ".join(item_labels(meta["chunk"], limit=2))
        parts.append(
            f"{meta['chunk_id']} age={age}s size={len(meta['chunk'])} items=[{labels}]"
        )
    if len(pending) > limit:
        parts.append(f"...(+{len(pending) - limit} chunks)")
    return " ; ".join(parts)


def chunk_items(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def choose_chunk_size(
    total: int,
    cluster_cpus: int,
    requested: int | None,
    upper_bound: int,
) -> int:
    if requested and requested > 0:
        return requested
    if total <= 0:
        return 1
    target_tasks = max(cluster_cpus * 4, 1)
    return max(1, min(upper_bound, math.ceil(total / target_tasks)))


def cluster_cpu_count() -> int:
    resources = ray.cluster_resources()
    return max(int(resources.get("CPU", 1)), 1)


def task2_marker(output_root: Path, output_bc: Path, opt_level: str) -> Path:
    rel_output = output_bc.relative_to(output_root)
    return (
        output_root
        / ".task2_reoptimize_state"
        / opt_state_token(opt_level)
        / f"{rel_output}.done"
    )


def discover_source_files(dataset_path: Path, output_root: Path, opt_level: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        split_root = dataset_path / split
        if not split_root.exists():
            continue
        for source in sorted(split_root.rglob("*.ll")):
            if source.name.endswith(GRAPH_TEMP_SUFFIXES):
                continue
            rel_path = source.relative_to(dataset_path)
            output_bc = (output_root / rel_path).with_suffix(".bc")
            records.append(
                {
                    "split": split,
                    "source_ll": str(source),
                    "rel_path": str(rel_path),
                    "output_bc": str(output_bc),
                    "marker": str(task2_marker(output_root, output_bc, opt_level)),
                }
            )
    return records


def task3_item_from_task2(item: dict[str, Any]) -> dict[str, Any]:
    output_bc = Path(item["output_bc"])
    return {
        "split": item["split"],
        "output_bc": item["output_bc"],
        "rel_path": item["rel_path"],
        "function_dir": str(output_bc.with_suffix("")) + "_functions",
    }


def discover_function_files(function_dirs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in function_dirs:
        function_dir = Path(item["function_dir"])
        if not function_dir.exists():
            continue
        for path in sorted(function_dir.glob("*.ll")):
            if path.name.endswith(GRAPH_TEMP_SUFFIXES):
                continue
            records.append(
                {
                    "split": item["split"],
                    "function_ll": str(path),
                    "function_dir": item["function_dir"],
                }
            )
    return records


def sqlite_uri(path: Path) -> str:
    return f"file:{path}?mode=ro"


def init_graph_db(db_path: Path) -> None:
    ensure_dir(db_path.parent)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_path TEXT UNIQUE,
                purify_path TEXT,
                instrumented_path TEXT,
                cfg_dot TEXT,
                ddg_dot TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_path ON results(input_path)")
        conn.commit()


def graph_db_processed(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    processed = set()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot FROM results")
        for input_path, purify_path, instrumented_path, cfg_dot, ddg_dot in cursor.fetchall():
            if all(path and Path(path).exists() for path in (purify_path, instrumented_path, cfg_dot, ddg_dot)):
                processed.add(input_path)
    return processed


def merge_graph_shards(output_root: Path, shard_paths: list[str]) -> None:
    rows_by_split: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for shard in shard_paths:
        for row in read_jsonl(Path(shard)):
            rows_by_split[row["split"]].append(
                (
                    row["input_path"],
                    row["purify_path"],
                    row["instrumented_path"],
                    row["cfg_dot"],
                    row["ddg_dot"],
                )
            )

    for split, rows in rows_by_split.items():
        db_path = output_root / split / "results.db"
        init_graph_db(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO results
                (input_path, purify_path, instrumented_path, cfg_dot, ddg_dot)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()


def graph_db_file_items(output_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for split in SPLITS:
        db_path = output_root / split / "results.db"
        if not db_path.exists():
            continue
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT input_path FROM results")
            for (input_path,) in cursor.fetchall():
                items.append({"split": split, "function_ll": input_path})
    return items


def dataset_dir_complete(path: Path) -> bool:
    return path.exists() and (path / "dataset_info.json").exists()


def prepare_output_root(output_root: Path, resume: bool, force_clean: bool) -> None:
    if output_root.exists() and force_clean:
        shutil.rmtree(output_root)
    if output_root.exists() and not resume:
        raise SystemExit(
            f"Output path already exists: {output_root}. Use --resume or --force-clean."
        )
    ensure_dir(output_root)


def stage_manifest_dir(output_root: Path) -> Path:
    return output_root / "manifests"


def write_stage_manifests(
    output_root: Path,
    stage: str,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    base = stage_manifest_dir(output_root)
    write_jsonl(base / f"{stage}_success.jsonl", successes)
    write_jsonl(base / f"{stage}_failed.jsonl", failures)
    write_jsonl(base / f"{stage}_skipped.jsonl", skipped)


def run_command(command: list[str], cwd: str | None = None, timeout: int = 3600) -> tuple[bool, str, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return False, exc.stdout or "", f"Timed out after {timeout}s"
    except Exception as exc:
        return False, "", repr(exc)


def build_one_cmake_project(source_dir: Path, build_dir: Path, expected_so: str) -> Path:
    ensure_dir(build_dir)
    configure = ["cmake", "-S", str(source_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"]
    success, stdout, stderr = run_command(configure, timeout=600)
    if not success:
        raise RuntimeError(f"cmake configure failed for {source_dir}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    build = ["cmake", "--build", str(build_dir), "--parallel", "2"]
    success, stdout, stderr = run_command(build, timeout=1200)
    if not success:
        raise RuntimeError(f"cmake build failed for {source_dir}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    candidates = list(build_dir.rglob(expected_so))
    if not candidates:
        raise RuntimeError(f"Expected {expected_so} not found under {build_dir}")
    return candidates[0]


def ensure_graph_plugins(repo_root: str) -> dict[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    base_dir = Path(
        os.environ.get(
            "REGRAPH_RAY_PLUGIN_DIR",
            os.path.join(tempfile.gettempdir(), f"regraph_ray_plugins_{job_id}"),
        )
    )
    ensure_dir(base_dir)
    lock_path = base_dir / "build.lock"
    repo = Path(repo_root)
    builds = {
        "ddg": (
            repo / "GraphBuilder" / "ddg_exporter",
            base_dir / "ddg_exporter",
            "libDDGPrinter.so",
        ),
        "cfg": (
            repo / "GraphBuilder" / "cfg_exporter",
            base_dir / "cfg_exporter",
            "libMyCFGPrinterPass.so",
        ),
        "purify": (
            repo / "GraphBuilder" / "meta_remover",
            base_dir / "meta_remover",
            "libStripAllMetadataPass.so",
        ),
    }

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        resolved = {}
        for key, (source_dir, build_dir, expected_so) in builds.items():
            existing = list(build_dir.rglob(expected_so)) if build_dir.exists() else []
            resolved[key] = str(existing[0] if existing else build_one_cmake_project(source_dir, build_dir, expected_so))
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    return resolved


@ray.remote
def task2_chunk(chunk_id: str, items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    configure_imports(context["repo_root"], context.get("cache_root"))
    from task2_reoptimize import reoptimize_file

    successes = []
    failures = []
    for item in items:
        try:
            output_bc = Path(item["output_bc"])
            ensure_dir(output_bc.parent)
            success, stdout, stderr = reoptimize_file(
                item["source_ll"],
                item["output_bc"],
                context["opt_level"],
                item["marker"],
            )
            if success and output_bc.exists() and output_bc.stat().st_size > 0:
                successes.append(item)
            else:
                failures.append({"item": item, "error": stderr or stdout or "task2 failed"})
        except Exception as exc:
            failures.append({"item": item, "error": f"{exc}\n{traceback.format_exc()}"})
    return chunk_result(chunk_id, "task2", items, successes, failures, start)


@ray.remote
def task3_chunk(chunk_id: str, items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    configure_imports(context["repo_root"], context.get("cache_root"))
    import task3_extract

    task3_extract.EXTRACT_SCRIPT = str(Path(context["repo_root"]) / "Scripts" / "split_llvm_ir.sh")
    successes = []
    failures = []
    for item in items:
        try:
            function_dir = Path(item["function_dir"])
            ensure_dir(function_dir)
            success, stdout, stderr = task3_extract.extract_functions(item["output_bc"], item["function_dir"])
            has_map = (function_dir / "function_map.csv").exists()
            has_functions = any(function_dir.glob("*.ll"))
            if success and has_map and has_functions:
                successes.append(item)
            else:
                failures.append(
                    {
                        "item": item,
                        "error": stderr or stdout or "task3 produced no function .ll files",
                    }
                )
        except Exception as exc:
            failures.append({"item": item, "error": f"{exc}\n{traceback.format_exc()}"})
    return chunk_result(chunk_id, "task3", items, successes, failures, start)


@ray.remote
def graph_chunk(chunk_id: str, items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    configure_imports(context["repo_root"], context.get("cache_root"))
    successes = []
    failures = []
    shard_rows = []
    shard_path = Path(context["output_root"]) / ".ray_state" / "graph_shards" / f"{chunk_id}.jsonl"
    try:
        plugins = ensure_graph_plugins(context["repo_root"])
        from GraphBuilder import graph_generator

        graph_generator.DEFAULT_DDG_SO_PATH = plugins["ddg"]
        graph_generator.DEFAULT_CFG_SO_PATH = plugins["cfg"]
        graph_generator.DEFAULT_PURIFY_SO_PATH = plugins["purify"]

        for item in items:
            try:
                result = graph_generator.process_one_file(item["function_ll"])
                if result is None:
                    failures.append({"item": item, "error": "GraphBuilder returned None"})
                    continue
                input_path, purify_path, instrumented_path, cfg_dot, ddg_dot = result
                row = {
                    "split": item["split"],
                    "input_path": input_path,
                    "purify_path": purify_path,
                    "instrumented_path": instrumented_path,
                    "cfg_dot": cfg_dot,
                    "ddg_dot": ddg_dot,
                }
                shard_rows.append(row)
                successes.append(item)
            except Exception as exc:
                failures.append({"item": item, "error": f"{exc}\n{traceback.format_exc()}"})
    except Exception as exc:
        failures.extend(
            {"item": item, "error": f"plugin/setup failure: {exc}\n{traceback.format_exc()}"}
            for item in items
        )

    if shard_rows:
        write_jsonl(shard_path, shard_rows)
    result = chunk_result(chunk_id, "graph", items, successes, failures, start)
    result["shards"] = [str(shard_path)] if shard_rows else []
    return result


@ray.remote
def dataprocess_chunk(chunk_id: str, items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    configure_imports(context["repo_root"], context.get("cache_root"))
    from DataProcess.dataset_features import get_dataset_features
    from DataProcess.parallel_processor import process_chunk_standalone
    import datasets

    successes = []
    failures = []
    shards = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_split[item["split"]].append(item)

    for split, split_items in by_split.items():
        db_path = Path(context["output_root"]) / split / "results.db"
        try:
            results = process_chunk_standalone(
                [item["function_ll"] for item in split_items],
                sqlite_uri(db_path),
                context["tokenizer_path"],
                True,
            )
            records = []
            success_paths = set()
            for result in results:
                if result.success:
                    records.append(result.to_dict())
                    success_paths.add(result.file_path)
            for item in split_items:
                if item["function_ll"] in success_paths:
                    successes.append(item)
                else:
                    failures.append({"item": item, "error": "DataProcess did not return success"})
            if records:
                shard_dir = Path(context["output_root"]) / ".ray_state" / "raw_shards" / split
                ensure_dir(shard_dir)
                shard_path = shard_dir / f"{chunk_id}.parquet"
                dataset = datasets.Dataset.from_list(records, features=get_dataset_features())
                dataset.to_parquet(str(shard_path))
                shards.append(str(shard_path))
        except Exception as exc:
            failures.extend(
                {"item": item, "error": f"{exc}\n{traceback.format_exc()}"}
                for item in split_items
            )

    result = chunk_result(chunk_id, "dataprocess", items, successes, failures, start)
    result["shards"] = shards
    return result


@ray.remote
def wash_shard_chunk(chunk_id: str, items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    configure_imports(context["repo_root"], context.get("cache_root"))
    from DataProcess.dataset_features import get_dataset_features
    from DataProcess.dataset_wash import has_required_graphs, truncate_example
    from Tokenizer.ir_tokenizer import load_tokenizer
    import datasets

    successes = []
    failures = []
    shards = []
    tokenizer = load_tokenizer(context["tokenizer_path"])
    eos_token_id = tokenizer.eos_token_id
    max_seq_length = context["max_seq_length"]

    for item in items:
        try:
            dataset = datasets.Dataset.from_parquet(
                item["parquet"],
                features=get_dataset_features(),
                cache_dir=os.environ["HF_DATASETS_CACHE"],
            )
            washed = dataset.filter(has_required_graphs)
            washed = washed.map(
                lambda example: truncate_example(example, eos_token_id, max_seq_length)
            )
            if len(washed) == 0:
                successes.append({**item, "empty": True})
                continue
            shard_dir = Path(context["output_root"]) / ".ray_state" / "wash_shards" / item["split"]
            ensure_dir(shard_dir)
            shard_path = shard_dir / f"{chunk_id}-{Path(item['parquet']).stem}.parquet"
            washed.to_parquet(str(shard_path))
            shards.append(str(shard_path))
            successes.append(item)
        except Exception as exc:
            failures.append({"item": item, "error": f"{exc}\n{traceback.format_exc()}"})
    result = chunk_result(chunk_id, "wash", items, successes, failures, start)
    result["shards"] = shards
    return result


@ray.remote
def final_split_task(chunk_id: str, items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    configure_imports(context["repo_root"], context.get("cache_root"))
    successes = []
    failures = []
    for item in items:
        command = [
            sys.executable,
            "-m",
            "Pretrain.split_train_validation",
            item["dataset_path"],
            "--base-path",
            item["base_path"],
            "--train-ratio",
            "1.0",
            "--output-dir",
            item["output_dir"],
        ]
        success, stdout, stderr = run_command(command, cwd=context["repo_root"], timeout=7200)
        if success:
            successes.append(item)
        else:
            failures.append({"item": item, "error": stderr or stdout or "split_train_validation failed"})
    return chunk_result(chunk_id, "final", items, successes, failures, start)


def chunk_result(
    chunk_id: str,
    stage: str,
    items: list[dict[str, Any]],
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    start: float,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "stage": stage,
        "host": socket.gethostname(),
        "elapsed_s": round(time.time() - start, 3),
        "processed": len(items),
        "successes": successes,
        "failures": failures,
        "split_counts": {
            "success": split_counts_for(successes, "success"),
            "failed": split_counts_for([failure["item"] for failure in failures], "failed"),
        },
    }


class StageRunner:
    def __init__(
        self,
        output_root: Path,
        progress: Progress,
        overall_task: TaskID,
        logger: RunLogger,
        summary_interval_s: int,
    ):
        self.output_root = output_root
        self.progress = progress
        self.overall_task = overall_task
        self.logger = logger
        self.summary_interval_s = summary_interval_s
        self.overall_total = 0

    def run(
        self,
        stage: str,
        items: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
        remote_func: Any,
        chunk_size: int,
        context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        total = len(items) + len(skipped)
        self.overall_total += total
        self.progress.update(self.overall_task, total=self.overall_total)

        split_counts: dict[str, dict[str, int]] = defaultdict(dict)
        merge_split_counts(split_counts, split_counts_for(skipped, "skipped"))

        task_id = self.progress.add_task(
            f"{stage} | {split_summary(split_counts)}",
            total=total,
        )
        if skipped:
            self.progress.update(task_id, advance=len(skipped))
            self.progress.update(self.overall_task, advance=len(skipped))

        self.logger.info(
            f"stage={stage} total={total} run={len(items)} skipped={len(skipped)} chunk_size={chunk_size}"
        )
        self.logger.event(
            "stage_start",
            stage=stage,
            total=total,
            run=len(items),
            skipped=len(skipped),
            chunk_size=chunk_size,
        )

        successes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        shards: list[str] = []
        pending: dict[Any, dict[str, Any]] = {}

        for index, chunk in enumerate(chunk_items(items, chunk_size)):
            chunk_id = f"{stage}-{index:06d}"
            ref = remote_func.remote(chunk_id, chunk, context)
            pending[ref] = {
                "chunk": chunk,
                "chunk_id": chunk_id,
                "retries": 0,
                "single": False,
                "started_at": time.time(),
            }
            self.logger.event(
                "chunk_start",
                stage=stage,
                chunk_id=chunk_id,
                items=len(chunk),
                item_preview=item_labels(chunk, limit=8),
            )

        last_summary = time.time()
        while pending:
            ready, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=1)
            if not ready:
                if time.time() - last_summary >= self.summary_interval_s:
                    self.logger.info(
                        f"stage={stage} pending_chunks={len(pending)} progress={self.progress.tasks[task_id].completed}/{total} {split_summary(split_counts)} pending={pending_summary(pending)}",
                        screen=False,
                    )
                    last_summary = time.time()
                continue

            for ref in ready:
                meta = pending.pop(ref)
                try:
                    result = ray.get(ref)
                except Exception as exc:
                    terminal_failures = self._handle_chunk_exception(
                        stage,
                        remote_func,
                        context,
                        pending,
                        meta,
                        exc,
                    )
                    if terminal_failures:
                        failures.extend(terminal_failures)
                        merge_split_counts(
                            split_counts,
                            split_counts_for(
                                [failure["item"] for failure in terminal_failures],
                                "failed",
                            ),
                        )
                        advanced = len(meta["chunk"])
                        self.progress.update(task_id, advance=advanced)
                        self.progress.update(self.overall_task, advance=advanced)
                        self.progress.update(
                            task_id,
                            description=f"{stage} | {split_summary(split_counts)}",
                        )
                    continue

                result_successes = result.get("successes", [])
                result_failures = result.get("failures", [])
                successes.extend(result_successes)
                failures.extend(result_failures)
                shards.extend(result.get("shards", []))
                merge_split_counts(split_counts, split_counts_for(result_successes, "success"))
                merge_split_counts(
                    split_counts,
                    split_counts_for([failure["item"] for failure in result_failures], "failed"),
                )

                advanced = result.get("processed", len(result_successes) + len(result_failures))
                self.progress.update(task_id, advance=advanced)
                self.progress.update(self.overall_task, advance=advanced)
                self.progress.update(task_id, description=f"{stage} | {split_summary(split_counts)}")
                self.logger.event(
                    "chunk_complete",
                    stage=stage,
                    chunk_id=result.get("chunk_id"),
                    host=result.get("host"),
                    elapsed_s=result.get("elapsed_s"),
                    processed=advanced,
                    successes=len(result_successes),
                    failures=len(result_failures),
                )

        if failures:
            self.logger.write_failures(stage, failures)
            preview = failures[:5]
            for failure in preview:
                path = item_label(failure.get("item", {}))
                self.logger.info(f"stage={stage} failure={path} error={failure.get('error', '')[:300]}")
            if len(failures) > len(preview):
                self.logger.info(f"stage={stage} additional_failures={len(failures) - len(preview)}")

        self.logger.info(
            f"stage={stage} complete success={len(successes)} failed={len(failures)} skipped={len(skipped)} shards={len(shards)}"
        )
        self.logger.event(
            "stage_complete",
            stage=stage,
            success=len(successes),
            failed=len(failures),
            skipped=len(skipped),
            shards=len(shards),
        )
        write_stage_manifests(self.output_root, stage, successes, failures, skipped)
        return successes, failures, skipped, shards

    def _handle_chunk_exception(
        self,
        stage: str,
        remote_func: Any,
        context: dict[str, Any],
        pending: dict[Any, dict[str, Any]],
        meta: dict[str, Any],
        exc: Exception,
    ) -> list[dict[str, Any]]:
        chunk = meta["chunk"]
        if meta["retries"] < 1:
            ref = remote_func.remote(meta["chunk_id"] + "-retry", chunk, context)
            pending[ref] = {
                **meta,
                "chunk_id": meta["chunk_id"] + "-retry",
                "retries": meta["retries"] + 1,
                "started_at": time.time(),
            }
            self.logger.event("chunk_retry", stage=stage, chunk_id=meta["chunk_id"], error=repr(exc))
            return []
        if len(chunk) > 1 and not meta.get("single"):
            self.logger.event("chunk_split_retry", stage=stage, chunk_id=meta["chunk_id"], size=len(chunk), error=repr(exc))
            for index, item in enumerate(chunk):
                chunk_id = f"{meta['chunk_id']}-single-{index}"
                ref = remote_func.remote(chunk_id, [item], context)
                pending[ref] = {
                    "chunk": [item],
                    "chunk_id": chunk_id,
                    "retries": 0,
                    "single": True,
                    "started_at": time.time(),
                }
            return []
        return [{"item": item, "error": f"Ray task exception: {repr(exc)}"} for item in chunk]


class SafeRateColumn(ProgressColumn):
    def render(self, task) -> Text:
        if task.speed is None:
            return Text("rate=?/s")
        return Text(f"rate={task.speed:.2f}/s")


def split_task2_items(items: list[dict[str, Any]], resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_items = []
    skipped = []
    for item in items:
        output_bc = Path(item["output_bc"])
        marker = Path(item["marker"])
        if resume and output_bc.exists() and output_bc.stat().st_size > 0 and marker.exists():
            skipped.append(item)
        else:
            run_items.append(item)
    return run_items, skipped


def split_task3_items(items: list[dict[str, Any]], resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_items = []
    skipped = []
    for item in items:
        function_dir = Path(item["function_dir"])
        if (
            resume
            and (function_dir / "function_map.csv").exists()
            and any(function_dir.glob("*.ll"))
        ):
            skipped.append(item)
        else:
            run_items.append(item)
    return run_items, skipped


def split_graph_items(items: list[dict[str, Any]], output_root: Path, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    processed_by_split = {
        split: graph_db_processed(output_root / split / "results.db") for split in SPLITS
    }
    run_items = []
    skipped = []
    for item in items:
        if resume and item["function_ll"] in processed_by_split.get(item["split"], set()):
            skipped.append(item)
        else:
            run_items.append(item)
    return run_items, skipped


def save_dataset_from_parquet(
    split: str,
    shards: list[str],
    output_path: Path,
    repo_root: str,
    cache_root: str | None,
) -> int:
    configure_imports(repo_root, cache_root)
    import datasets
    from DataProcess.dataset_features import get_dataset_features

    if not shards:
        return 0
    dataset = datasets.load_dataset(
        "parquet",
        data_files=sorted(shards),
        split="train",
        features=get_dataset_features(),
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    if output_path.exists():
        shutil.rmtree(output_path)
    dataset.save_to_disk(output_path)
    return len(dataset)


def collect_parquet_shards(root: Path) -> dict[str, list[str]]:
    by_split: dict[str, list[str]] = {}
    for split in SPLITS:
        split_dir = root / split
        if split_dir.exists():
            by_split[split] = [str(path) for path in sorted(split_dir.glob("*.parquet"))]
        else:
            by_split[split] = []
    return by_split


def run_final_materialization(
    stage: str,
    shard_root: Path,
    output_root: Path,
    suffix: str,
    repo_root: str,
    cache_root: str | None,
    resume: bool,
    logger: RunLogger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    successes = []
    failures = []
    skipped = []
    shards_by_split = collect_parquet_shards(shard_root)
    for split, shards in shards_by_split.items():
        output_path = output_root / f"{split}_{suffix}_dataset"
        item = {"split": split, "dataset_path": str(output_path), "shards": shards}
        if resume and dataset_dir_complete(output_path):
            skipped.append(item)
            continue
        try:
            count = save_dataset_from_parquet(split, shards, output_path, repo_root, cache_root)
            if count <= 0:
                failures.append({"item": item, "error": f"no rows for {split}"})
            else:
                successes.append({**item, "rows": count})
                logger.info(f"materialized {stage} split={split} rows={count} path={output_path}")
        except Exception as exc:
            failures.append({"item": item, "error": f"{exc}\n{traceback.format_exc()}"})
    write_stage_manifests(output_root, f"{stage}_materialize", successes, failures, skipped)
    if failures:
        logger.write_failures(f"{stage}_materialize", failures)
    return successes, failures, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray multinode opt ablation pipeline")
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
    parser.add_argument("--graph-chunk-size", type=int, default=0)
    parser.add_argument("--dataprocess-chunk-size", type=int, default=0)
    parser.add_argument("--wash-chunk-size", type=int, default=0)
    parser.add_argument("--progress-summary-interval-s", type=int, default=60)
    parser.add_argument("--max-seq-length", type=int, default=2048)
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

    output_root = Path(args.output_path) if args.output_path else dataset_path.with_name(f"{dataset_path.name}-{opt_state_token(opt_level)}")
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

        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)
        cpus = cluster_cpu_count()
        logger.info(f"ray_cluster_resources={ray.cluster_resources()}")
        logger.info(f"ray_cluster_cpus={cpus}")

        tokenizer_path = os.environ.get(
            "REGRAPH_TOKENIZER_PATH",
            str(Path(repo_root) / "Tokenizer" / "output_tokenizer" / "llvm_ir_bpe.json"),
        )
        context = {
            "repo_root": repo_root,
            "output_root": str(output_root),
            "cache_root": cache_root,
            "opt_level": opt_level,
            "tokenizer_path": tokenizer_path,
            "max_seq_length": args.max_seq_length,
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            SafeRateColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            overall_task = progress.add_task("overall", total=0)
            runner = StageRunner(
                output_root,
                progress,
                overall_task,
                logger,
                args.progress_summary_interval_s,
            )

            source_items = discover_source_files(dataset_path, output_root, opt_level)
            if not source_items:
                raise SystemExit(f"No source .ll files found under {dataset_path}")
            task2_run, task2_skip = split_task2_items(source_items, args.resume)
            task2_chunk_size = choose_chunk_size(
                len(task2_run), cpus, args.task2_chunk_size or None, 50
            )
            task2_success, task2_fail, task2_skipped, _ = runner.run(
                "task2",
                task2_run,
                task2_skip,
                task2_chunk,
                task2_chunk_size,
                context,
            )

            task2_ready = task2_success + task2_skipped
            task3_items = [task3_item_from_task2(item) for item in task2_ready]
            task3_run, task3_skip = split_task3_items(task3_items, args.resume)
            task3_chunk_size = choose_chunk_size(
                len(task3_run), cpus, args.task3_chunk_size or None, 50
            )
            task3_success, task3_fail, task3_skipped, _ = runner.run(
                "task3",
                task3_run,
                task3_skip,
                task3_chunk,
                task3_chunk_size,
                context,
            )

            function_items = discover_function_files(task3_success + task3_skipped)
            graph_run, graph_skip = split_graph_items(function_items, output_root, args.resume)
            graph_chunk_size = choose_chunk_size(
                len(graph_run), cpus, args.graph_chunk_size or None, 200
            )
            graph_success, graph_fail, graph_skipped, graph_shards = runner.run(
                "graph",
                graph_run,
                graph_skip,
                graph_chunk,
                graph_chunk_size,
                context,
            )
            merge_graph_shards(output_root, graph_shards)

            dataprocess_items = graph_db_file_items(output_root)
            raw_complete_splits = {
                split
                for split in SPLITS
                if args.resume and dataset_dir_complete(output_root / f"{split}_raw_dataset")
            }
            dp_run = [item for item in dataprocess_items if item["split"] not in raw_complete_splits]
            dp_skip = [item for item in dataprocess_items if item["split"] in raw_complete_splits]
            dp_chunk_size = choose_chunk_size(
                len(dp_run), cpus, args.dataprocess_chunk_size or None, 200
            )
            dp_success, dp_fail, dp_skipped, _ = runner.run(
                "dataprocess",
                dp_run,
                dp_skip,
                dataprocess_chunk,
                dp_chunk_size,
                context,
            )

            raw_success, raw_fail, raw_skip = run_final_materialization(
                "raw",
                output_root / ".ray_state" / "raw_shards",
                output_root,
                "raw",
                repo_root,
                cache_root,
                args.resume,
                logger,
            )

            raw_shards = []
            for split in SPLITS:
                shard_dir = output_root / ".ray_state" / "raw_shards" / split
                raw_shards.extend(
                    {"split": split, "parquet": str(path)}
                    for path in sorted(shard_dir.glob("*.parquet"))
                )
            wash_complete_splits = {
                split
                for split in SPLITS
                if args.resume and dataset_dir_complete(output_root / f"{split}_wash_dataset")
            }
            wash_run = [item for item in raw_shards if item["split"] not in wash_complete_splits]
            wash_skip = [item for item in raw_shards if item["split"] in wash_complete_splits]
            wash_chunk_size = choose_chunk_size(
                len(wash_run), cpus, args.wash_chunk_size or None, 100
            )
            wash_success, wash_fail, wash_skipped, _ = runner.run(
                "wash",
                wash_run,
                wash_skip,
                wash_shard_chunk,
                wash_chunk_size,
                context,
            )

            wash_mat_success, wash_mat_fail, wash_mat_skip = run_final_materialization(
                "wash",
                output_root / ".ray_state" / "wash_shards",
                output_root,
                "wash",
                repo_root,
                cache_root,
                args.resume,
                logger,
            )

            final_items = []
            final_skipped = []
            for split in SPLITS:
                wash_dataset = output_root / f"{split}_wash_dataset"
                final_dir = output_root / f"{split}_final_set"
                item = {
                    "split": split,
                    "dataset_path": str(wash_dataset),
                    "base_path": str(output_root / split),
                    "output_dir": str(final_dir),
                }
                if args.resume and dataset_dir_complete(final_dir / "train_dataset_pool"):
                    final_skipped.append(item)
                elif dataset_dir_complete(wash_dataset):
                    final_items.append(item)
                else:
                    logger.info(f"final skipped missing wash dataset split={split} path={wash_dataset}")

            final_success, final_fail, final_skip, _ = runner.run(
                "final",
                final_items,
                final_skipped,
                final_split_task,
                1,
                context,
            )

        all_failures = (
            task2_fail
            + task3_fail
            + graph_fail
            + dp_fail
            + raw_fail
            + wash_fail
            + wash_mat_fail
            + final_fail
        )
        failed_splits = [
            split
            for split in SPLITS
            if not dataset_dir_complete(output_root / f"{split}_final_set" / "train_dataset_pool")
        ]
        if failed_splits:
            logger.info(f"failed_splits={failed_splits}")
        if all_failures or failed_splits:
            logger.info(f"pipeline finished with failures total_failures={len(all_failures)}")
            return 1
        logger.info("pipeline completed successfully")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
