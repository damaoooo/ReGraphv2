#!/usr/bin/env python3
"""
Task 3: fused function extraction, graph building, tokenization, and parquet output.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq
import pyarrow as pa
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import console, ensure_directory  # noqa: E402
from DataProcess.dataset_features import get_dataset_features  # noqa: E402
from GraphBuilder.cfg_graph_builder import CFGGraphBuilder  # noqa: E402
from GraphBuilder.ddg_graph_builder import DataDependencyGraphBuilder  # noqa: E402
from Tokenizer.ir_tokenizer import load_tokenizer  # noqa: E402
from Tokenizer.normalizer import normalize_file  # noqa: E402
from Utils.utils import (  # noqa: E402
    DEFAULT_CFG_SO_PATH,
    DEFAULT_DDG_SO_PATH,
    DEFAULT_PURIFY_SO_PATH,
    DEFAULT_TOKENIZER_PATH,
)


app = typer.Typer(help="Fused Task 3 pipeline")

SPLIT_NAMES = ("train", "validation", "test")
STATE_DIRNAME = ".task3_fused_state"
DEFAULT_TARGET_SHARD_SIZE_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_PARQUET_FILES = 5000
DEFAULT_MAX_SEQ_LENGTH = 2048


@dataclass(frozen=True)
class WorkItem:
    split: Optional[str]
    bc_path: str
    binary_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "bc_path": self.bc_path,
            "binary_name": self.binary_name,
        }


def split_key(split: Optional[str]) -> str:
    return split if split else "__default__"


def split_label(split: Optional[str]) -> str:
    return split if split else "dataset"


def output_split_dir(root: Path, split: Optional[str]) -> Path:
    return root / split if split else root


def safe_chunk_id(split: Optional[str], index: int) -> str:
    return f"{split_label(split)}-{index:06d}"


def jsonl_append(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_directory(str(path.parent))
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def jsonl_read(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def manifest_dir(output_root: Path) -> Path:
    return output_root / "manifests"


def state_root(output_root: Path) -> Path:
    return output_root / STATE_DIRNAME


def raw_shard_root(output_root: Path) -> Path:
    return state_root(output_root) / "raw_shards"


def chunk_manifest_root(output_root: Path) -> Path:
    return state_root(output_root) / "chunk_manifests"


def final_parquet_root(output_root: Path) -> Path:
    return output_root / "parquet"


def load_completed_bc_paths(output_root: Path) -> set[str]:
    completed: set[str] = set()
    candidate_files = [
        manifest_dir(output_root) / "task3_success.jsonl",
        manifest_dir(output_root) / "task3_no_functions.jsonl",
    ]
    chunk_root = chunk_manifest_root(output_root)
    if chunk_root.exists():
        candidate_files.extend(chunk_root.glob("*_success.jsonl"))
        candidate_files.extend(chunk_root.glob("*_no_functions.jsonl"))

    for path in candidate_files:
        for row in jsonl_read(path):
            bc_path = row.get("bc_path")
            if bc_path:
                completed.add(os.path.abspath(bc_path))
    return completed


def prepare_output_root(output_root: Path, resume: bool) -> None:
    ensure_directory(str(output_root))
    if resume:
        ensure_directory(str(manifest_dir(output_root)))
        ensure_directory(str(state_root(output_root)))
        ensure_directory(str(final_parquet_root(output_root)))
        return

    for child in (manifest_dir(output_root), state_root(output_root), final_parquet_root(output_root)):
        if child.exists():
            shutil.rmtree(child)
    ensure_directory(str(manifest_dir(output_root)))
    ensure_directory(str(state_root(output_root)))
    ensure_directory(str(final_parquet_root(output_root)))


def discover_work_items(input_root: Path) -> List[WorkItem]:
    input_root = input_root.resolve()
    split_dirs = [name for name in SPLIT_NAMES if (input_root / name).is_dir()]
    items: List[WorkItem] = []

    if split_dirs:
        for split in split_dirs:
            split_root = input_root / split
            for bc_path in sorted(split_root.rglob("*.bc")):
                binary_name = os.path.splitext(os.path.relpath(bc_path, input_root))[0]
                items.append(
                    WorkItem(
                        split=split,
                        bc_path=str(bc_path.resolve()),
                        binary_name=binary_name,
                    )
                )
        return items

    for bc_path in sorted(input_root.rglob("*.bc")):
        binary_name = os.path.splitext(os.path.relpath(bc_path, input_root))[0]
        items.append(
            WorkItem(
                split=None,
                bc_path=str(bc_path.resolve()),
                binary_name=binary_name,
            )
        )
    return items


def group_by_split(items: List[WorkItem]) -> Dict[Optional[str], List[WorkItem]]:
    grouped: Dict[Optional[str], List[WorkItem]] = {}
    for item in items:
        grouped.setdefault(item.split, []).append(item)
    return grouped


def choose_chunk_size(total_items: int, requested_chunk_size: int, max_parquet_files: int) -> int:
    if total_items <= 0:
        return 1
    max_raw_shards = max(1, max_parquet_files // 2)
    if requested_chunk_size > 0:
        requested_chunks = math.ceil(total_items / requested_chunk_size)
        if requested_chunks > max_raw_shards:
            raise typer.BadParameter(
                f"--chunk-size={requested_chunk_size} would create {requested_chunks} raw shards, "
                f"above the default raw shard budget {max_raw_shards}. Increase chunk size or max parquet files."
            )
        return requested_chunk_size
    return max(1, math.ceil(total_items / max_raw_shards))


def chunk_items(items: List[WorkItem], chunk_size: int) -> List[List[WorkItem]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def run_command(command: List[str], cwd: Optional[Path] = None, timeout_seconds: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
    )


def stderr_excerpt(stderr: str, limit: int = 2000) -> str:
    stderr = stderr.strip()
    if len(stderr) <= limit:
        return stderr
    return stderr[:limit] + "...<truncated>"


def parse_quoted_path(line: str) -> Optional[str]:
    parts = line.split("'")
    if len(parts) >= 2:
        return parts[1]
    return None


def list_defined_functions(bc_path: Path, timeout_seconds: int) -> List[str]:
    result = run_command(["llvm-nm", str(bc_path)], timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(f"llvm-nm failed: {stderr_excerpt(result.stderr)}")

    functions = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "T":
            functions.append(parts[2])
    return functions


def extract_function_ir(bc_path: Path, function_name: str, output_path: Path, timeout_seconds: int) -> None:
    ensure_directory(str(output_path.parent))
    result = run_command(
        ["llvm-extract", "-S", f"--func={function_name}", str(bc_path), "-o", str(output_path)],
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(f"llvm-extract failed: {stderr_excerpt(result.stderr)}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("llvm-extract finished but did not create a non-empty function IR file")


def opt_initial_purify(llvm_ir_path: Path, timeout_seconds: int) -> Path:
    purified_file = llvm_ir_path.with_name(llvm_ir_path.stem + "_purified.ll")
    result = run_command(
        [
            "opt",
            "-S",
            "-load-pass-plugin=" + DEFAULT_PURIFY_SO_PATH,
            "--passes=strip-all-metadata",
            "-non-global-value-max-name-size=16384",
            llvm_ir_path.name,
            "-o",
            purified_file.name,
        ],
        cwd=llvm_ir_path.parent,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(f"metadata strip opt failed: {stderr_excerpt(result.stderr)}")
    if not purified_file.exists() or purified_file.stat().st_size == 0:
        raise RuntimeError("metadata strip opt finished but purified IR is missing")
    return purified_file


def opt_generate_ddg(purified_llvm_ir_path: Path, timeout_seconds: int) -> tuple[Path, Path]:
    instrumented_file = purified_llvm_ir_path.with_name(purified_llvm_ir_path.stem + "_instrumented.ll")
    result = run_command(
        [
            "opt",
            "-load-pass-plugin=" + DEFAULT_DDG_SO_PATH,
            "-passes=dot-id-graph",
            "-non-global-value-max-name-size=16384",
            purified_llvm_ir_path.name,
            "-S",
            "-o",
            instrumented_file.name,
        ],
        cwd=purified_llvm_ir_path.parent,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(f"DDG opt failed: {stderr_excerpt(result.stderr)}")

    dot_path: Optional[str] = None
    for line in result.stderr.splitlines():
        if "Writing ID-tagged graph to" in line:
            dot_path = parse_quoted_path(line)
            break
    if dot_path is None:
        raise RuntimeError(f"DDG opt did not report a dot file: {stderr_excerpt(result.stderr)}")

    ddg_dot = Path(dot_path)
    if not ddg_dot.is_absolute():
        ddg_dot = purified_llvm_ir_path.parent / ddg_dot
    if not instrumented_file.exists() or not ddg_dot.exists():
        raise RuntimeError("DDG opt finished but instrumented IR or dot file is missing")
    return instrumented_file, ddg_dot


def opt_generate_cfg(purified_llvm_ir_path: Path, timeout_seconds: int) -> Path:
    result = run_command(
        [
            "opt",
            "-load-pass-plugin=" + DEFAULT_CFG_SO_PATH,
            "-passes=dot-my-cfg",
            "-non-global-value-max-name-size=16384",
            purified_llvm_ir_path.name,
            "-o",
            os.devnull,
        ],
        cwd=purified_llvm_ir_path.parent,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CFG opt failed: {stderr_excerpt(result.stderr)}")

    dot_path: Optional[str] = None
    for line in result.stderr.splitlines():
        if "Write CFG to " in line:
            dot_path = parse_quoted_path(line)
            break
    if dot_path is None:
        raise RuntimeError(f"CFG opt did not report a dot file: {stderr_excerpt(result.stderr)}")

    cfg_dot = Path(dot_path)
    if not cfg_dot.is_absolute():
        cfg_dot = purified_llvm_ir_path.parent / cfg_dot
    if not cfg_dot.exists():
        raise RuntimeError("CFG opt finished but dot file is missing")
    return cfg_dot


def truncate_record(record: Dict[str, Any], eos_token_id: Optional[int], max_seq_length: int) -> Dict[str, Any]:
    input_ids = [int(token_id) for token_id in record["input_ids"]]
    if len(input_ids) > max_seq_length:
        eos = eos_token_id if eos_token_id is not None else input_ids[max_seq_length - 1]
        input_ids = input_ids[: max_seq_length - 1] + [int(eos)]
    record["input_ids"] = input_ids

    def keep_edge(edge: List[float]) -> bool:
        return bool(edge) and max(edge) < max_seq_length

    record["ddg_graph"] = [
        [int(value) for value in edge]
        for edge in record["ddg_graph"]
        if keep_edge(list(edge))
    ]
    record["cfg_graph"] = [
        [float(value) for value in edge]
        for edge in record["cfg_graph"]
        if keep_edge(list(edge))
    ]
    return record


def build_function_record(
    item: Dict[str, Any],
    function_name: str,
    function_index: int,
    tokenizer: Any,
    max_seq_length: int,
    timeout_seconds: int,
    temp_root: Path,
) -> Dict[str, Any]:
    bc_path = Path(item["bc_path"])
    function_hash = hashlib.sha1(f"{bc_path}::{function_name}".encode("utf-8")).hexdigest()
    function_dir = temp_root / f"{function_index:06d}_{function_hash[:16]}"
    ensure_directory(str(function_dir))

    function_ir = function_dir / "function.ll"
    extract_function_ir(bc_path, function_name, function_ir, timeout_seconds)

    purified_ir = opt_initial_purify(function_ir, timeout_seconds)
    instrumented_ir, ddg_dot = opt_generate_ddg(purified_ir, timeout_seconds)
    cfg_dot = opt_generate_cfg(purified_ir, timeout_seconds)

    ddg_builder = DataDependencyGraphBuilder(tokenizer)
    ddg_graph = ddg_builder.generate_ddg_matrix(str(ddg_dot), str(instrumented_ir), str(purified_ir))

    cfg_builder = CFGGraphBuilder(tokenizer, str(purified_ir), str(cfg_dot))
    cfg_graph = cfg_builder.build_cfg_edges()

    normalized_ir = normalize_file(str(purified_ir))
    tokens = tokenizer(normalized_ir)
    input_ids = tokens.get("input_ids")
    if not input_ids:
        raise RuntimeError("tokenizer returned empty input_ids")
    if ddg_graph is None:
        raise RuntimeError("DDG graph builder returned None")
    if cfg_graph is None:
        raise RuntimeError("CFG graph builder returned None")

    record = {
        "binary_name": item["binary_name"],
        "function_name": function_name,
        "file_path": f"{bc_path}::{function_name}",
        "input_ids": input_ids,
        "cfg_graph": [list(edge) for edge in cfg_graph],
        "ddg_graph": [list(edge) for edge in ddg_graph],
    }
    return truncate_record(record, tokenizer.eos_token_id, max_seq_length)


def write_records_to_parquet(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    ensure_directory(str(path.parent))
    import datasets

    dataset = datasets.Dataset.from_list(records, features=get_dataset_features())
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_parquet(str(tmp_path))
    os.replace(tmp_path, path)


def process_bc_chunk(chunk_id: str, items: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    repo_root = context["repo_root"]
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    tokenizer = load_tokenizer(context["tokenizer_path"])
    timeout_seconds = int(context["timeout_seconds"])
    max_seq_length = int(context["max_seq_length"])
    debug = bool(context["debug"])
    split = items[0].get("split") if items else None
    split_dir_name = split_key(split)
    raw_dir = Path(context["raw_shard_root"]) / split_dir_name
    chunk_manifest_dir = Path(context["chunk_manifest_root"])
    ensure_directory(str(raw_dir))
    ensure_directory(str(chunk_manifest_dir))

    if debug:
        chunk_temp_root = Path(context["debug_root"]) / chunk_id
        ensure_directory(str(chunk_temp_root))
        cleanup_temp = False
    else:
        temp_base = Path(context["temp_root"])
        ensure_directory(str(temp_base))
        chunk_temp_root = Path(tempfile.mkdtemp(prefix=f"{chunk_id}_", dir=str(temp_base)))
        cleanup_temp = True

    records: List[Dict[str, Any]] = []
    success_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    no_function_rows: List[Dict[str, Any]] = []

    try:
        for item in items:
            bc_path = item["bc_path"]
            try:
                functions = list_defined_functions(Path(bc_path), timeout_seconds)
            except Exception as exc:
                failed_rows.append(
                    {
                        **item,
                        "status": "failed",
                        "stage": "llvm-nm",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

            if not functions:
                no_function_rows.append(
                    {
                        **item,
                        "status": "no_functions",
                        "function_count": 0,
                    }
                )
                continue

            function_success = 0
            function_failed = 0
            for function_index, function_name in enumerate(functions):
                try:
                    record = build_function_record(
                        item=item,
                        function_name=function_name,
                        function_index=function_index,
                        tokenizer=tokenizer,
                        max_seq_length=max_seq_length,
                        timeout_seconds=timeout_seconds,
                        temp_root=chunk_temp_root / hashlib.sha1(bc_path.encode("utf-8")).hexdigest()[:16],
                    )
                    records.append(record)
                    function_success += 1
                except Exception as exc:
                    function_failed += 1
                    failed_rows.append(
                        {
                            **item,
                            "status": "function_failed",
                            "function_name": function_name,
                            "function_index": function_index,
                            "stage": "function",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

            bc_row = {
                **item,
                "function_count": len(functions),
                "function_success": function_success,
                "function_failed": function_failed,
            }
            if function_success > 0:
                success_rows.append({**bc_row, "status": "success"})
            else:
                failed_rows.append({**bc_row, "status": "bc_failed", "error": "no function records succeeded"})

        raw_shard_path = None
        if records:
            raw_shard_path = raw_dir / f"{chunk_id}.parquet"
            write_records_to_parquet(records, raw_shard_path)
            for row in success_rows:
                row["raw_shard"] = str(raw_shard_path)

        success_manifest = chunk_manifest_dir / f"{chunk_id}_success.jsonl"
        failed_manifest = chunk_manifest_dir / f"{chunk_id}_failed.jsonl"
        no_functions_manifest = chunk_manifest_dir / f"{chunk_id}_no_functions.jsonl"
        jsonl_append(success_manifest, success_rows)
        jsonl_append(failed_manifest, failed_rows)
        jsonl_append(no_functions_manifest, no_function_rows)

        return {
            "chunk_id": chunk_id,
            "split": split,
            "items": len(items),
            "records": len(records),
            "success_bc": len(success_rows),
            "failed_entries": len(failed_rows),
            "no_function_bc": len(no_function_rows),
            "raw_shard": str(raw_shard_path) if raw_shard_path else None,
            "success_manifest": str(success_manifest) if success_rows else None,
            "failed_manifest": str(failed_manifest) if failed_rows else None,
            "no_functions_manifest": str(no_functions_manifest) if no_function_rows else None,
            "elapsed_seconds": time.time() - start,
        }
    finally:
        if cleanup_temp:
            shutil.rmtree(chunk_temp_root, ignore_errors=True)


def collect_chunk_manifests(output_root: Path) -> None:
    chunk_root = chunk_manifest_root(output_root)
    if not chunk_root.exists():
        return

    manifest_map = {
        "success": manifest_dir(output_root) / "task3_success.jsonl",
        "failed": manifest_dir(output_root) / "task3_failed.jsonl",
        "no_functions": manifest_dir(output_root) / "task3_no_functions.jsonl",
    }
    for kind, destination in manifest_map.items():
        sources = sorted(chunk_root.glob(f"*_{kind}.jsonl"))
        if not sources:
            continue
        with destination.open("w", encoding="utf-8") as output_handle:
            for source in sources:
                with source.open("r", encoding="utf-8") as input_handle:
                    shutil.copyfileobj(input_handle, output_handle)


def compact_split_parquet(
    split: Optional[str],
    source_shards: List[Path],
    final_root: Path,
    target_shard_size_bytes: int,
    max_final_files: int,
) -> List[Path]:
    if not source_shards:
        return []
    if max_final_files <= 0:
        raise RuntimeError("No final parquet file budget remains after raw shard allocation")

    final_dir = output_split_dir(final_root, split)
    tmp_final_dir = final_dir.with_name(final_dir.name + ".tmp_compact")
    if tmp_final_dir.exists():
        shutil.rmtree(tmp_final_dir)
    ensure_directory(str(tmp_final_dir))

    final_paths: List[Path] = []
    writer: Optional[pq.ParquetWriter] = None
    current_estimated_bytes = 0
    current_path: Optional[Path] = None
    file_index = 0

    def close_current_writer() -> None:
        nonlocal writer, current_path, current_estimated_bytes
        if writer is None:
            return
        writer.close()
        writer = None
        if current_path is not None:
            final_path = current_path.with_suffix("")
            os.replace(current_path, final_path)
            final_paths.append(final_path)
        current_path = None
        current_estimated_bytes = 0

    try:
        for shard in source_shards:
            parquet_file = pq.ParquetFile(shard)
            if writer is None:
                current_path = tmp_final_dir / f"data_{file_index:05d}.parquet.tmp"
                writer = pq.ParquetWriter(str(current_path), parquet_file.schema_arrow, compression="snappy")

            for batch in parquet_file.iter_batches():
                writer.write_table(pa.Table.from_batches([batch]))

            current_estimated_bytes += shard.stat().st_size
            can_roll = len(final_paths) + 1 < max_final_files
            if current_estimated_bytes >= target_shard_size_bytes and can_roll:
                close_current_writer()
                file_index += 1

        close_current_writer()

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_final_dir, final_dir)
        return [final_dir / path.name for path in final_paths]
    except Exception:
        if writer is not None:
            writer.close()
        shutil.rmtree(tmp_final_dir, ignore_errors=True)
        raise


def existing_final_shards(final_root: Path, split: Optional[str]) -> List[Path]:
    directory = output_split_dir(final_root, split)
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.parquet") if path.is_file())


def raw_shards_for_split(output_root: Path, split: Optional[str]) -> List[Path]:
    directory = raw_shard_root(output_root) / split_key(split)
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.parquet") if path.is_file())


def compact_all_splits(
    output_root: Path,
    splits: Iterable[Optional[str]],
    target_shard_size_bytes: int,
    max_parquet_files: int,
) -> None:
    final_root = final_parquet_root(output_root)
    compaction_success = []
    compaction_failed = []

    split_list = list(splits)
    raw_count = sum(len(raw_shards_for_split(output_root, split)) for split in split_list)
    active_splits = [split for split in split_list if raw_shards_for_split(output_root, split)]
    remaining_final_budget = max_parquet_files - raw_count
    if active_splits and remaining_final_budget < len(active_splits):
        raise RuntimeError(
            f"Raw shard count {raw_count} leaves only {remaining_final_budget} final parquet slots "
            f"for {len(active_splits)} active split(s). Increase --max-parquet-files or chunk size."
        )

    for split in split_list:
        raw_shards = raw_shards_for_split(output_root, split)
        final_shards = existing_final_shards(final_root, split)
        if not raw_shards:
            if final_shards:
                console.print(f"[yellow]No new raw shards for split={split_label(split)}; keeping existing final parquet[/yellow]")
            continue

        source_shards = final_shards + raw_shards
        total_source_bytes = sum(path.stat().st_size for path in source_shards)
        active_index = active_splits.index(split)
        active_remaining = len(active_splits) - active_index
        split_budget = max(1, remaining_final_budget // active_remaining)
        adjusted_target_size = max(
            target_shard_size_bytes,
            math.ceil(total_source_bytes / split_budget) if total_source_bytes else target_shard_size_bytes,
        )

        try:
            final_paths = compact_split_parquet(
                split=split,
                source_shards=source_shards,
                final_root=final_root,
                target_shard_size_bytes=adjusted_target_size,
                max_final_files=split_budget,
            )
            for raw_shard in raw_shards:
                raw_shard.unlink(missing_ok=True)
            remaining_final_budget -= len(final_paths)
            compaction_success.append(
                {
                    "split": split,
                    "source_shards": len(source_shards),
                    "raw_shards": len(raw_shards),
                    "existing_final_shards": len(final_shards),
                    "final_shards": len(final_paths),
                    "target_shard_size_bytes": adjusted_target_size,
                    "final_paths": [str(path) for path in final_paths],
                }
            )
            console.print(
                f"[green]Compacted split={split_label(split)} raw={len(raw_shards)} "
                f"final={len(final_paths)}[/green]"
            )
        except Exception as exc:
            compaction_failed.append(
                {
                    "split": split,
                    "source_shards": [str(path) for path in source_shards],
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    jsonl_append(manifest_dir(output_root) / "compaction_success.jsonl", compaction_success)
    jsonl_append(manifest_dir(output_root) / "compaction_failed.jsonl", compaction_failed)
    if compaction_failed:
        raise RuntimeError(f"Compaction failed for {len(compaction_failed)} split(s)")


def run_local_backend(chunks: List[tuple[str, List[WorkItem]]], context: Dict[str, Any], workers: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing bc chunks", total=len(chunks))

        if workers <= 1:
            for chunk_id, chunk_items_list in chunks:
                result = process_bc_chunk(chunk_id, [item.to_dict() for item in chunk_items_list], context)
                results.append(result)
                progress.update(task, advance=1)
            return results

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_chunk = {
                executor.submit(
                    process_bc_chunk,
                    chunk_id,
                    [item.to_dict() for item in chunk_items_list],
                    context,
                ): chunk_id
                for chunk_id, chunk_items_list in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    error_row = {
                        "chunk_id": chunk_id,
                        "status": "chunk_failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    jsonl_append(manifest_dir(Path(context["output_root"])) / "task3_failed.jsonl", [error_row])
                    console.print(f"[red]Chunk failed: {chunk_id}: {exc}[/red]")
                finally:
                    progress.update(task, advance=1)
    return results


def run_ray_backend(chunks: List[tuple[str, List[WorkItem]]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Ray backend requested but ray is not installed in this environment") from exc

    address = os.environ.get("RAY_ADDRESS")
    if address:
        ray.init(address=address, log_to_driver=False)
    else:
        ray.init(log_to_driver=False)

    try:
        cluster_cpus = int(ray.cluster_resources().get("CPU", 0))
        console.print(f"[green]Ray cluster CPUs: {cluster_cpus}[/green]")

        @ray.remote
        def ray_process_bc_chunk(chunk_id: str, chunk_items_list: List[Dict[str, Any]], worker_context: Dict[str, Any]):
            return process_bc_chunk(chunk_id, chunk_items_list, worker_context)

        pending = [
            ray_process_bc_chunk.remote(chunk_id, [item.to_dict() for item in chunk_items_list], context)
            for chunk_id, chunk_items_list in chunks
        ]
        results: List[Dict[str, Any]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing bc chunks on Ray", total=len(pending))
            while pending:
                ready, pending = ray.wait(pending, num_returns=1, timeout=30)
                if not ready:
                    console.print(f"[yellow]Ray pending chunks: {len(pending)}[/yellow]")
                    continue
                for ref in ready:
                    try:
                        results.append(ray.get(ref))
                    except Exception as exc:
                        error_row = {
                            "status": "chunk_failed",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        jsonl_append(manifest_dir(Path(context["output_root"])) / "task3_failed.jsonl", [error_row])
                        console.print(f"[red]Ray chunk failed: {exc}[/red]")
                    finally:
                        progress.update(task, advance=1)
        return results
    finally:
        ray.shutdown()


@app.command()
def main(
    input_path: str = typer.Option(..., help="Input directory containing Task2 .bc files"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for parquet shards and manifests"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or ray"),
    workers: int = typer.Option(multiprocessing.cpu_count(), "--workers", "-j", help="Local backend worker count"),
    resume: bool = typer.Option(False, "--resume", help="Skip .bc files completed in previous manifests"),
    debug: bool = typer.Option(False, "--debug", help="Keep temporary IR/dot files under the debug directory"),
    tokenizer_path: str = typer.Option(DEFAULT_TOKENIZER_PATH, "--tokenizer-path", help="Tokenizer JSON path"),
    max_seq_length: int = typer.Option(DEFAULT_MAX_SEQ_LENGTH, "--max-seq-length", help="Truncate token ids and graph edges to this length"),
    target_shard_size_bytes: int = typer.Option(
        DEFAULT_TARGET_SHARD_SIZE_BYTES,
        "--target-shard-size-bytes",
        help="Target final parquet shard size; file cap may force larger shards",
    ),
    max_parquet_files: int = typer.Option(
        DEFAULT_MAX_PARQUET_FILES,
        "--max-parquet-files",
        help="Upper budget for raw plus final parquet files",
    ),
    chunk_size: int = typer.Option(0, "--chunk-size", help="Number of .bc files per chunk; 0 chooses from file budget"),
    command_timeout_seconds: int = typer.Option(0, "--command-timeout-seconds", help="Per LLVM/opt command timeout; 0 disables timeout"),
):
    """Run fused Task 3."""
    normalized_backend = backend.lower()
    if normalized_backend not in {"local", "ray"}:
        raise typer.BadParameter("--backend must be either local or ray")
    if max_seq_length <= 1:
        raise typer.BadParameter("--max-seq-length must be greater than 1")
    if max_parquet_files < 2:
        raise typer.BadParameter("--max-parquet-files must be at least 2")

    input_root = Path(input_path).resolve()
    output_root = Path(output).resolve()
    if not input_root.exists() or not input_root.is_dir():
        console.print(f"[red]Input directory not found: {input_root}[/red]")
        raise typer.Exit(code=1)
    if not Path(tokenizer_path).exists():
        console.print(f"[red]Tokenizer not found: {tokenizer_path}[/red]")
        raise typer.Exit(code=1)

    prepare_output_root(output_root, resume=resume)
    console.print(f"[green]Input: {input_root}[/green]")
    console.print(f"[green]Output: {output_root}[/green]")
    console.print(f"[green]Backend: {normalized_backend}[/green]")
    console.print(f"[green]Tokenizer: {tokenizer_path}[/green]")
    console.print(f"[green]Max seq length: {max_seq_length}[/green]")
    console.print(f"[green]Max parquet files: {max_parquet_files}[/green]")

    all_items = discover_work_items(input_root)
    all_splits = list(group_by_split(all_items).keys()) or [None]
    items = all_items
    console.print(f"[cyan]Discovered {len(items)} .bc files[/cyan]")
    if resume:
        completed = load_completed_bc_paths(output_root)
        before = len(items)
        items = [item for item in items if os.path.abspath(item.bc_path) not in completed]
        console.print(f"[yellow]Resume skipped {before - len(items)} completed .bc files[/yellow]")

    if not items:
        console.print("[yellow]No .bc files to process[/yellow]")
        compact_all_splits(
            output_root=output_root,
            splits=all_splits,
            target_shard_size_bytes=target_shard_size_bytes,
            max_parquet_files=max_parquet_files,
        )
        return

    grouped = group_by_split(items)
    chunks: List[tuple[str, List[WorkItem]]] = []
    for split, split_items in grouped.items():
        split_chunk_size = choose_chunk_size(len(split_items), chunk_size, max_parquet_files)
        console.print(
            f"[cyan]Split={split_label(split)} files={len(split_items)} chunk_size={split_chunk_size}[/cyan]"
        )
        for index, chunk in enumerate(chunk_items(split_items, split_chunk_size)):
            chunks.append((safe_chunk_id(split, index), chunk))

    if len(chunks) > max_parquet_files // 2:
        console.print(
            f"[red]Planned raw shard count {len(chunks)} exceeds raw shard budget {max_parquet_files // 2}[/red]"
        )
        raise typer.Exit(code=1)

    context = {
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "raw_shard_root": str(raw_shard_root(output_root)),
        "chunk_manifest_root": str(chunk_manifest_root(output_root)),
        "temp_root": str(state_root(output_root) / "tmp"),
        "debug_root": str(state_root(output_root) / "debug"),
        "tokenizer_path": tokenizer_path,
        "max_seq_length": max_seq_length,
        "timeout_seconds": command_timeout_seconds,
        "debug": debug,
    }

    run_started = time.time()
    if normalized_backend == "ray":
        results = run_ray_backend(chunks, context)
    else:
        results = run_local_backend(chunks, context, workers=workers)

    collect_chunk_manifests(output_root)
    raw_shard_count = sum(1 for _ in raw_shard_root(output_root).glob("**/*.parquet"))
    console.print(f"[green]Chunk processing complete: chunks={len(results)} raw_shards={raw_shard_count}[/green]")

    compact_all_splits(
        output_root=output_root,
        splits=grouped.keys(),
        target_shard_size_bytes=target_shard_size_bytes,
        max_parquet_files=max_parquet_files,
    )

    final_shard_count = sum(1 for _ in final_parquet_root(output_root).glob("**/*.parquet"))
    raw_shard_count_after = sum(1 for _ in raw_shard_root(output_root).glob("**/*.parquet"))
    console.print(
        f"[bold green]Task 3 completed in {time.time() - run_started:.2f}s. "
        f"temporary parquet count={raw_shard_count_after}, final parquet count={final_shard_count}, "
        f"total parquet count={raw_shard_count_after + final_shard_count}, max_parquet_files={max_parquet_files}[/bold green]"
    )


if __name__ == "__main__":
    app()
