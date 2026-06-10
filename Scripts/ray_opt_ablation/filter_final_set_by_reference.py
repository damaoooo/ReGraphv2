#!/usr/bin/env python3
"""Filter a ReLL final_set by a CSV or another final_set reference."""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import datasets
import typer
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
from rich.table import Table


app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="Filter a final_set by a Dataset-1 CSV or another final_set reference.",
)
console = Console()

SPLIT_CSV_FILENAMES = {
    "train": ("training_Dataset-1.csv", "train_Dataset-1.csv"),
    "training": ("training_Dataset-1.csv", "train_Dataset-1.csv"),
    "validation": ("validation_Dataset-1.csv",),
    "test": ("testing_Dataset-1.csv", "test_Dataset-1.csv"),
    "testing": ("testing_Dataset-1.csv", "test_Dataset-1.csv"),
}
SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "validation": "validation",
    "testing": "test",
    "test": "test",
}
OPT_LEVEL_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<opt>O0|O1|O2|O3|Os|Oz)_(?P<binary>.+)$")

_WORKER_ALLOWED: set[str] | None = None


class ReferenceKind(str, Enum):
    auto = "auto"
    csv = "csv"
    final_set = "final-set"


class MatchMode(str, Enum):
    exact = "exact"
    origin = "origin"


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SPLIT_ALIASES:
        raise ValueError(f"Unsupported split: {split}. Expected train, validation, or test.")
    return SPLIT_ALIASES[normalized]


def infer_split(path: Path) -> str | None:
    for part in [path.name, path.parent.name]:
        lower = part.lower()
        for token in ("train", "training", "validation", "test", "testing"):
            if token in lower:
                return normalize_split(token)
    return None


def resolve_input_pool(input_path: Path) -> tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()
    if (input_path / "train_dataset_pool").is_dir():
        return input_path, input_path / "train_dataset_pool"
    if (input_path / "dataset_info.json").is_file() and input_path.name == "train_dataset_pool":
        return input_path.parent, input_path
    raise ValueError(
        "Input must be a final_set directory containing train_dataset_pool, "
        "or the train_dataset_pool directory itself."
    )


def csv_file_for_split(csv_filter_dir: Path, split: str) -> Path:
    candidates = SPLIT_CSV_FILENAMES[split]
    for filename in candidates:
        path = csv_filter_dir / filename
        if path.is_file():
            return path
    tried = ", ".join(str(csv_filter_dir / name) for name in candidates)
    raise FileNotFoundError(f"No CSV file for split={split} under {csv_filter_dir}; tried: {tried}")


def binary_name_from_idb_path(idb_path: str, split: str) -> str | None:
    parts = idb_path.replace("\\", "/").split("/")
    project_index: int | None = None
    for index in range(len(parts) - 2):
        if parts[index] == "Dataset-1":
            project_index = index + 1
            break
    if project_index is None:
        if len(parts) < 2:
            return None
        project_index = len(parts) - 2
    if project_index + 1 >= len(parts):
        return None
    project = parts[project_index]
    stem = parts[project_index + 1]
    if stem.endswith(".i64") or stem.endswith(".idb"):
        stem = stem.rsplit(".", 1)[0]
    if not project or not stem:
        return None
    return f"{split}/{project}/{stem}"


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def final_set_artifacts() -> list[str]:
    return [
        "train_dataset_pool",
        "train_task_dataset",
        "train_positive_map.pkl",
        "validation_dataset_pool",
        "validation_task_dataset",
        "validation_positive_map.pkl",
        "split_indices.json",
        "filter_summary.json",
    ]


def replace_final_set_in_place(final_set: Path, staged_final_set: Path) -> None:
    backup_root = final_set.parent / f".{final_set.name}.pre-filter-{os.getpid()}"
    remove_path(backup_root)
    backup_root.mkdir(parents=True, exist_ok=False)
    moved_to_backup: list[str] = []
    installed: list[str] = []

    try:
        for name in final_set_artifacts():
            current = final_set / name
            if current.exists() or current.is_symlink():
                shutil.move(str(current), str(backup_root / name))
                moved_to_backup.append(name)

        for name in final_set_artifacts():
            staged = staged_final_set / name
            if staged.exists() or staged.is_symlink():
                shutil.move(str(staged), str(final_set / name))
                installed.append(name)

        remove_path(backup_root)
    except Exception:
        for name in installed:
            remove_path(final_set / name)
        for name in moved_to_backup:
            backup = backup_root / name
            if backup.exists() or backup.is_symlink():
                shutil.move(str(backup), str(final_set / name))
        raise
    finally:
        remove_path(staged_final_set)
        remove_path(backup_root)


def derive_origin_and_opt(binary_name: str) -> tuple[str, str]:
    normalized = binary_name.replace("\\", "/")
    directory, _, basename = normalized.rpartition("/")
    match = OPT_LEVEL_PATTERN.match(basename)
    if match:
        origin_basename = f"{match.group('prefix')}_{match.group('binary')}"
        origin = f"{directory}/{origin_basename}" if directory else origin_basename
        return origin, match.group("opt")

    legacy_parts = basename.rsplit("-", 2)
    if len(legacy_parts) == 3 and legacy_parts[1] in {"O0", "O1", "O2", "O3", "Os", "Oz"}:
        origin_basename = legacy_parts[0]
        origin = f"{directory}/{origin_basename}" if directory else origin_basename
        return origin, legacy_parts[1]

    return normalized, ""


def ensure_metadata_columns(dataset: datasets.Dataset) -> datasets.Dataset:
    if "origin_binary_name" in dataset.column_names and "opt_level" in dataset.column_names:
        return dataset

    metadata = dataset.select_columns(["binary_name"])
    origins: list[str] = []
    opt_levels: list[str] = []
    for batch in metadata.iter(batch_size=10000):
        for binary_name in batch["binary_name"]:
            origin, opt_level = derive_origin_and_opt(binary_name)
            origins.append(origin)
            opt_levels.append(opt_level)

    if "origin_binary_name" not in dataset.column_names:
        dataset = dataset.add_column("origin_binary_name", origins)
    if "opt_level" not in dataset.column_names:
        dataset = dataset.add_column("opt_level", opt_levels)
    return dataset


def save_empty_validation_outputs(output_root: Path, filtered_pool: datasets.Dataset) -> None:
    filtered_pool.select([]).save_to_disk(str(output_root / "validation_dataset_pool"))
    datasets.Dataset.from_dict({"anchor_idx": []}).save_to_disk(str(output_root / "validation_task_dataset"))
    with (output_root / "validation_positive_map.pkl").open("wb") as handle:
        pickle.dump({}, handle)


def write_split_indices(output_root: Path, train_size: int) -> None:
    payload = {"train": list(range(train_size)), "validation": []}
    (output_root / "split_indices.json").write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def binary_name_from_reference_path(file_path: str) -> str | None:
    parts = file_path.replace("\\", "/").split("/")
    try:
        dataset_index = parts.index("Dataset-1")
        split = parts[dataset_index + 1]
        project = parts[dataset_index + 2]
        function_dir = parts[dataset_index + 3]
    except (ValueError, IndexError):
        return None

    stem = function_dir.removesuffix("_functions")
    if not split or not project or not stem:
        return None
    return f"{split}/{project}/{stem}"


def origin_from_binary_name(binary_name: str) -> str:
    normalized = binary_name.replace("\\", "/")
    directory, _, basename = normalized.rpartition("/")
    match = OPT_LEVEL_PATTERN.match(basename)
    if not match:
        return normalized
    origin_basename = f"{match.group('prefix')}_{match.group('binary')}"
    return f"{directory}/{origin_basename}" if directory else origin_basename


def key_binary_for_match(binary_name: str, match_mode: MatchMode | str) -> str:
    match_value = match_mode.value if isinstance(match_mode, MatchMode) else str(match_mode)
    if match_value == MatchMode.origin.value:
        return origin_from_binary_name(binary_name)
    return binary_name


def detect_reference_kind(reference: Path, reference_kind: ReferenceKind) -> ReferenceKind:
    if reference_kind != ReferenceKind.auto:
        return reference_kind

    if reference.is_file():
        if reference.suffix.lower() == ".csv":
            return ReferenceKind.csv
        raise typer.BadParameter(f"Reference file is not a CSV: {reference}")

    if reference.is_dir():
        if (reference / "train_dataset_pool").is_dir() or (reference / "dataset_info.json").is_file():
            return ReferenceKind.final_set
        csv_candidates = {name for names in SPLIT_CSV_FILENAMES.values() for name in names}
        if any((reference / name).is_file() for name in csv_candidates) or any(reference.glob("*.csv")):
            return ReferenceKind.csv

    raise typer.BadParameter(
        "Could not infer reference type. Use --reference-kind csv or --reference-kind final-set."
    )


def resolve_csv_path(reference: Path, input_final_set: Path, split: Optional[str]) -> tuple[Path, str]:
    normalized_split: str | None = normalize_split(split) if split else None
    if normalized_split is None:
        normalized_split = infer_split(input_final_set) or infer_split(reference)
    if normalized_split is None:
        raise typer.BadParameter("Could not infer split for CSV reference. Pass --split train|validation|test.")

    if reference.is_dir():
        return csv_file_for_split(reference, normalized_split), normalized_split
    return reference, normalized_split


def load_csv_allowed_keys(
    csv_path: Path,
    split: str,
    match_mode: MatchMode,
) -> tuple[set[str], dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV reference not found: {csv_path}")

    allowed: set[str] = set()
    rows = 0
    bad_rows = 0
    duplicate_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "idb_path" not in reader.fieldnames or "func_name" not in reader.fieldnames:
            raise ValueError(f"CSV must contain idb_path and func_name columns: {csv_path}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("rows={task.completed:,}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Reading CSV reference", total=None)
            for row in reader:
                rows += 1
                binary_name = binary_name_from_idb_path(row.get("idb_path", ""), split)
                function_name = row.get("func_name", "")
                if not binary_name or not function_name:
                    bad_rows += 1
                    progress.advance(task)
                    continue
                key = f"{key_binary_for_match(binary_name, match_mode)}::{function_name}"
                if key in allowed:
                    duplicate_rows += 1
                allowed.add(key)
                progress.advance(task)

    return allowed, {
        "reference_kind": ReferenceKind.csv.value,
        "reference_csv_path": str(csv_path),
        "reference_split": split,
        "reference_rows": rows,
        "reference_allowed": len(allowed),
        "reference_bad_rows": bad_rows,
        "reference_duplicate_rows": duplicate_rows,
    }


def reference_columns(reference_pool: datasets.Dataset, match_mode: MatchMode) -> tuple[list[str], str]:
    column_names = set(reference_pool.column_names)
    if "function_name" not in column_names:
        raise ValueError("Reference dataset must contain function_name.")

    if match_mode == MatchMode.origin and "origin_binary_name" in column_names:
        return ["origin_binary_name", "function_name"], "origin_binary_name"
    if "binary_name" in column_names:
        return ["binary_name", "function_name"], "binary_name"
    if "file_path" in column_names:
        return ["file_path", "function_name"], "file_path"
    raise ValueError("Reference dataset must contain binary_name, origin_binary_name, or file_path.")


def load_final_set_allowed_keys(
    reference_pool: datasets.Dataset,
    match_mode: MatchMode,
) -> tuple[set[str], dict[str, Any]]:
    columns, key_column = reference_columns(reference_pool, match_mode)
    allowed: set[str] = set()
    bad_rows = 0
    duplicate_rows = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reading final_set reference", total=len(reference_pool))
        for batch in reference_pool.select_columns(columns).iter(batch_size=50000):
            function_names = batch["function_name"]
            raw_keys = batch[key_column]
            for raw_key, function_name in zip(raw_keys, function_names):
                if key_column == "file_path":
                    binary_name = binary_name_from_reference_path(raw_key)
                    key_binary = key_binary_for_match(binary_name, match_mode) if binary_name else None
                elif key_column == "binary_name":
                    key_binary = key_binary_for_match(raw_key, match_mode)
                else:
                    key_binary = raw_key

                if not key_binary or not function_name:
                    bad_rows += 1
                    continue
                key = f"{key_binary}::{function_name}"
                if key in allowed:
                    duplicate_rows += 1
                allowed.add(key)
            progress.advance(task, len(function_names))

    return allowed, {
        "reference_kind": ReferenceKind.final_set.value,
        "reference_rows": len(reference_pool),
        "reference_allowed": len(allowed),
        "reference_bad_rows": bad_rows,
        "reference_duplicate_rows": duplicate_rows,
        "reference_key_column": key_column,
    }


def load_reference_allowed_keys(
    reference: Path,
    input_final_set: Path,
    reference_kind: ReferenceKind,
    match_mode: MatchMode,
    split: Optional[str],
) -> tuple[set[str], dict[str, Any]]:
    resolved_kind = detect_reference_kind(reference, reference_kind)

    if resolved_kind == ReferenceKind.csv:
        csv_path, normalized_split = resolve_csv_path(reference, input_final_set, split)
        allowed, stats = load_csv_allowed_keys(csv_path, normalized_split, match_mode)
        stats["reference_path"] = str(reference)
        return allowed, stats

    reference_final_set, reference_pool_path = resolve_input_pool(reference)
    console.print(f"[cyan]Loading reference pool:[/cyan] {reference_pool_path}")
    reference_pool = datasets.load_from_disk(str(reference_pool_path))
    allowed, stats = load_final_set_allowed_keys(reference_pool, match_mode)
    stats.update(
        {
            "reference_path": str(reference),
            "reference_final_set": str(reference_final_set),
            "reference_pool": str(reference_pool_path),
        }
    )
    return allowed, stats


def input_key_column(dataset: datasets.Dataset, match_mode: MatchMode) -> str:
    if "function_name" not in dataset.column_names:
        raise ValueError("Input dataset must contain function_name.")
    key_column = "origin_binary_name" if match_mode == MatchMode.origin else "binary_name"
    if key_column not in dataset.column_names:
        raise ValueError(f"Input dataset must contain {key_column} for match_mode={match_mode.value}.")
    return key_column


def scan_batch(start: int, key_binaries: list[str], function_names: list[str], allowed: set[str]) -> tuple[int, int, list[int], set[str]]:
    keep_indices: list[int] = []
    matched_keys: set[str] = set()
    for offset, (key_binary, function_name) in enumerate(zip(key_binaries, function_names)):
        key = f"{key_binary}::{function_name}"
        if key in allowed:
            keep_indices.append(start + offset)
            matched_keys.add(key)
    return start, len(function_names), keep_indices, matched_keys


def scan_batch_worker(task: tuple[int, list[str], list[str]]) -> tuple[int, int, list[int], set[str]]:
    if _WORKER_ALLOWED is None:
        raise RuntimeError("Worker allowed-key set was not initialized.")
    start, key_binaries, function_names = task
    return scan_batch(start, key_binaries, function_names, _WORKER_ALLOWED)


def iter_scan_tasks(
    metadata: datasets.Dataset,
    key_column: str,
    batch_size: int,
) -> Any:
    start = 0
    for batch in metadata.iter(batch_size=batch_size):
        key_binaries = list(batch[key_column])
        function_names = list(batch["function_name"])
        yield start, key_binaries, function_names
        start += len(function_names)


def worker_count(requested_workers: int) -> int:
    if requested_workers < 0:
        raise typer.BadParameter("--workers must be >= 0.")
    if requested_workers > 0:
        return requested_workers
    return max(1, os.cpu_count() or 1)


def collect_keep_indices(
    dataset: datasets.Dataset,
    allowed: set[str],
    match_mode: MatchMode,
    batch_size: int,
    workers: int,
) -> tuple[list[int], set[str]]:
    key_column = input_key_column(dataset, match_mode)
    metadata = dataset.select_columns([key_column, "function_name"])
    effective_workers = worker_count(workers)

    if effective_workers > 1 and "fork" not in mp.get_all_start_methods():
        console.print("[yellow]Multiprocessing fork is unavailable; falling back to one worker.[/yellow]")
        effective_workers = 1

    keep_chunks: list[tuple[int, list[int]]] = []
    matched_keys: set[str] = set()
    kept_count = 0

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("kept={task.fields[kept]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task(
            f"Scanning input pool ({effective_workers} worker{'s' if effective_workers != 1 else ''})",
            total=len(dataset),
            kept="0",
        )

        if effective_workers == 1:
            for start, key_binaries, function_names in iter_scan_tasks(metadata, key_column, batch_size):
                result_start, batch_len, keep_indices, batch_matched = scan_batch(
                    start,
                    key_binaries,
                    function_names,
                    allowed,
                )
                keep_chunks.append((result_start, keep_indices))
                matched_keys.update(batch_matched)
                kept_count += len(keep_indices)
                progress.update(task, advance=batch_len, kept=f"{kept_count:,}")
        else:
            global _WORKER_ALLOWED
            _WORKER_ALLOWED = allowed
            ctx = mp.get_context("fork")
            try:
                with ctx.Pool(processes=effective_workers) as pool:
                    for result_start, batch_len, keep_indices, batch_matched in pool.imap_unordered(
                        scan_batch_worker,
                        iter_scan_tasks(metadata, key_column, batch_size),
                        chunksize=1,
                    ):
                        keep_chunks.append((result_start, keep_indices))
                        matched_keys.update(batch_matched)
                        kept_count += len(keep_indices)
                        progress.update(task, advance=batch_len, kept=f"{kept_count:,}")
            finally:
                _WORKER_ALLOWED = None

    keep_indices: list[int] = []
    for _, chunk_indices in sorted(keep_chunks, key=lambda item: item[0]):
        keep_indices.extend(chunk_indices)
    return keep_indices, matched_keys


def build_positive_map_with_progress(dataset: datasets.Dataset, batch_size: int) -> dict[int, list[int]]:
    dataset = ensure_metadata_columns(dataset)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    columns = dataset.select_columns(["origin_binary_name", "function_name"])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building positive map groups", total=len(dataset))
        seen = 0
        for batch in columns.iter(batch_size=batch_size):
            origins = batch["origin_binary_name"]
            function_names = batch["function_name"]
            for offset, (origin, function_name) in enumerate(zip(origins, function_names)):
                if origin and function_name:
                    groups[(origin, function_name)].append(seen + offset)
            seen += len(function_names)
            progress.advance(task, len(function_names))

    positive_map: dict[int, list[int]] = {}
    with console.status("[cyan]Expanding positive map anchors...[/cyan]"):
        for indices in groups.values():
            if len(indices) <= 1:
                continue
            for index in indices:
                positive_map[index] = [other for other in indices if other != index]
    return positive_map


def print_summary(summary: dict[str, Any]) -> None:
    table = Table(title="Filter Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for key in [
        "reference_kind",
        "match_mode",
        "input_rows",
        "kept_rows",
        "dropped_rows",
        "reference_allowed",
        "matched_unique",
        "missing_reference_allowed",
        "positive_map_entries",
        "task_anchors",
        "output_final_set",
    ]:
        value = summary.get(key)
        if isinstance(value, int):
            value = f"{value:,}"
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def main(
    input_path: Path = typer.Argument(..., metavar="INPUT", help="Input final_set directory to filter."),
    reference: Path = typer.Argument(..., metavar="REFERENCE", help="CSV file/dir or final_set used as whitelist."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write filtered final_set here. Omit to replace input in place."),
    reference_kind: ReferenceKind = typer.Option(ReferenceKind.auto, "--reference-kind", help="Reference type."),
    match_mode: MatchMode = typer.Option(MatchMode.exact, "--match-mode", help="Function key matching mode."),
    split: Optional[str] = typer.Option(None, "--split", help="CSV split: train, validation, or test. Usually inferred."),
    workers: int = typer.Option(0, "--workers", "-j", help="Scan workers. 0 means all CPU cores; 1 disables multiprocessing."),
    batch_size: int = typer.Option(50000, "--batch-size", help="Rows per scan batch."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow replacing an existing --output directory."),
) -> None:
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be > 0.")

    input_final_set, input_pool = resolve_input_pool(input_path)
    reference = reference.expanduser().resolve()
    replace_in_place = output is None
    final_output = input_final_set if replace_in_place else output.expanduser().resolve()

    if not replace_in_place:
        if final_output == input_final_set or final_output == input_pool:
            raise typer.BadParameter("For in-place replacement, omit --output. --output must be a different path.")
        if final_output.exists() or final_output.is_symlink():
            if not overwrite:
                raise typer.BadParameter(f"Output already exists: {final_output}. Use --overwrite to replace it.")
            remove_path(final_output)
        final_output.parent.mkdir(parents=True, exist_ok=True)

    output_root = final_output.parent / f".{final_output.name}.filtering-{os.getpid()}"
    remove_path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    console.rule("[bold blue]Filter final_set by reference[/bold blue]")
    console.print(f"[cyan]Input final_set:[/cyan] {input_final_set}")
    console.print(f"[cyan]Input pool:[/cyan] {input_pool}")
    console.print(f"[cyan]Reference:[/cyan] {reference}")
    console.print(f"[cyan]Output:[/cyan] {final_output}")
    console.print(f"[cyan]Mode:[/cyan] {'replace input in place' if replace_in_place else 'write filtered copy'}")
    console.print(f"[cyan]Match:[/cyan] {match_mode.value}")
    console.print(f"[cyan]Workers:[/cyan] {worker_count(workers)}")

    try:
        allowed, reference_stats = load_reference_allowed_keys(
            reference=reference,
            input_final_set=input_final_set,
            reference_kind=reference_kind,
            match_mode=match_mode,
            split=split,
        )
        console.print(
            "[green]Reference loaded:[/green] "
            f"rows={reference_stats['reference_rows']:,}, "
            f"allowed={reference_stats['reference_allowed']:,}, "
            f"bad={reference_stats['reference_bad_rows']:,}, "
            f"duplicates={reference_stats['reference_duplicate_rows']:,}"
        )

        console.print(f"[cyan]Loading input pool:[/cyan] {input_pool}")
        dataset = datasets.load_from_disk(str(input_pool))
        console.print(f"[green]Input rows:[/green] {len(dataset):,}")

        keep_indices, matched_keys = collect_keep_indices(
            dataset=dataset,
            allowed=allowed,
            match_mode=match_mode,
            batch_size=batch_size,
            workers=workers,
        )
        if not keep_indices:
            raise RuntimeError("Reference filter matched zero rows; check split/reference compatibility.")

        console.print(
            "[green]Filter matched:[/green] "
            f"kept={len(keep_indices):,}, matched_unique={len(matched_keys):,}, "
            f"missing_reference={max(0, len(allowed) - len(matched_keys)):,}"
        )

        filtered_pool = dataset.select(keep_indices)
        with console.status("[cyan]Ensuring metadata columns...[/cyan]"):
            filtered_pool = ensure_metadata_columns(filtered_pool)

        console.print(f"[cyan]Saving train_dataset_pool:[/cyan] {output_root / 'train_dataset_pool'}")
        filtered_pool.save_to_disk(str(output_root / "train_dataset_pool"))

        positive_map = build_positive_map_with_progress(filtered_pool, batch_size)
        with (output_root / "train_positive_map.pkl").open("wb") as handle:
            pickle.dump(positive_map, handle)
        anchor_indices = sorted(positive_map)

        console.print(f"[cyan]Saving train_task_dataset:[/cyan] anchors={len(anchor_indices):,}")
        datasets.Dataset.from_dict({"anchor_idx": anchor_indices}).save_to_disk(str(output_root / "train_task_dataset"))
        save_empty_validation_outputs(output_root, filtered_pool)
        write_split_indices(output_root, len(filtered_pool))

        summary = {
            "input_final_set": str(input_final_set),
            "input_pool": str(input_pool),
            "reference_path": str(reference),
            "output_final_set": str(final_output),
            "staged_final_set": str(output_root),
            "replace_in_place": replace_in_place,
            "match_mode": match_mode.value,
            "workers": worker_count(workers),
            "batch_size": batch_size,
            **reference_stats,
            "input_rows": len(dataset),
            "kept_rows": len(filtered_pool),
            "dropped_rows": len(dataset) - len(filtered_pool),
            "matched_unique": len(matched_keys),
            "missing_reference_allowed": max(0, len(allowed) - len(matched_keys)),
            "positive_map_entries": len(positive_map),
            "task_anchors": len(anchor_indices),
        }
        (output_root / "filter_summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if replace_in_place:
            console.print("[yellow]Installing filtered final_set in place...[/yellow]")
            replace_final_set_in_place(input_final_set, output_root)
        else:
            console.print(f"[cyan]Installing filtered final_set:[/cyan] {final_output}")
            os.replace(output_root, final_output)

        print_summary(summary)
        console.print(f"[green]Done.[/green] Summary: {final_output / 'filter_summary.json'}")
    except Exception as exc:
        remove_path(output_root)
        console.print(f"[bold red]ERROR:[/bold red] {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
