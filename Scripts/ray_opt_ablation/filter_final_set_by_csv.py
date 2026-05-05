#!/usr/bin/env python3
"""Filter an existing ReGraph final_set with Dataset-1 CSV function lists."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets


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
    raise FileNotFoundError(
        f"No CSV file for split={split} under {csv_filter_dir}; "
        f"tried: {', '.join(str(csv_filter_dir / name) for name in candidates)}"
    )


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


def load_allowed_keys(csv_path: Path, split: str) -> tuple[set[str], dict[str, Any]]:
    allowed: set[str] = set()
    rows = 0
    skipped = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "idb_path" not in reader.fieldnames or "func_name" not in reader.fieldnames:
            raise ValueError(f"CSV must contain idb_path and func_name columns: {csv_path}")
        for row in reader:
            rows += 1
            binary_name = binary_name_from_idb_path(row.get("idb_path", ""), split)
            function_name = row.get("func_name", "")
            if not binary_name or not function_name:
                skipped += 1
                continue
            allowed.add(f"{binary_name}::{function_name}")
    return allowed, {"csv_path": str(csv_path), "csv_rows": rows, "csv_allowed": len(allowed), "csv_skipped": skipped}


def collect_keep_indices(dataset: datasets.Dataset, allowed: set[str], batch_size: int, progress_every: int) -> tuple[list[int], set[str]]:
    metadata = dataset.select_columns(["binary_name", "function_name"])
    keep_indices: list[int] = []
    matched_keys: set[str] = set()
    seen = 0
    for batch in metadata.iter(batch_size=batch_size):
        binary_names = batch["binary_name"]
        function_names = batch["function_name"]
        for offset, (binary_name, function_name) in enumerate(zip(binary_names, function_names)):
            key = f"{binary_name}::{function_name}"
            if key in allowed:
                keep_indices.append(seen + offset)
                matched_keys.add(key)
        seen += len(binary_names)
        if progress_every > 0 and (seen % progress_every) < len(binary_names):
            print(f"[scan] rows={seen} kept={len(keep_indices)}", flush=True)
    return keep_indices, matched_keys


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


def build_positive_map(dataset: datasets.Dataset, batch_size: int) -> dict[int, list[int]]:
    dataset = ensure_metadata_columns(dataset)
    columns = dataset.select_columns(["origin_binary_name", "function_name"])
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    seen = 0
    for batch in columns.iter(batch_size=batch_size):
        origins = batch["origin_binary_name"]
        function_names = batch["function_name"]
        for offset, (origin, function_name) in enumerate(zip(origins, function_names)):
            if origin and function_name:
                groups[(origin, function_name)].append(seen + offset)
        seen += len(origins)

    positive_map: dict[int, list[int]] = {}
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        for index in indices:
            positive_map[index] = [other for other in indices if other != index]
    return positive_map


def save_empty_validation_outputs(output_root: Path, filtered_pool: datasets.Dataset) -> None:
    filtered_pool.select([]).save_to_disk(str(output_root / "validation_dataset_pool"))
    datasets.Dataset.from_dict({"anchor_idx": []}).save_to_disk(str(output_root / "validation_task_dataset"))
    with (output_root / "validation_positive_map.pkl").open("wb") as handle:
        pickle.dump({}, handle)


def write_split_indices(output_root: Path, train_size: int) -> None:
    payload = {"train": list(range(train_size)), "validation": []}
    (output_root / "split_indices.json").write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


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


def resolve_csv_path(csv_arg: Path, split: str) -> Path:
    csv_arg = csv_arg.expanduser().resolve()
    if csv_arg.is_dir():
        return csv_file_for_split(csv_arg, split)
    return csv_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a final_set with a Dataset-1 CSV. By default the input final_set is replaced in place; "
            "use --output to write a new final_set and leave the input untouched."
        )
    )
    parser.add_argument("input", help="Input final_set directory")
    parser.add_argument("csv", help="CSV file path; a CSV directory is also accepted if split can be inferred from input")
    parser.add_argument("--output", "-o", default="", help="Optional output final_set directory; default replaces input in place")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_final_set, input_pool = resolve_input_pool(Path(args.input))
    split = infer_split(input_final_set)
    if not split:
        split = infer_split(Path(args.csv))
    if not split:
        raise SystemExit("Could not infer split from input directory or CSV path.")

    csv_path = resolve_csv_path(Path(args.csv), split)
    if not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")

    csv_split = infer_split(csv_path)
    if csv_split and csv_split != split:
        raise SystemExit(f"Split mismatch: input looks like {split}, but CSV looks like {csv_split}: {csv_path}")

    replace_in_place = not bool(args.output)
    final_output = input_final_set if replace_in_place else Path(args.output).expanduser().resolve()
    if not replace_in_place:
        if final_output == input_final_set or final_output == input_pool:
            raise SystemExit("For in-place replacement, omit --output. --output must be a different path.")
        if final_output.exists() or final_output.is_symlink():
            raise SystemExit(f"Output already exists: {final_output}")
        final_output.parent.mkdir(parents=True, exist_ok=True)

    output_root = final_output.parent / f".{final_output.name}.filtering-{os.getpid()}"
    remove_path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    print(f"[input] final_set={input_final_set}", flush=True)
    print(f"[input] pool={input_pool}", flush=True)
    if replace_in_place:
        print("[mode] replace input in place", flush=True)
    else:
        print(f"[mode] write filtered copy to {final_output}", flush=True)
    print(f"[stage] temp_final_set={output_root}", flush=True)
    print(f"[csv] split={split} file={csv_path}", flush=True)

    allowed, csv_stats = load_allowed_keys(csv_path, split)
    print(f"[csv] rows={csv_stats['csv_rows']} allowed={csv_stats['csv_allowed']} skipped={csv_stats['csv_skipped']}", flush=True)

    dataset = datasets.load_from_disk(str(input_pool))
    if "binary_name" not in dataset.column_names or "function_name" not in dataset.column_names:
        raise SystemExit("Input dataset must contain binary_name and function_name columns.")
    print(f"[load] rows={len(dataset)} columns={dataset.column_names}", flush=True)

    batch_size = 10000
    progress_every = 1_000_000
    keep_indices, matched_keys = collect_keep_indices(dataset, allowed, batch_size, progress_every)
    if not keep_indices:
        raise SystemExit("CSV filter matched zero rows; check that the final_set and CSV belong to the same split.")
    print(
        f"[filter] kept_rows={len(keep_indices)} matched_unique={len(matched_keys)} "
        f"missing_csv={max(0, len(allowed) - len(matched_keys))}",
        flush=True,
    )

    filtered_pool = dataset.select(keep_indices)
    filtered_pool = ensure_metadata_columns(filtered_pool)
    filtered_pool.save_to_disk(str(output_root / "train_dataset_pool"))
    print(f"[save] train_dataset_pool rows={len(filtered_pool)}", flush=True)

    positive_map = build_positive_map(filtered_pool, batch_size)
    with (output_root / "train_positive_map.pkl").open("wb") as handle:
        pickle.dump(positive_map, handle)
    anchor_indices = sorted(positive_map)
    datasets.Dataset.from_dict({"anchor_idx": anchor_indices}).save_to_disk(str(output_root / "train_task_dataset"))
    print(f"[save] train_task_dataset anchors={len(anchor_indices)} positive_map={len(positive_map)}", flush=True)

    save_empty_validation_outputs(output_root, filtered_pool)
    write_split_indices(output_root, len(filtered_pool))

    summary = {
        "input_final_set": str(input_final_set),
        "input_pool": str(input_pool),
        "output_final_set": str(final_output),
        "staged_final_set": str(output_root),
        "replace_in_place": replace_in_place,
        "split": split,
        **csv_stats,
        "input_rows": len(dataset),
        "kept_rows": len(filtered_pool),
        "dropped_rows": len(dataset) - len(filtered_pool),
        "matched_unique": len(matched_keys),
        "missing_csv_allowed": max(0, len(allowed) - len(matched_keys)),
        "positive_map_entries": len(positive_map),
        "task_anchors": len(anchor_indices),
    }
    (output_root / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if replace_in_place:
        print("[replace] installing filtered final_set in place", flush=True)
        replace_final_set_in_place(input_final_set, output_root)
    else:
        print(f"[install] moving filtered final_set to {final_output}", flush=True)
        os.replace(output_root, final_output)
    print(f"[done] summary={final_output / 'filter_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
