#!/usr/bin/env python3
"""Filter an existing ReGraph final_set with a CSV or final_set reference."""
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
OPT_LEVEL_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<opt>O0|O1|O2|O3|Os|Og|Oz|Oc|Oc2)_(?P<binary>.+)$")
REFERENCE_KINDS = {"auto", "csv", "final-set"}
MATCH_MODES = {"exact", "origin"}


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


def final_set_dir_for_split(root: Path, split: str) -> Path:
    return root / f"{normalize_split(split)}_final_set"


def final_set_split_dirs(root: Path) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for split in ("train", "validation", "test"):
        candidate = final_set_dir_for_split(root, split)
        if (candidate / "train_dataset_pool").is_dir():
            results.append((split, candidate))
    return results


def resolve_reference_pool(reference: Path, input_final_set: Path, split: str | None) -> tuple[Path, Path]:
    reference = reference.expanduser().resolve()
    try:
        return resolve_input_pool(reference)
    except ValueError:
        pass

    normalized_split = normalize_split(split) if split else infer_split(input_final_set)
    if normalized_split:
        split_final_set = final_set_dir_for_split(reference, normalized_split)
        if (split_final_set / "train_dataset_pool").is_dir():
            return split_final_set.resolve(), (split_final_set / "train_dataset_pool").resolve()

    split_dirs = final_set_split_dirs(reference)
    if len(split_dirs) == 1:
        _, split_final_set = split_dirs[0]
        return split_final_set.resolve(), (split_final_set / "train_dataset_pool").resolve()
    if split_dirs:
        available = ", ".join(split for split, _ in split_dirs)
        raise ValueError(f"Reference root contains multiple final_set splits ({available}); pass --split.")

    raise ValueError(
        "Reference must be a CSV file/dir, a final_set directory, a train_dataset_pool directory, "
        "or a root containing <split>_final_set directories."
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


def origin_from_binary_name(binary_name: str) -> str:
    normalized = binary_name.replace("\\", "/")
    directory, _, basename = normalized.rpartition("/")
    match = OPT_LEVEL_PATTERN.match(basename)
    if not match:
        return normalized
    origin_basename = f"{match.group('prefix')}_{match.group('binary')}"
    return f"{directory}/{origin_basename}" if directory else origin_basename


def key_binary_for_match(binary_name: str, match_mode: str) -> str:
    if match_mode == "origin":
        return origin_from_binary_name(binary_name)
    return binary_name


def load_csv_allowed_keys(csv_path: Path, split: str, match_mode: str) -> tuple[set[str], dict[str, Any]]:
    allowed: set[str] = set()
    rows = 0
    skipped = 0
    duplicates = 0
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
            key = f"{key_binary_for_match(binary_name, match_mode)}::{function_name}"
            if key in allowed:
                duplicates += 1
            allowed.add(key)
    return allowed, {
        "reference_kind": "csv",
        "reference_csv_path": str(csv_path),
        "reference_rows": rows,
        "reference_allowed": len(allowed),
        "reference_skipped": skipped,
        "reference_duplicates": duplicates,
    }


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


def reference_columns(reference_pool: datasets.Dataset, match_mode: str) -> tuple[list[str], str]:
    column_names = set(reference_pool.column_names)
    if "function_name" not in column_names:
        raise ValueError("Reference dataset must contain function_name.")

    if match_mode == "origin" and "origin_binary_name" in column_names:
        return ["origin_binary_name", "function_name"], "origin_binary_name"
    if "binary_name" in column_names:
        return ["binary_name", "function_name"], "binary_name"
    if "file_path" in column_names:
        return ["file_path", "function_name"], "file_path"
    raise ValueError("Reference dataset must contain binary_name, origin_binary_name, or file_path.")


def load_final_set_allowed_keys(reference_pool: datasets.Dataset, match_mode: str, batch_size: int) -> tuple[set[str], dict[str, Any]]:
    columns, key_column = reference_columns(reference_pool, match_mode)
    allowed: set[str] = set()
    skipped = 0
    duplicates = 0
    seen = 0
    for batch in reference_pool.select_columns(columns).iter(batch_size=batch_size):
        raw_keys = batch[key_column]
        function_names = batch["function_name"]
        for raw_key, function_name in zip(raw_keys, function_names):
            if key_column == "file_path":
                binary_name = binary_name_from_reference_path(raw_key)
                key_binary = key_binary_for_match(binary_name, match_mode) if binary_name else None
            elif key_column == "binary_name":
                key_binary = key_binary_for_match(raw_key, match_mode)
            else:
                key_binary = raw_key

            if not key_binary or not function_name:
                skipped += 1
                continue
            key = f"{key_binary}::{function_name}"
            if key in allowed:
                duplicates += 1
            allowed.add(key)
        seen += len(function_names)
        if seen and seen % 1_000_000 == 0:
            print(f"[reference] rows={seen} allowed={len(allowed)}", flush=True)

    return allowed, {
        "reference_kind": "final-set",
        "reference_rows": len(reference_pool),
        "reference_allowed": len(allowed),
        "reference_skipped": skipped,
        "reference_duplicates": duplicates,
        "reference_key_column": key_column,
    }


def detect_reference_kind(reference: Path, reference_kind: str) -> str:
    if reference_kind != "auto":
        return reference_kind
    if reference.is_file():
        if reference.suffix.lower() == ".csv":
            return "csv"
        raise ValueError(f"Reference file is not a CSV: {reference}")
    if reference.is_dir():
        if (reference / "train_dataset_pool").is_dir() or (reference / "dataset_info.json").is_file():
            return "final-set"
        if final_set_split_dirs(reference):
            return "final-set"
        csv_candidates = {name for names in SPLIT_CSV_FILENAMES.values() for name in names}
        if any((reference / name).is_file() for name in csv_candidates) or any(reference.glob("*.csv")):
            return "csv"
    raise ValueError("Could not infer reference type. Use --reference-kind csv or final-set.")


def load_reference_allowed_keys(
    reference: Path,
    input_final_set: Path,
    split: str | None,
    reference_kind: str,
    match_mode: str,
    batch_size: int,
) -> tuple[set[str], dict[str, Any]]:
    resolved_kind = detect_reference_kind(reference, reference_kind)
    if resolved_kind == "csv":
        normalized_split = normalize_split(split) if split else infer_split(input_final_set) or infer_split(reference)
        if not normalized_split:
            raise ValueError("Could not infer split for CSV reference. Pass --split train|validation|test.")
        csv_path = resolve_csv_path(reference, normalized_split)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        csv_split = infer_split(csv_path)
        if csv_split and csv_split != normalized_split:
            raise ValueError(f"Split mismatch: input looks like {normalized_split}, but CSV looks like {csv_split}: {csv_path}")
        allowed, stats = load_csv_allowed_keys(csv_path, normalized_split, match_mode)
        stats["reference_path"] = str(reference)
        stats["reference_split"] = normalized_split
        return allowed, stats

    reference_final_set, reference_pool = resolve_reference_pool(reference, input_final_set, split)
    dataset = datasets.load_from_disk(str(reference_pool))
    allowed, stats = load_final_set_allowed_keys(dataset, match_mode, batch_size)
    stats.update(
        {
            "reference_path": str(reference),
            "reference_final_set": str(reference_final_set),
            "reference_pool": str(reference_pool),
        }
    )
    return allowed, stats


def input_key_column(dataset: datasets.Dataset, match_mode: str) -> str:
    if "function_name" not in dataset.column_names:
        raise ValueError("Input dataset must contain function_name.")
    key_column = "origin_binary_name" if match_mode == "origin" else "binary_name"
    if key_column not in dataset.column_names:
        raise ValueError(f"Input dataset must contain {key_column} for match_mode={match_mode}.")
    return key_column


def collect_keep_indices(dataset: datasets.Dataset, allowed: set[str], match_mode: str, batch_size: int, progress_every: int) -> tuple[list[int], set[str]]:
    key_column = input_key_column(dataset, match_mode)
    metadata = dataset.select_columns([key_column, "function_name"])
    keep_indices: list[int] = []
    matched_keys: set[str] = set()
    seen = 0
    for batch in metadata.iter(batch_size=batch_size):
        key_binaries = batch[key_column]
        function_names = batch["function_name"]
        for offset, (key_binary, function_name) in enumerate(zip(key_binaries, function_names)):
            key = f"{key_binary}::{function_name}"
            if key in allowed:
                keep_indices.append(seen + offset)
                matched_keys.add(key)
        seen += len(key_binaries)
        if progress_every > 0 and (seen % progress_every) < len(key_binaries):
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
    if len(legacy_parts) == 3 and legacy_parts[1] in {"O0", "O1", "O2", "O3", "Os", "Og", "Oz", "Oc", "Oc2"}:
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
            "Filter a final_set with a Dataset-1 CSV or another final_set reference. "
            "By default the input final_set is replaced in place; use --output to write a new final_set "
            "and leave the input untouched."
        )
    )
    parser.add_argument("input", help="Input final_set directory")
    parser.add_argument(
        "reference",
        help=(
            "CSV file/dir, final_set, train_dataset_pool, or root containing *_final_set directories "
            "used as whitelist"
        ),
    )
    parser.add_argument("--output", "-o", default="", help="Optional output final_set directory; default replaces input in place")
    parser.add_argument("--reference-kind", choices=sorted(REFERENCE_KINDS), default="auto", help="Reference type")
    parser.add_argument("--match-mode", choices=sorted(MATCH_MODES), default="exact", help="Function key matching mode")
    parser.add_argument("--split", default="", help="CSV/reference split: train, validation, or test. Usually inferred")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_final_set, input_pool = resolve_input_pool(Path(args.input))
    reference = Path(args.reference).expanduser().resolve()
    split = normalize_split(args.split) if args.split else infer_split(input_final_set) or infer_split(reference)

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
    print(f"[reference] path={reference}", flush=True)
    print(f"[reference] split={split or 'auto'} kind={args.reference_kind} match_mode={args.match_mode}", flush=True)

    allowed, reference_stats = load_reference_allowed_keys(
        reference,
        input_final_set,
        split,
        args.reference_kind,
        args.match_mode,
        batch_size=50000,
    )
    print(
        f"[reference] kind={reference_stats['reference_kind']} rows={reference_stats['reference_rows']} "
        f"allowed={reference_stats['reference_allowed']} skipped={reference_stats['reference_skipped']} "
        f"duplicates={reference_stats['reference_duplicates']}",
        flush=True,
    )

    dataset = datasets.load_from_disk(str(input_pool))
    if args.match_mode == "origin" and "origin_binary_name" not in dataset.column_names:
        dataset = ensure_metadata_columns(dataset)
    input_key_column(dataset, args.match_mode)
    print(f"[load] rows={len(dataset)} columns={dataset.column_names}", flush=True)

    batch_size = 10000
    progress_every = 1_000_000
    keep_indices, matched_keys = collect_keep_indices(dataset, allowed, args.match_mode, batch_size, progress_every)
    if not keep_indices:
        raise SystemExit("Reference filter matched zero rows; check that the final_set and reference belong to the same split.")
    print(
        f"[filter] kept_rows={len(keep_indices)} matched_unique={len(matched_keys)} "
        f"missing_reference={max(0, len(allowed) - len(matched_keys))}",
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
        "split": split or "",
        "match_mode": args.match_mode,
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
        print("[replace] installing filtered final_set in place", flush=True)
        replace_final_set_in_place(input_final_set, output_root)
    else:
        print(f"[install] moving filtered final_set to {final_output}", flush=True)
        os.replace(output_root, final_output)
    print(f"[done] summary={final_output / 'filter_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
