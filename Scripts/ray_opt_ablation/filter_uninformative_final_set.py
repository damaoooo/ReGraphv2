#!/usr/bin/env python3
"""Filter uninformative optimized stubs from a ReGraph final_set.

The filter removes lifted-IR artifacts that are not useful for binary function
similarity, especially when conservative or aggressive re-optimization turns
unrelated functions into identical tiny bodies:

  define void @func0(...) { unreachable }
  define i1 @func0(...) { ret i1 false }

It also removes large exact input_id collision buckets that span many different
function groups after tokenization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import datasets
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Tokenizer.ir_tokenizer import load_tokenizer


DEFAULT_TOKENIZER = REPO_ROOT / "Tokenizer" / "output_tokenizer" / "llvm_ir_bpe.json"
OPT_LEVEL_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<opt>O0|O1|O2|O3|Os|Og|Oz|Oc)_(?P<binary>.+)$")
CONST_RET_RE = re.compile(
    r"\bret\s+"
    r"(?:(?:noundef\s+)?(?:i\d+|ptr|float|double))\s+"
    r"(?:false|true|null|[-+]?\d+(?:\.\d+)?)\b"
)
SEMANTIC_OP_RE = re.compile(
    r"\b("
    r"call|invoke|load|store|cmpxchg|atomicrmw|alloca|"
    r"add|sub|mul|udiv|sdiv|urem|shl|lshr|ashr|and|or|xor|"
    r"icmp|fcmp|getelementptr|select|phi|switch|br"
    r")\b"
)


def hash_ids(input_ids: list[int]) -> str:
    arr = np.asarray(input_ids, dtype=np.int32)
    return f"{len(input_ids)}:{hashlib.blake2b(arr.tobytes(), digest_size=12).hexdigest()}"


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def resolve_input_pool(input_path: Path) -> tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()
    if (input_path / "train_dataset_pool").is_dir():
        return input_path, input_path / "train_dataset_pool"
    if input_path.name == "train_dataset_pool" and (input_path / "dataset_info.json").is_file():
        return input_path.parent, input_path
    raise ValueError("Input must be a final_set directory or its train_dataset_pool directory.")


def derive_origin_and_opt(binary_name: str) -> tuple[str, str]:
    normalized = binary_name.replace("\\", "/")
    directory, _, basename = normalized.rpartition("/")
    match = OPT_LEVEL_PATTERN.match(basename)
    if match:
        origin_basename = f"{match.group('prefix')}_{match.group('binary')}"
        origin = f"{directory}/{origin_basename}" if directory else origin_basename
        return origin, match.group("opt")
    return normalized, ""


def ensure_metadata_columns(dataset: datasets.Dataset) -> datasets.Dataset:
    if "origin_binary_name" in dataset.column_names and "opt_level" in dataset.column_names:
        return dataset
    if "binary_name" not in dataset.column_names:
        raise ValueError("Dataset must contain binary_name when metadata columns are missing.")

    origins: list[str] = []
    opt_levels: list[str] = []
    for batch in dataset.select_columns(["binary_name"]).iter(batch_size=50000):
        for binary_name in batch["binary_name"]:
            origin, opt_level = derive_origin_and_opt(binary_name)
            origins.append(origin)
            opt_levels.append(opt_level)

    if "origin_binary_name" not in dataset.column_names:
        dataset = dataset.add_column("origin_binary_name", origins)
    if "opt_level" not in dataset.column_names:
        dataset = dataset.add_column("opt_level", opt_levels)
    return dataset


def row_group_key(row: dict[str, Any]) -> tuple[str, str]:
    origin = row.get("origin_binary_name") or row.get("binary_name") or row.get("file_path") or ""
    function_name = row.get("function_name") or ""
    return str(origin), str(function_name)


def decoded_body(decoded: str) -> str:
    if "{" not in decoded or "}" not in decoded:
        return decoded
    return decoded.split("{", 1)[-1].rsplit("}", 1)[0]


def is_uninformative_stub(decoded: str, token_len: int, max_tokens: int) -> str | None:
    if token_len > max_tokens:
        return None

    text = decoded.replace("\n", " ")
    padded = f" {text} "
    if " unreachable " in padded:
        return "short_unreachable"

    body = decoded_body(text)
    body_without_ret = body.replace(" ret ", " ")
    if CONST_RET_RE.search(text) and not SEMANTIC_OP_RE.search(body_without_ret):
        return "short_constant_return"
    if " ret void " in padded and not SEMANTIC_OP_RE.search(body_without_ret):
        return "short_ret_void"
    return None


def build_positive_map(dataset: datasets.Dataset, batch_size: int) -> dict[int, list[int]]:
    dataset = ensure_metadata_columns(dataset)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    columns = dataset.select_columns(["origin_binary_name", "function_name"])
    seen = 0
    for batch in columns.iter(batch_size=batch_size):
        for offset, (origin, function_name) in enumerate(zip(batch["origin_binary_name"], batch["function_name"])):
            if origin and function_name:
                groups[(origin, function_name)].append(seen + offset)
        seen += len(batch["function_name"])

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
        "kept_original_indices.npy",
        "dropped_original_indices.npy",
    ]


def replace_final_set_in_place(final_set: Path, staged_final_set: Path) -> None:
    backup_root = final_set.parent / f".{final_set.name}.pre-uninformative-filter-{os.getpid()}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove uninformative stubs from a ReGraph final_set.")
    parser.add_argument("input", help="Input final_set directory or train_dataset_pool")
    parser.add_argument("--output", "-o", default="", help="Optional output final_set. Default replaces input in place.")
    parser.add_argument("--tokenizer", default=os.environ.get("REGRAPH_TOKENIZER_PATH", str(DEFAULT_TOKENIZER)))
    parser.add_argument("--max-stub-tokens", type=int, default=int(os.environ.get("REGRAPH_STUB_MAX_TOKENS", "128")))
    parser.add_argument(
        "--drop-collision-min-rows",
        type=int,
        default=int(os.environ.get("REGRAPH_STUB_COLLISION_MIN_ROWS", "50")),
    )
    parser.add_argument(
        "--drop-collision-min-groups",
        type=int,
        default=int(os.environ.get("REGRAPH_STUB_COLLISION_MIN_GROUPS", "20")),
    )
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_final_set, input_pool = resolve_input_pool(Path(args.input))
    replace_in_place = not bool(args.output)
    final_output = input_final_set if replace_in_place else Path(args.output).expanduser().resolve()

    if args.max_stub_tokens <= 0:
        raise SystemExit("--max-stub-tokens must be positive")
    if args.drop_collision_min_rows <= 1 or args.drop_collision_min_groups <= 1:
        raise SystemExit("--drop-collision-min-rows and --drop-collision-min-groups must be > 1")
    if not replace_in_place:
        if final_output.exists() or final_output.is_symlink():
            if not args.overwrite:
                raise SystemExit(f"Output already exists: {final_output}. Use --overwrite.")
            remove_path(final_output)
        final_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[input] final_set={input_final_set}", flush=True)
    print(f"[input] pool={input_pool}", flush=True)
    print(f"[config] max_stub_tokens={args.max_stub_tokens}", flush=True)
    print(
        f"[config] collision_min_rows={args.drop_collision_min_rows} "
        f"collision_min_groups={args.drop_collision_min_groups}",
        flush=True,
    )

    dataset = datasets.load_from_disk(str(input_pool))
    dataset = ensure_metadata_columns(dataset)
    tokenizer = load_tokenizer(args.tokenizer)

    read_cols = ["input_ids", "function_name"]
    for column in ("binary_name", "origin_binary_name", "opt_level", "file_path"):
        if column in dataset.column_names:
            read_cols.append(column)

    hash_rows: dict[str, list[int]] = defaultdict(list)
    hash_groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    stub_indices: set[int] = set()
    stub_reasons: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    seen = 0
    for batch in dataset.select_columns(read_cols).iter(batch_size=args.batch_size):
        size = len(batch["input_ids"])
        for offset in range(size):
            index = seen + offset
            input_ids = batch["input_ids"][offset]
            row = {column: batch[column][offset] for column in read_cols if column != "input_ids"}
            group_key = row_group_key(row)
            input_hash = hash_ids(input_ids)
            hash_rows[input_hash].append(index)
            hash_groups[input_hash].add(group_key)

            decoded = tokenizer.decode(input_ids)
            reason = is_uninformative_stub(decoded, len(input_ids), args.max_stub_tokens)
            if reason:
                stub_indices.add(index)
                stub_reasons[reason] += 1
                if len(examples[reason]) < 10:
                    examples[reason].append(
                        {
                            "idx": index,
                            "binary_name": row.get("binary_name"),
                            "function_name": row.get("function_name"),
                            "opt_level": row.get("opt_level"),
                            "tokens": len(input_ids),
                            "decoded": decoded,
                        }
                    )
        seen += size
        if seen and seen % 1_000_000 == 0:
            print(f"[scan] rows={seen} stubs={len(stub_indices)}", flush=True)

    collision_indices: set[int] = set()
    collision_buckets: list[dict[str, Any]] = []
    for input_hash, indices in hash_rows.items():
        group_count = len(hash_groups[input_hash])
        if len(indices) >= args.drop_collision_min_rows and group_count >= args.drop_collision_min_groups:
            collision_indices.update(indices)
            collision_buckets.append({"hash": input_hash, "rows": len(indices), "groups": group_count})
    collision_buckets.sort(key=lambda item: item["rows"], reverse=True)

    dropped_indices = stub_indices | collision_indices
    keep_indices = [index for index in range(len(dataset)) if index not in dropped_indices]
    summary = {
        "input_final_set": str(input_final_set),
        "input_pool": str(input_pool),
        "output_final_set": str(final_output),
        "replace_in_place": replace_in_place,
        "input_rows": len(dataset),
        "kept_rows": len(keep_indices),
        "dropped_rows": len(dropped_indices),
        "dropped_pct": 0.0 if len(dataset) == 0 else 100.0 * len(dropped_indices) / len(dataset),
        "stub_rows": len(stub_indices),
        "stub_reasons": dict(stub_reasons),
        "collision_rows": len(collision_indices),
        "collision_bucket_count": len(collision_buckets),
        "collision_threshold": {
            "min_rows": args.drop_collision_min_rows,
            "min_groups": args.drop_collision_min_groups,
        },
        "top_collision_buckets": collision_buckets[:20],
        "examples": examples,
        "max_stub_tokens": args.max_stub_tokens,
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)
        return 0

    output_root = final_output.parent / f".{final_output.name}.uninformative-filtering-{os.getpid()}"
    remove_path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        filtered_pool = dataset.select(keep_indices)
        filtered_pool.save_to_disk(str(output_root / "train_dataset_pool"))
        positive_map = build_positive_map(filtered_pool, args.batch_size)
        with (output_root / "train_positive_map.pkl").open("wb") as handle:
            pickle.dump(positive_map, handle)
        anchor_indices = sorted(positive_map)
        datasets.Dataset.from_dict({"anchor_idx": anchor_indices}).save_to_disk(str(output_root / "train_task_dataset"))
        save_empty_validation_outputs(output_root, filtered_pool)
        write_split_indices(output_root, len(filtered_pool))
        np.save(output_root / "kept_original_indices.npy", np.asarray(keep_indices, dtype=np.int64))
        np.save(output_root / "dropped_original_indices.npy", np.asarray(sorted(dropped_indices), dtype=np.int64))

        summary["positive_map_entries"] = len(positive_map)
        summary["task_anchors"] = len(anchor_indices)
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
    except Exception:
        remove_path(output_root)
        raise

    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)
    print(f"[done] summary={final_output / 'filter_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
