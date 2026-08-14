#!/usr/bin/env python3
"""Add raw ASM text to an existing final_set without changing its rows or map."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-final-set", type=Path, required=True)
    parser.add_argument("--output-final-set", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--read-batch-size", type=int, default=256)
    parser.add_argument("--save-num-proc", type=int, default=8)
    parser.add_argument("--max-shard-size", default="1GB")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_asm_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
    texts: list[str] = []
    for file_path in batch["file_path"]:
        texts.append(
            Path(file_path).read_text(encoding="utf-8", errors="replace")
        )
    return {"text": texts}


def output_is_complete(output_final_set: Path) -> bool:
    return (
        (output_final_set / "train_dataset_pool").is_dir()
        and (output_final_set / "train_positive_map.pkl").is_file()
        and (output_final_set / "build_summary.json").is_file()
    )


def validate_args(args: argparse.Namespace) -> None:
    required = [
        args.input_final_set / "train_dataset_pool",
        args.input_final_set / "train_positive_map.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required input(s):\n" + "\n".join(missing))
    if args.num_proc <= 0 or args.save_num_proc <= 0:
        raise SystemExit("Process counts must be positive")
    if args.read_batch_size <= 0:
        raise SystemExit("--read-batch-size must be positive")


def build_text_pool(
    source_pool: Dataset,
    work_dir: Path,
    num_proc: int,
    batch_size: int,
) -> Dataset:
    if "file_path" not in source_pool.column_names:
        raise SystemExit(
            f"Source pool has no file_path column: {source_pool.column_names}"
        )

    removable = [
        column
        for column in ("input_ids", "text")
        if column in source_pool.column_names
    ]
    metadata_pool = (
        source_pool.remove_columns(removable) if removable else source_pool
    )
    cache_file = work_dir / "asm_text.arrow"
    return metadata_pool.map(
        read_asm_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        cache_file_name=str(cache_file),
        suffix_template="_{rank:05d}_of_{num_proc:05d}",
        load_from_cache_file=True,
        keep_in_memory=False,
        desc="Reading raw ASM text",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_final_set = args.input_final_set.resolve()
    output_final_set = args.output_final_set.resolve()
    if output_is_complete(output_final_set):
        print(f"Output is already complete: {output_final_set}")
        return

    work_dir = output_final_set.parent / f".{output_final_set.name}.build"
    temporary_output = output_final_set.parent / f".{output_final_set.name}.tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    source_pool = load_from_disk(str(input_final_set / "train_dataset_pool"))
    source_rows = len(source_pool)
    print(
        f"Source pool: {source_rows:,} rows, columns={source_pool.column_names}",
        flush=True,
    )
    text_pool = build_text_pool(
        source_pool=source_pool,
        work_dir=work_dir,
        num_proc=args.num_proc,
        batch_size=args.read_batch_size,
    )
    if len(text_pool) != source_rows:
        raise RuntimeError(
            f"Row count changed from {source_rows:,} to {len(text_pool):,}"
        )

    for idx in {0, source_rows // 2, source_rows - 1}:
        row = text_pool[idx]
        raw_text = Path(row["file_path"]).read_text(
            encoding="utf-8", errors="replace"
        )
        if row["text"] != raw_text:
            raise RuntimeError(f"ASM text mismatch at row {idx}: {row['file_path']}")

    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True)
    text_pool.save_to_disk(
        str(temporary_output / "train_dataset_pool"),
        max_shard_size=args.max_shard_size,
        num_proc=args.save_num_proc,
    )

    source_map = input_final_set / "train_positive_map.pkl"
    output_map = temporary_output / "train_positive_map.pkl"
    shutil.copy2(source_map, output_map)
    source_map_hash = sha256_file(source_map)
    output_map_hash = sha256_file(output_map)
    if source_map_hash != output_map_hash:
        raise RuntimeError("Copied positive map failed SHA-256 verification")

    summary = {
        "input_final_set": str(input_final_set),
        "output_final_set": str(output_final_set),
        "rows": source_rows,
        "source_columns": source_pool.column_names,
        "output_columns": text_pool.column_names,
        "positive_map_sha256": source_map_hash,
        "preserved_row_order": True,
        "raw_text_exact": True,
        "filtering_applied": False,
    }
    (temporary_output / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if output_final_set.exists():
        shutil.rmtree(output_final_set)
    temporary_output.rename(output_final_set)
    shutil.rmtree(work_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote exact ASM Qwen final_set: {output_final_set}")


if __name__ == "__main__":
    main()
