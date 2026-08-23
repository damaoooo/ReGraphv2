#!/usr/bin/env python3
"""Remove author-specific paths from a staged ReLL artifact package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_from_disk


REPLACEMENTS = (
    ("/home/damaoooo/Downloads/regraphv2", "/artifact/rell"),
    ("/home/damaoooo/Downloads/ReGraphv2", "/artifact/rell"),
    ("/home/damaoooo", "/home/anonymous"),
    ("/scratch/zhoul0e", "/scratch/anonymous"),
    ("/scratch/damaoooo", "/scratch/anonymous"),
)

IDENTITY_MARKERS = (b"damaoooo", b"zhoul0e", b"@kaust")
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


def sanitize_text(value: str) -> str:
    for source, replacement in REPLACEMENTS:
        value = value.replace(source, replacement)
    value = value.replace("damaoooo", "anonymous")
    value = value.replace("zhoul0e", "anonymous")
    return value


def sanitize_dataset(dataset_dir: Path) -> None:
    # Empty validation splits have metadata but no Arrow shard.  Their text
    # metadata is sanitized later, and there is no binary payload to rewrite.
    if not any(dataset_dir.glob("*.arrow")):
        return

    dataset = load_from_disk(str(dataset_dir))
    if "file_path" in dataset.column_names:
        dataset = dataset.map(
            lambda batch: {
                "file_path": [sanitize_text(value) for value in batch["file_path"]]
            },
            batched=True,
            batch_size=4096,
            desc=f"Sanitizing {dataset_dir.name}",
        )

    sanitized_dir = dataset_dir.with_name(f"{dataset_dir.name}.sanitized")
    if sanitized_dir.exists():
        shutil.rmtree(sanitized_dir)
    dataset.save_to_disk(str(sanitized_dir))
    shutil.rmtree(dataset_dir)
    sanitized_dir.rename(dataset_dir)


def sanitize_text_files(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        original = path.read_text(encoding="utf-8", errors="strict")
        sanitized = sanitize_text(original)
        if sanitized != original:
            path.write_text(sanitized, encoding="utf-8")


def contains_marker(path: Path) -> bool:
    overlap = max(len(marker) for marker in IDENTITY_MARKERS) - 1
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            data = tail + chunk
            if any(marker in data.lower() for marker in IDENTITY_MARKERS):
                return True
            tail = data[-overlap:]
    return False


def verify_anonymity(root: Path) -> None:
    leaked = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and contains_marker(path)]
    if leaked:
        formatted = "\n".join(f"  - {path}" for path in leaked)
        raise RuntimeError(f"identity markers remain in staged artifact:\n{formatted}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="staged artifact directory")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"staged artifact directory does not exist: {root}")

    dataset_dirs = sorted(
        path
        for path in root.rglob("*")
        if path.is_dir()
        and path.name in {"train_dataset_pool", "validation_dataset_pool"}
        and (path / "state.json").is_file()
    )
    for dataset_dir in dataset_dirs:
        sanitize_dataset(dataset_dir)

    sanitize_text_files(root)
    verify_anonymity(root)
    print(f"[INFO] Sanitized {len(dataset_dirs)} Hugging Face dataset directories.")
    print("[INFO] No author identity markers remain in the staged artifact.")


if __name__ == "__main__":
    main()
