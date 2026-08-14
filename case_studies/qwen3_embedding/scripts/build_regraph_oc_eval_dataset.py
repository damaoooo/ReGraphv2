#!/usr/bin/env python3
"""Build a ReLL/Qwen evaluation dataset from a regraphv2 OC final_set.

The regraphv2 final_set stores LLVM IR as tokenizer input_ids. ReLL's Qwen
evaluator expects a Hugging Face dataset with a text column, so this script
decodes the OC input_ids back to token-space LLVM IR text and copies the
positive map alongside it.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-final-set",
        required=True,
        type=Path,
        help="Path to a regraphv2 final_set directory containing train_dataset_pool and train_positive_map.pkl.",
    )
    parser.add_argument(
        "--regraph-root",
        required=True,
        type=Path,
        help="Path to the regraphv2 repository root, used to import Tokenizer.ir_tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Path to Tokenizer/output_tokenizer/llvm_ir_bpe.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for the ReLL-format dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    return parser.parse_args()


def load_regraph_tokenizer(regraph_root: Path, tokenizer_path: Path):
    sys.path.insert(0, str(regraph_root.resolve()))
    from Tokenizer.ir_tokenizer import load_tokenizer  # noqa: PLC0415

    return load_tokenizer(str(tokenizer_path.resolve()))


def main() -> None:
    args = parse_args()
    source_final_set = args.source_final_set.resolve()
    source_pool = source_final_set / "train_dataset_pool"
    source_positive_map = source_final_set / "train_positive_map.pkl"
    output_dir = args.output.resolve()
    output_pool = output_dir / "train_dataset_pool"
    output_positive_map = output_dir / "train_positive_map.pkl"

    if not source_pool.exists():
        raise SystemExit(f"Missing source dataset pool: {source_pool}")
    if not source_positive_map.exists():
        raise SystemExit(f"Missing source positive map: {source_positive_map}")
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise SystemExit(f"Output already exists, use --force to rebuild: {output_dir}")

    tokenizer = load_regraph_tokenizer(args.regraph_root, args.tokenizer)
    dataset = load_from_disk(str(source_pool))

    required_columns = {
        "input_ids",
        "binary_name",
        "file_path",
        "function_name",
        "opt_level",
        "origin_binary_name",
        "original_idx",
    }
    missing_columns = sorted(required_columns - set(dataset.column_names))
    if missing_columns:
        raise SystemExit(f"Source dataset is missing required columns: {missing_columns}")

    def decode_batch(input_ids, binary_name, file_path, function_name, opt_level, origin_binary_name, original_idx):
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        file_names = []
        for path_value, binary_value in zip(file_path, binary_name):
            raw_name = str(path_value).split("::", 1)[0]
            name = Path(raw_name).name if raw_name else Path(str(binary_value)).name
            file_names.append(name)

        return {
            "text": texts,
            "token_len": [len(ids) for ids in input_ids],
            "file_name": file_names,
            "function_name": [str(value) for value in function_name],
            "binary_name": [str(value) for value in binary_name],
            "origin_binary_name": [str(value) for value in origin_binary_name],
            "opt_level": [str(value) for value in opt_level],
            "original_idx": [int(value) for value in original_idx],
        }

    decoded = dataset.map(
        decode_batch,
        batched=True,
        batch_size=args.batch_size,
        input_columns=[
            "input_ids",
            "binary_name",
            "file_path",
            "function_name",
            "opt_level",
            "origin_binary_name",
            "original_idx",
        ],
        remove_columns=dataset.column_names,
        desc="Decoding OC input_ids to LLVM IR text",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    decoded.save_to_disk(str(output_pool))
    shutil.copy2(source_positive_map, output_positive_map)

    with source_positive_map.open("rb") as handle:
        positive_map = pickle.load(handle)
    max_index = max([max(map(int, positives)) for positives in positive_map.values() if positives] + [0])
    max_index = max(max_index, max(map(int, positive_map.keys()), default=0))
    if max_index >= len(decoded):
        raise SystemExit(
            f"Positive map references index {max_index}, but decoded dataset has only {len(decoded)} rows."
        )

    print(f"Wrote dataset: {output_pool}")
    print(f"Wrote positive map: {output_positive_map}")
    print(f"Rows: {len(decoded):,}")
    print(f"Anchors: {len(positive_map):,}")


if __name__ == "__main__":
    main()
