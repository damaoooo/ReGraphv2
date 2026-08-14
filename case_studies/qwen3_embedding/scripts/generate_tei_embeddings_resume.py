#!/usr/bin/env python3
"""Generate TEI embeddings with an on-disk resumable NumPy cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import requests
from datasets import load_from_disk
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_pool", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Final .npy embedding cache path.")
    parser.add_argument("--tei-endpoint", required=True)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--instruction", default="Represent this LLVM IR for searching for similar functions:")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N batches.")
    parser.add_argument("--force", action="store_true", help="Remove existing final/temp cache and start over.")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def post_with_retries(endpoint: str, inputs: list[str], timeout: int, max_retries: int, base_delay: float) -> np.ndarray:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                endpoint.rstrip("/") + "/embed",
                json={"inputs": inputs},
                timeout=(10, timeout),
            )
            response.raise_for_status()
            return np.asarray(response.json(), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_seconds = min(base_delay * (2 ** (attempt - 1)), 60.0) + random.random()
            print(
                f"Batch request failed ({type(exc).__name__}: {exc}); "
                f"retry {attempt}/{max_retries} after {sleep_seconds:.1f}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"TEI request failed after {max_retries} attempts") from last_exc


def encode_batch(tokenizer, instruction: str, texts: list[str], max_length: int) -> list[str]:
    instructed = [instruction + text for text in texts]
    truncated = tokenizer(instructed, truncation=True, max_length=max_length, padding=False)
    return tokenizer.batch_decode(truncated["input_ids"], skip_special_tokens=True)


def load_state(state_path: Path) -> dict | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.npy")
    state_path = output_path.with_suffix(output_path.suffix + ".progress.json")

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.force:
        for path in (output_path, tmp_path, state_path):
            if path.exists():
                path.unlink()
    if output_path.exists():
        print(f"Embedding cache already exists: {output_path}")
        return

    dataset = load_from_disk(str(args.dataset_pool))
    total = len(dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    embeddings = None
    next_index = 0
    dim = None

    if state is not None:
        if int(state["total"]) != total:
            raise SystemExit(f"State total {state['total']} does not match dataset size {total}")
        next_index = int(state["next_index"])
        dim = int(state["dim"])
        embeddings = np.lib.format.open_memmap(tmp_path, mode="r+")
        if embeddings.shape != (total, dim):
            raise SystemExit(f"Temp cache shape {embeddings.shape} does not match state {(total, dim)}")
        print(f"Resuming {tmp_path}: next_index={next_index:,}, dim={dim}", flush=True)

    start_time = time.time()
    processed_at_start = next_index
    total_batches = math.ceil(total / args.batch_size)

    while next_index < total:
        end_index = min(next_index + args.batch_size, total)
        batch_texts = dataset[next_index:end_index]["text"]
        payload_texts = encode_batch(tokenizer, args.instruction, batch_texts, args.max_length)
        batch_embeddings = post_with_retries(
            args.tei_endpoint,
            payload_texts,
            args.timeout,
            args.max_retries,
            args.retry_base_delay,
        )

        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(batch_texts):
            raise RuntimeError(
                f"Bad embedding shape for rows {next_index}:{end_index}: "
                f"expected ({len(batch_texts)}, dim), got {batch_embeddings.shape}"
            )

        if embeddings is None:
            dim = int(batch_embeddings.shape[1])
            embeddings = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(total, dim))
            atomic_write_json(
                state_path,
                {
                    "total": total,
                    "dim": dim,
                    "next_index": 0,
                    "batch_size": args.batch_size,
                    "tmp_path": str(tmp_path),
                    "output_path": str(output_path),
                },
            )
        elif batch_embeddings.shape[1] != dim:
            raise RuntimeError(f"Embedding dim changed from {dim} to {batch_embeddings.shape[1]}")

        embeddings[next_index:end_index] = batch_embeddings
        embeddings.flush()
        next_index = end_index
        atomic_write_json(
            state_path,
            {
                "total": total,
                "dim": dim,
                "next_index": next_index,
                "batch_size": args.batch_size,
                "tmp_path": str(tmp_path),
                "output_path": str(output_path),
            },
        )

        batch_number = math.ceil(next_index / args.batch_size)
        if batch_number == 1 or batch_number % args.progress_every == 0 or next_index == total:
            elapsed = max(time.time() - start_time, 1e-6)
            processed_now = next_index - processed_at_start
            speed = processed_now / elapsed
            remaining = (total - next_index) / speed if speed > 0 else float("inf")
            print(
                f"embedded {next_index:,}/{total:,} "
                f"batch {batch_number:,}/{total_batches:,} "
                f"speed={speed:.2f} seq/s eta={remaining/60:.1f} min",
                flush=True,
            )

    if embeddings is None:
        raise RuntimeError("No embeddings generated")
    embeddings.flush()
    del embeddings
    os.replace(tmp_path, output_path)
    if state_path.exists():
        state_path.unlink()
    print(f"Wrote embedding cache: {output_path}", flush=True)


if __name__ == "__main__":
    main()
