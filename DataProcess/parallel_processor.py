"""One bounded, deterministic processing path for Hugging Face datasets."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sqlite3

import datasets

from .dataset_features import get_dataset_features
from .file_processor import FileProcessor

CHUNK_SIZE = 400
_worker_processor = None


def initialize_worker(tokenizer_path: str) -> None:
    """Load the tokenizer once per worker; graph artifacts remain reusable."""
    from Tokenizer.ir_tokenizer import load_tokenizer

    global _worker_processor
    _worker_processor = FileProcessor(load_tokenizer(tokenizer_path), cleanup_temp_files=False)


def process_chunk(rows: list[tuple]) -> tuple[list[dict], list[str]]:
    """Return valid samples and failed paths without treating empty graphs as errors."""
    if _worker_processor is None:
        raise RuntimeError("Worker has not been initialized")
    samples, failed_paths = [], []
    for row in rows:
        if len(row) != 5 or not all(row):
            raise ValueError(f"Incomplete graph artifact record: {row!r}")
        result = _worker_processor.process_single_file(*row)
        if result.success:
            samples.append(result.to_dict())
        else:
            failed_paths.append(row[0])
    return samples, failed_paths


def iter_database_chunks(db_path: str):
    """Stream a stable database order instead of caching file lists or using large IN queries."""
    database_uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        cursor = connection.execute(
            "SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot "
            "FROM results ORDER BY input_path"
        )
        while rows := cursor.fetchmany(CHUNK_SIZE):
            yield rows


def iter_processed_chunks(db_path: str, tokenizer_path: str, num_processes: int):
    chunks = iter_database_chunks(db_path)
    if num_processes == 1:
        initialize_worker(tokenizer_path)
        for rows in chunks:
            yield process_chunk(rows)
        return

    with ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=initialize_worker,
        initargs=(tokenizer_path,),
        mp_context=multiprocessing.get_context('spawn'),
    ) as executor:
        pending = deque()
        for rows in chunks:
            pending.append(executor.submit(process_chunk, rows))
            if len(pending) >= 2 * num_processes:
                # Submission order keeps dataset indices independent of worker scheduling.
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def create_hf_dataset(
    db_path: str, tokenizer_path: str, num_processes: int, cache_directory: str
) -> datasets.Dataset:
    """Build typed Arrow data; task failures propagate and bad samples are counted."""
    def generate_samples():
        successful_count = 0
        failed_count = 0
        for samples, failed_paths in iter_processed_chunks(db_path, tokenizer_path, num_processes):
            failed_count += len(failed_paths)
            if failed_paths:
                print(f"Skipped {len(failed_paths)} invalid samples; first: {failed_paths[0]}", flush=True)
            for sample in samples:
                successful_count += 1
                yield sample
        print(f"Dataset samples: {successful_count} valid, {failed_count} invalid", flush=True)
        if successful_count == 0:
            raise ValueError("No valid samples were produced")

    return datasets.Dataset.from_generator(
        generate_samples, features=get_dataset_features(), cache_dir=cache_directory
    )
