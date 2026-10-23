"""Build HF datasets from graph records: schema, workers, and publication."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile

import datasets

from .file_processor import FileProcessor


def get_dataset_features():
    """Define the features structure for HuggingFace dataset"""
    return datasets.Features({
        'file_path': datasets.Value('string'),
        'ddg_graph': datasets.Sequence(
            datasets.Sequence(datasets.Value('int32'))
        ),
        'cfg_graph': datasets.Sequence(
            datasets.Sequence(datasets.Value('float32'))
        ),
        'input_ids': datasets.Sequence(datasets.Value('int32')),
    })


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


class DatasetBuilder:
    """Read current graph records and publish only a fully saved dataset."""

    def __init__(self, tokenizer_path: str, db_file: str, num_processes: int | None = None):
        self.tokenizer_path = str(Path(tokenizer_path).resolve())
        self.db_file = str(Path(db_file).resolve())
        self.num_processes = num_processes if num_processes is not None else (os.cpu_count() or 1)
        if self.num_processes < 1:
            raise ValueError("num_processes must be positive")

    def process_dataset(self, output_path: str) -> Path:
        output_directory = Path(output_path).resolve()
        if not Path(self.db_file).is_file():
            raise FileNotFoundError(self.db_file)
        if output_directory.exists() and (
            not output_directory.is_dir() or any(output_directory.iterdir())
        ):
            raise FileExistsError(f"Output must be a new or empty directory: {output_directory}")

        output_directory.parent.mkdir(parents=True, exist_ok=True)
        # A fresh Arrow cache prevents reuse after the database, IR, or tokenizer changes.
        with tempfile.TemporaryDirectory(
            prefix=f".{output_directory.name}-build-", dir=output_directory.parent
        ) as temporary_directory:
            workspace = Path(temporary_directory)
            dataset = create_hf_dataset(
                self.db_file, self.tokenizer_path, self.num_processes, str(workspace / "cache")
            )
            staged_output = workspace / "dataset"
            dataset.save_to_disk(str(staged_output))
            sample_count = len(dataset)
            del dataset
            # Do not overwrite a nonempty dataset, including one created during this build.
            if output_directory.exists():
                output_directory.rmdir()
            os.replace(staged_output, output_directory)

        print(f"Saved {sample_count} samples to {output_directory}")
        return output_directory
