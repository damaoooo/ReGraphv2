"""
Parallel processing utilities for ASM dataset building
"""
import logging
import multiprocessing
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List

import datasets
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from .dataset_features import get_dataset_features
from .file_processor import FileProcessor
from .pd_writer import PDWriter
from .processing_result import ProcessingResult

console = Console()


def process_single_file_standalone(
    input_path: str,
    tokenizer: PreTrainedTokenizerFast,
    cleanup_temp_files: bool = True,
) -> ProcessingResult:
    """Standalone version of process_single_file."""
    try:
        processor = FileProcessor(
            tokenizer=tokenizer,
            cleanup_temp_files=cleanup_temp_files,
        )
        return processor.process_single_file(input_file=input_path)
    except Exception as exc:
        return ProcessingResult(
            file_path=input_path,
            success=False,
            error_message=str(exc),
        )


def process_chunk_standalone(
    file_paths: List[str],
    tokenizer_path: str,
    cleanup_temp_files: bool = True,
) -> List[ProcessingResult]:
    """Process a chunk of ASM files in a standalone worker."""
    from Tokenizer.ir_tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)
    return [
        process_single_file_standalone(
            input_path=file_path,
            tokenizer=tokenizer,
            cleanup_temp_files=cleanup_temp_files,
        )
        for file_path in file_paths
    ]


def process_chunk_to_queue(
    file_paths: List[str],
    tokenizer_path: str,
    cleanup_temp_files: bool,
    queue: multiprocessing.Queue,
) -> List[int]:
    """Process a chunk and stream successful rows to the parquet writer."""
    results = process_chunk_standalone(
        file_paths=file_paths,
        tokenizer_path=tokenizer_path,
        cleanup_temp_files=cleanup_temp_files,
    )
    success = sum(1 for result in results if result.success)
    successful_rows = [result.to_dict() for result in results if result.success]
    if successful_rows:
        queue.put(successful_rows)
    return [success, len(results)]


class ParallelProcessor:
    """Handles sequential, batched, and streaming-parallel file processing."""

    def __init__(self, num_processes: int):
        self.num_processes = num_processes
        self.logger = logging.getLogger(__name__)

    def process_files_sequential(
        self,
        file_processor: FileProcessor,
        file_paths: List[str],
    ) -> List[ProcessingResult]:
        """Process files sequentially for debugging."""
        console.print("[yellow]Using sequential processing for debugging")
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[green]Processing {len(file_paths)} files sequentially",
                total=len(file_paths),
            )

            for index, file_path in enumerate(file_paths):
                try:
                    result = file_processor.process_single_file(file_path)
                    results.append(result)

                    if result.success:
                        self.logger.info(
                            f"Successfully processed ({index + 1}/{len(file_paths)}): {file_path}"
                        )
                    else:
                        self.logger.error(
                            f"Failed to process ({index + 1}/{len(file_paths)}): "
                            f"{file_path} - {result.error_message}"
                        )
                except Exception as exc:
                    self.logger.error(f"Exception processing {file_path}: {exc}")
                    results.append(
                        ProcessingResult(
                            file_path=file_path,
                            success=False,
                            error_message=str(exc),
                        )
                    )
                finally:
                    progress.update(task, advance=1)

        return results

    def process_files_batch(
        self,
        file_processor: FileProcessor,
        file_paths: List[str],
        batch_size: int = 1000,
    ) -> List[ProcessingResult]:
        """Process files in batches with a thread pool."""
        all_results = []
        total_batches = (len(file_paths) + batch_size - 1) // batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task(
                f"[green]Processing {len(file_paths)} files in {total_batches} batches",
                total=len(file_paths),
            )

            for index in range(0, len(file_paths), batch_size):
                batch = file_paths[index : index + batch_size]
                batch_num = index // batch_size + 1
                batch_task = progress.add_task(
                    f"[blue]Batch {batch_num}/{total_batches}",
                    total=len(batch),
                )

                with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                    futures = {
                        executor.submit(file_processor.process_single_file, file_path): file_path
                        for file_path in batch
                    }

                    batch_results = []
                    for future in as_completed(futures):
                        file_path = futures[future]
                        try:
                            batch_results.append(future.result())
                        except Exception as exc:
                            self.logger.error(f"Error processing {file_path}: {exc}")
                            batch_results.append(
                                ProcessingResult(
                                    file_path=file_path,
                                    success=False,
                                    error_message=str(exc),
                                )
                            )
                        finally:
                            progress.update(batch_task, advance=1)
                            progress.update(overall_task, advance=1)

                all_results.extend(batch_results)
                successful = sum(1 for result in batch_results if result.success)
                progress.update(
                    batch_task,
                    description=(
                        f"[green]Batch {batch_num} completed: "
                        f"{successful}/{len(batch)} successful"
                    ),
                )

        return all_results

    def process_files_parallel(
        self,
        file_paths: List[str],
        tokenizer_path: str,
        cleanup_temp_files: bool,
        output_path: str,
        start_index: int,
    ) -> List[int]:
        """Stream processed rows into parquet files with worker processes."""
        if not file_paths:
            console.print("[yellow]No ASM files left to process")
            return [0, 0]

        console.print(f"[yellow]Using parallel processing with {self.num_processes} workers")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            success_total = 0
            task = progress.add_task(
                f"[green]Processing {len(file_paths)} files",
                total=len(file_paths),
            )

            chunk_size = 500
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()
            pd_writer = PDWriter(
                queue=result_queue,
                output_path=output_path,
                bin_size=chunk_size * 20,
                start_index=start_index,
            )
            pd_process = multiprocessing.Process(target=pd_writer.start)
            pd_process.start()

            try:
                with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                    future_to_files = {}
                    for index in range(0, len(file_paths), chunk_size):
                        chunk = file_paths[index : index + chunk_size]
                        future = executor.submit(
                            process_chunk_to_queue,
                            chunk,
                            tokenizer_path,
                            cleanup_temp_files,
                            result_queue,
                        )
                        future_to_files[future] = chunk

                    for future in as_completed(future_to_files):
                        chunk = future_to_files[future]
                        try:
                            success, total = future.result()
                            success_total += success
                            progress.update(task, advance=total)
                        except Exception as exc:
                            self.logger.error(f"Error processing chunk: {exc}")
                            progress.update(task, advance=len(chunk))
                        finally:
                            future_to_files.pop(future, None)
            finally:
                result_queue.put("STOP")
                pd_process.join()
                manager.shutdown()

        console.print(f"[green]Parallel processing completed. Results written to {output_path}")
        return [success_total, len(file_paths)]


def create_hf_dataset_from_files(
    file_paths: List[str],
    tokenizer_path: str,
    num_processes: int,
    cleanup_temp_files: bool = True,
) -> "datasets.Dataset":
    """Create a Hugging Face dataset from ASM files."""

    def standalone_generator():
        print(
            f"Starting parallel processing generator for HF with {num_processes} workers",
            file=sys.stderr,
        )

        chunk_size = 400

        with tqdm(
            total=len(file_paths),
            desc="Processing files for HF dataset",
            file=sys.stderr,
            dynamic_ncols=True,
        ) as progress_bar:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                future_to_files = {}
                for index in range(0, len(file_paths), chunk_size):
                    chunk = file_paths[index : index + chunk_size]
                    future = executor.submit(
                        process_chunk_standalone,
                        chunk,
                        tokenizer_path,
                        cleanup_temp_files,
                    )
                    future_to_files[future] = chunk

                processed_count = 0
                total_files = len(file_paths)
                for future in as_completed(future_to_files):
                    chunk = future_to_files[future]
                    try:
                        chunk_results = future.result()
                        for result in chunk_results:
                            if result.success:
                                yield result.to_dict()
                        processed_count += len(chunk_results)
                        progress_bar.update(len(chunk_results))
                        progress_bar.set_postfix(
                            {"processed": processed_count, "total": total_files}
                        )
                    except Exception as exc:
                        print("Worker chunk failed while building HF dataset", file=sys.stderr)
                        print(f"Chunk: {chunk}", file=sys.stderr)
                        print(f"Error: {exc}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        progress_bar.update(len(chunk))

    return datasets.Dataset.from_generator(
        standalone_generator,
        features=get_dataset_features(),
    )
