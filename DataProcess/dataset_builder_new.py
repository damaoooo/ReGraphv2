"""
Main dataset builder class for ASM files
"""
import json
import logging
import os
import pickle
import time
from typing import List, Union

import datasets
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from transformers import PreTrainedTokenizerFast

from .dataset_utils import find_asm_files
from .file_processor import FileProcessor
from .parallel_processor import ParallelProcessor, create_hf_dataset_from_files
from .processing_result import ProcessingResult

console = Console()


class DatasetBuilder:
    """Main class for building datasets from ASM files."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        tokenizer_path: str,
        input_dir: str,
        num_processes: int = None,
        cleanup_temp_files: bool = True,
        cache: bool = True,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.input_dir = os.path.abspath(input_dir)
        self.num_processes = num_processes or (os.cpu_count() or 1)
        self.cleanup_temp_files = cleanup_temp_files
        self.cache = cache
        self._setup_logging()

        self.file_processor = FileProcessor(
            tokenizer=tokenizer,
            cleanup_temp_files=cleanup_temp_files,
        )
        self.parallel_processor = ParallelProcessor(num_processes=self.num_processes)

    def __getstate__(self):
        """Exclude logger and rich objects from pickling."""
        state = self.__dict__.copy()
        state.pop("logger", None)
        return state

    def __setstate__(self, state):
        """Recreate the logger in child processes."""
        self.__dict__.update(state)
        self._setup_logging(rich=False)

    def _setup_logging(self, rich: bool = True):
        """Setup logging configuration."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if rich:
            handlers = [
                RichHandler(console=console, rich_tracebacks=True),
                logging.FileHandler("dataset_builder.log"),
            ]
        else:
            handlers = [logging.FileHandler("dataset_builder_debug.log")]

        logging.basicConfig(
            level=logging.INFO,
            format="%(processName)s: %(message)s",
            datefmt="[%X]",
            handlers=handlers,
        )
        self.logger = logging.getLogger(__name__)

    def process_dataset(
        self,
        output_path: str,
        batch_size: int = 1000,
        use_parallel: bool = False,
        skip_filtering: bool = False,
        use_hf: bool = False,
    ):
        """Process all ASM files under the input directory."""
        console.print(
            f"[yellow]Using {self.num_processes} processes, batch size: {batch_size}"
        )

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            console.print(f"[green]Created output directory: {output_path}")
        else:
            console.print(f"[yellow]Output directory already exists: {output_path}")

        console.print("[yellow]Loading ASM file names...")

        file_cache = os.path.join(output_path, "file_list_cache.pkl")
        if self.cache and os.path.exists(file_cache):
            console.print(f"[yellow]Loading file cache from {file_cache}")
            with open(file_cache, "rb") as handle:
                filtered_files = pickle.load(handle)
        else:
            start_time = time.time()
            filtered_files = find_asm_files(self.input_dir)
            elapsed = time.time() - start_time
            console.print(f"[green]File scan completed in {elapsed:.2f} seconds")
            console.print(f"[green]ASM files found: {len(filtered_files)}")
            if self.cache:
                console.print(f"[yellow]Saving file cache to {file_cache}")
                with open(file_cache, "wb") as handle:
                    pickle.dump(filtered_files, handle)

        if not filtered_files:
            console.print(f"[red]No ASM files found under {self.input_dir}")
            return []

        console.print("[yellow]Checking for resume file...")
        processed_files_set = set()
        resume_file = os.path.join(output_path, "progress.txt")
        if os.path.exists(resume_file) and not skip_filtering:
            console.print("[yellow]Loading processed files from resume file...")
            with open(resume_file, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    processed_files_set.add(line.strip())

            original_count = len(filtered_files)
            filtered_files = [path for path in filtered_files if path not in processed_files_set]
            processed_count = original_count - len(filtered_files)
            console.print(
                "[yellow]Resuming from "
                f"{processed_count} previously processed files, {len(filtered_files)} files remaining"
            )
        else:
            console.print("[yellow]No resume file found, starting fresh")

        if use_hf:
            console.print("[bold blue]Using Hugging Face datasets from generator.")
            start_time = time.time()
            dataset = create_hf_dataset_from_files(
                file_paths=filtered_files,
                tokenizer_path=self.tokenizer_path,
                num_processes=self.num_processes,
                cleanup_temp_files=self.cleanup_temp_files,
            )
            end_time = time.time()

            console.print(
                f"[green]Dataset creation from generator completed in {end_time - start_time:.2f} seconds."
            )
            self.save_results(dataset, output_path, use_hf=True)

            table = Table(title="Processing Summary (Hugging Face Dataset)")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Successfully processed files", str(len(dataset)))
            table.add_row("Processing time", f"{end_time - start_time:.2f} seconds")
            if len(dataset) > 0:
                table.add_row(
                    "Average time per file",
                    f"{(end_time - start_time) / len(dataset):.4f} seconds",
                )

            console.print(table)
            return dataset

        if self.num_processes == 1:
            console.print("[yellow]Processing method: Sequential (debug mode)")
            start_time = time.time()
            results = self.parallel_processor.process_files_sequential(
                self.file_processor,
                filtered_files,
            )
            end_time = time.time()
        else:
            console.print(
                f"[yellow]Processing method: {'Parallel chunks' if use_parallel else 'Batched'}"
            )
            start_time = time.time()
            if use_parallel:
                results = self.parallel_processor.process_files_parallel(
                    file_paths=filtered_files,
                    tokenizer_path=self.tokenizer_path,
                    cleanup_temp_files=self.cleanup_temp_files,
                    output_path=output_path,
                    start_index=len(processed_files_set),
                )
            else:
                results = self.parallel_processor.process_files_batch(
                    self.file_processor,
                    filtered_files,
                    batch_size=batch_size,
                )
            end_time = time.time()

        successful_count = results[0] if use_parallel else sum(
            1 for result in results if result.success
        )

        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Files processed", str(len(filtered_files)))
        if len(filtered_files) > 0:
            table.add_row("Successful", str(successful_count))
            table.add_row("Failed", str(len(filtered_files) - successful_count))
            table.add_row("Processing time", f"{end_time - start_time:.2f} seconds")
            table.add_row(
                "Average time per file",
                f"{(end_time - start_time) / len(filtered_files):.4f} seconds",
            )

        console.print(table)
        if not use_parallel:
            self.save_results(results, output_path)
        return results

    def save_results(
        self,
        results: Union[List[ProcessingResult], "datasets.Dataset"],
        output_path: str,
        use_hf: bool = False,
    ):
        """Save summary results or a Hugging Face dataset."""
        if use_hf:
            console.print(f"[yellow]Saving Hugging Face dataset to: {output_path}")
            try:
                results.save_to_disk(output_path)
                console.print(f"[green]Dataset saved successfully to {output_path}")
            except Exception as exc:
                console.print(f"[red]Error saving Hugging Face dataset: {exc}")
            return

        console.print(f"[yellow]Saving results to: {output_path}")
        serializable_results = []
        for result in results:
            result_dict = {
                "file_path": result.file_path,
                "success": result.success,
                "error_message": result.error_message,
            }

            if result.input_ids is not None:
                result_dict["has_tokens"] = True
                result_dict["token_count"] = len(result.input_ids)
            else:
                result_dict["has_tokens"] = False

            serializable_results.append(result_dict)

        output_json = os.path.join(output_path, "results.json")
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(serializable_results, handle, indent=2)

        console.print("[green]Results saved successfully")


def generate_token_ids(tokenizer: PreTrainedTokenizerFast, asm_path: str):
    """Backward-compatible helper for direct tokenization of an ASM file."""
    processor = FileProcessor(tokenizer=tokenizer, cleanup_temp_files=True)
    result = processor.process_single_file(asm_path)
    if not result.success:
        return None
    return result.input_ids, result.attention_mask
