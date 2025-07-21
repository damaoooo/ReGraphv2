"""
Main dataset builder class for LLVM IR files
"""
import os
import sys
import time
import pickle
import json
import logging
import multiprocessing as mp
from typing import List, Union
from transformers import PreTrainedTokenizerFast
import datasets
import sqlite3

from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from .processing_result import ProcessingResult
from .file_processor import FileProcessor
from .parallel_processor import ParallelProcessor, create_hf_dataset_from_files
from .dataset_utils import FileFilter, cleanup_residual_temp_files

from Utils.utils import DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH, DEFAULT_CFG_SO_PATH

console = Console()


class DatasetBuilder:
    """Main class for building datasets from LLVM IR files with graph structures"""
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerFast,
                 tokenizer_path: str,
                 db_file: str,
                 num_processes: int = None,
                 cleanup_temp_files: bool = True,
                 cache: bool = True):
        self.tokenizer = tokenizer
        self.db_file = db_file
        self.tokenizer_path = tokenizer_path
        # Increase max workers to handle I/O bound operations
        self.num_processes = num_processes
        self.cleanup_temp_files = cleanup_temp_files
        self.cache = cache
        self._setup_logging()
        
        # Initialize components
        self.file_processor = FileProcessor(
            tokenizer=tokenizer,
            cleanup_temp_files=cleanup_temp_files
        )
        self.parallel_processor = ParallelProcessor(num_processes=self.num_processes)
        self.file_filter = FileFilter(num_processes=self.num_processes)
        
    def __getstate__(self):
        """Exclude logger and any rich objects from pickling."""
        state = self.__dict__.copy()
        # Remove objects that can't be pickled
        if 'logger' in state:
            del state['logger']
        return state

    def __setstate__(self, state):
        """Recreate logger in the new process without rich."""
        self.__dict__.update(state)
        self._setup_logging(rich=False)

    def _setup_logging(self, rich: bool = True):
        """Setup logging configuration."""
        # Remove existing handlers to avoid duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if rich:
            # Rich handler for main process
            handlers = [
                RichHandler(console=console, rich_tracebacks=True),
                logging.FileHandler('dataset_builder.log')
            ]
        else:
            # Standard handler for child processes
            handlers = [
                logging.FileHandler('dataset_builder_debug.log')
            ]
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(processName)s: %(message)s",
            datefmt="[%X]",
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)

    def process_dataset(self, output_path: str, batch_size: int = 1000, use_parallel: bool = False, skip_filtering: bool = False, use_hf: bool = False):
        """Main method to process entire dataset"""
        console.print(f"[yellow]Using {self.num_processes} processes, batch size: {batch_size}")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            console.print(f"[green]Created output directory: {output_path}")
        else:
            console.print(f"[yellow]Output directory already exists: {output_path}")
        

        console.print(f"[yellow]loading source file names...")
        
        file_cache = os.path.join(output_path, 'file_list_cache.pkl')
        if self.cache and os.path.exists(file_cache):
            console.print(f"[yellow]Loading file cache from {file_cache}")
            with open(file_cache, 'rb') as f:
                filtered_files = pickle.load(f)
        else:
            filter_start_time = time.time()
            
            with sqlite3.connect(self.db_file, uri=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT input_path FROM results")
                filtered_files = [row[0] for row in cursor.fetchall()]

            filter_end_time = time.time()
            console.print(f"[green]Pre-filtering completed in {filter_end_time - filter_start_time:.2f} seconds")
            console.print(f"[green]Files after filtering: {len(filtered_files)}")
            if self.cache:
                console.print(f"[yellow]Saving file cache to {file_cache}")
                with open(file_cache, 'wb') as f:
                    pickle.dump(filtered_files, f)

        # Load progress.txt as the resume point
        console.print("[yellow]Checking for resume file...")
        processed_files_set = set()
        resume_file = os.path.join(output_path, 'progress.txt')
        if os.path.exists(resume_file) and not skip_filtering:
            console.print("[yellow]Loading processed files from resume file...")
            
            with open(resume_file, 'r') as f:
                for line in f:
                    processed_files_set.add(line.strip())
        
            # Filter out already processed files using set lookup (O(1) per lookup)
            original_count = len(filtered_files)
            filtered_files = [f for f in filtered_files if f not in processed_files_set]
            processed_count = original_count - len(filtered_files)
            console.print(f"[yellow]Resuming from {processed_count} previously processed files, {len(filtered_files)} files remaining")
        else:
            processed_count = 0
            console.print("[yellow]No resume file found, starting fresh")

        if use_hf:
            console.print("[bold blue]Using Hugging Face datasets from generator.")
            start_time = time.time()
            
            # Create dataset using the standalone function
            dataset = create_hf_dataset_from_files(
                file_paths=filtered_files,
                db_path=self.db_file,
                tokenizer_path=self.tokenizer_path,
                num_processes=self.num_processes,
                cleanup_temp_files=self.cleanup_temp_files
            )
            
            end_time = time.time()
            
            console.print(f"[green]Dataset creation from generator completed in {end_time - start_time:.2f} seconds.")
            
            # Save the dataset
            self.save_results(dataset, output_path, use_hf=True)
            
            # Display summary
            table = Table(title="Processing Summary (Hugging Face Dataset)")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Successfully processed files", str(len(dataset)))
            table.add_row("Processing time", f"{end_time - start_time:.2f} seconds")
            if len(dataset) > 0:
                table.add_row("Average time per file", f"{(end_time - start_time) / len(dataset):.4f} seconds")
            
            console.print(table)
            return dataset

        # Use sequential processing for debugging when num_processes=1
        if self.num_processes == 1:
            console.print(f"[yellow]Processing method: Sequential (debug mode)")
            start_time = time.time()
            results = self.parallel_processor.process_files_sequential(self.file_processor, filtered_files, db_file=self.db_file)
            end_time = time.time()
        else:
            console.print(f"[yellow]Processing method: {'Parallel chunks' if use_parallel else 'Batched'}")
            start_time = time.time()
            if use_parallel:
                results = self.parallel_processor.process_files_parallel(file_processor=self.file_processor, file_paths=filtered_files, 
                                                                         db_file=self.db_file, output_path=output_path, start_index=len(processed_files_set))
            else:
                results = self.parallel_processor.process_files_batch(self.file_processor, filtered_files, db_file=self.db_file, batch_size=batch_size)
            end_time = time.time()
        
        if use_parallel:
            successful_count = results[0]
        else:
            successful_count = sum(1 for r in results if r.success)
        
        # Create summary table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
    
        table.add_row("Files processed", str(len(filtered_files)))
        if len(filtered_files) > 0:
            table.add_row("Successful", str(successful_count))
            table.add_row("Failed", str(len(filtered_files) - successful_count))
            table.add_row("Processing time", f"{end_time - start_time:.2f} seconds")
            table.add_row("Average time per file", f"{(end_time - start_time) / len(filtered_files):.4f} seconds")
        
        console.print(table)
        if use_hf or (self.num_processes == 1) or self.num_processes == 1:
            self.save_results(results, output_path)
        return results

    def save_results(self, results: Union[List[ProcessingResult], 'datasets.Dataset'], output_path: str, use_hf: bool = False):
        """Save results to JSON file or Hugging Face dataset directory"""
        if use_hf:
            console.print(f"[yellow]Saving Hugging Face dataset to: {output_path}")
            try:
                results.save_to_disk(output_path)
                console.print(f"[green]Dataset saved successfully to {output_path}")
            except Exception as e:
                console.print(f"[red]Error saving Hugging Face dataset: {e}")
            return

        console.print(f"[yellow]Saving results to: {output_path}")
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'file_path': result.file_path,
                'success': result.success,
                'error_message': result.error_message
            }
            
            # Handle non-serializable graph objects
            if result.ddg_graph is not None:
                result_dict['has_ddg'] = True
                result_dict['ddg_type'] = str(type(result.ddg_graph))
            else:
                result_dict['has_ddg'] = False
                
            if result.cfg_graph is not None:
                result_dict['has_cfg'] = True
                result_dict['cfg_type'] = str(type(result.cfg_graph))
            else:
                result_dict['has_cfg'] = False
                
            if result.input_ids is not None:
                result_dict['has_tokens'] = True
                result_dict['token_count'] = len(result.input_ids)
            else:
                result_dict['has_tokens'] = False
                
            serializable_results.append(result_dict)
        output_json = os.path.join(output_path, 'results.json')
        with open(output_json, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        console.print(f"[green]Results saved successfully")


# Legacy functions for backward compatibility
def generate_ddg(tokenizer: PreTrainedTokenizerFast, llvm_ir_path: str):
    processor = FileProcessor(
        tokenizer=tokenizer,
        ddg_so_path=DEFAULT_DDG_SO_PATH,
        purify_so_path=DEFAULT_PURIFY_SO_PATH,
        cfg_so_path=DEFAULT_CFG_SO_PATH
    )
    purified_file = processor._opt_initial_purify(llvm_ir_path)
    if purified_file is None:
        return None
    try:
        return processor._generate_ddg(llvm_ir_path, purified_file)
    finally:
        processor._cleanup_temp_files([purified_file])

def generate_cfg(tokenizer: PreTrainedTokenizerFast, llvm_ir_path: str):
    processor = FileProcessor(
        tokenizer=tokenizer,
        ddg_so_path=DEFAULT_DDG_SO_PATH,
        purify_so_path=DEFAULT_PURIFY_SO_PATH,
        cfg_so_path=DEFAULT_CFG_SO_PATH
    )
    purified_file = processor._opt_initial_purify(llvm_ir_path)
    if purified_file is None:
        return None
    try:
        return processor._generate_cfg(llvm_ir_path, purified_file)
    finally:
        processor._cleanup_temp_files([purified_file])

def generate_token_ids(tokenizer: PreTrainedTokenizerFast, llvm_ir_path: str):
    processor = FileProcessor(
        tokenizer=tokenizer,
        ddg_so_path=DEFAULT_DDG_SO_PATH,
        purify_so_path=DEFAULT_PURIFY_SO_PATH,
        cfg_so_path=DEFAULT_CFG_SO_PATH
    )
    purified_file = processor._opt_initial_purify(llvm_ir_path)
    if purified_file is None:
        return None
    try:
        return processor._generate_token_ids(llvm_ir_path, purified_file)
    finally:
        processor._cleanup_temp_files([purified_file])
