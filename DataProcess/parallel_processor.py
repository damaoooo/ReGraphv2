"""
Parallel processing utilities for dataset building
"""
import logging
import sys
from transformers import PreTrainedTokenizerFast
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from pebble import ProcessPool
from tqdm import tqdm
import traceback
import sqlite3
import multiprocessing

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from .processing_result import ProcessingResult
from .file_processor import FileProcessor
from .dataset_features import get_dataset_features
from .pd_writer import PDWriter
import datasets

console = Console()


def process_single_file_standalone(
    input_path: str,
    purified_file: str,
    cfg_dot: str,
    ddg_dot: str,
    instrumented_file: str,
    tokenizer: PreTrainedTokenizerFast,
    cleanup_temp_files: bool = True
) -> ProcessingResult:
    """Standalone version of process_single_file that doesn't require self"""
    try:
        processor = FileProcessor(
            tokenizer=tokenizer,
            cleanup_temp_files=cleanup_temp_files
        )
        return processor.process_single_file(input_file=input_path, purified_file=purified_file, 
                                             cfg_dot=cfg_dot, ddg_dot=ddg_dot, instrumented_file=instrumented_file)
        
    except Exception as e:
        return ProcessingResult(
            file_path=input_path,
            success=False,
            error_message=str(e)
        )


def process_chunk_standalone(
    file_paths: List[str],
    db_file: str,
    tokenizer_path: str,
    cleanup_temp_files: bool = True
) -> List[ProcessingResult]:
    """Standalone version of _process_chunk"""
    from Tokenizer.ir_tokenizer import load_tokenizer
    
    tokenizer = load_tokenizer(tokenizer_path)
    with sqlite3.connect(db_file, uri=True) as conn:
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(file_paths))
        sql_query = f"SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot FROM results WHERE input_path IN ({placeholders})"
        cursor.execute(sql_query, file_paths)
        file_paths = cursor.fetchall()
        conn.commit()

    results = []
    for file_path in file_paths:
        input_file, purified_file, instrumented_file, cfg_dot, ddg_dot,  = file_path
        if not input_file or not purified_file or not cfg_dot or not ddg_dot or not instrumented_file:
            print(f"Invalid file paths in chunk: {file_path}")
            continue
        result = process_single_file_standalone(
            input_path=input_file,
            purified_file=purified_file,
            cfg_dot=cfg_dot,
            ddg_dot=ddg_dot,
            instrumented_file=instrumented_file,
            tokenizer=tokenizer,
            cleanup_temp_files=cleanup_temp_files
        )
        results.append(result)
    return results


class ParallelProcessor:
    """Handles parallel processing of files"""
    
    def __init__(self, num_processes: int):
        self.num_processes = num_processes
        self.logger = logging.getLogger(__name__)
        
    def process_files_sequential(self, file_processor: FileProcessor, file_paths: List[str], db_file: str) -> List[ProcessingResult]:
        """Process files sequentially for debugging (when num_processes=1)"""
        console.print(f"[yellow]Using sequential processing for debugging")
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            conn = sqlite3.connect(db_file, uri=True)
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(file_paths))
            sql_query = f"SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot FROM results WHERE input_path IN ({placeholders})"
            cursor.execute(sql_query, file_paths)
            file_paths = cursor.fetchall()
            conn.commit()
            cursor.close()
            conn.close()
            
            task = progress.add_task(
                f"[green]Processing {len(file_paths)} files sequentially", 
                total=len(file_paths)
            )
            
            for i, file_path in enumerate(file_paths):
                try:
                    result = file_processor.process_single_file(
                        input_file=file_path[0],
                        purified_file=file_path[1],
                        instrumented_file=file_path[2],
                        cfg_dot=file_path[3],
                        ddg_dot=file_path[4]
                    )
                    results.append(result)
                    
                    # Log progress for debugging
                    if result.success:
                        self.logger.info(f"✓ Successfully processed ({i+1}/{len(file_paths)}): {file_path}")
                    else:
                        self.logger.error(f"✗ Failed to process ({i+1}/{len(file_paths)}): {file_path} - {result.error_message}")
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    self.logger.error(f"Exception processing {file_path}: {e}")
                    results.append(ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error_message=str(e)
                    ))
                    progress.update(task, advance=1)
                    
        return results
        
    def process_files_batch(self, file_processor: FileProcessor, file_paths: List[str], db_file, batch_size: int = 1000) -> List[ProcessingResult]:
        """Process files in batches using multiprocessing with rich progress bars"""
        all_results = []
        total_batches = (len(file_paths) + batch_size - 1) // batch_size
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            overall_task = progress.add_task(
                f"[green]Processing {len(file_paths)} files in {total_batches} batches", 
                total=len(file_paths)
            )
            
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                batch_task = progress.add_task(
                    f"[blue]Batch {batch_num}/{total_batches}", 
                    total=len(batch)
                )
                
                # Use ThreadPoolExecutor for I/O bound operations like subprocess calls
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                    # Submit all tasks in the batch with immediate execution
                    futures = []
                    with sqlite3.connect(db_file, uri=True) as conn:
                        cursor = conn.cursor()
                        placeholders = ','.join(['?'] * len(batch))
                        sql_query = f"SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot FROM results WHERE input_path IN ({placeholders})"
                        cursor.execute(sql_query, batch)
                        batch_files = cursor.fetchall()
                        conn.commit()

                    for file_path in batch_files:
                        input_file, purified_file, instrumented_file, cfg_dot, ddg_dot = file_path
                        future = executor.submit(file_processor.process_single_file, input_file, purified_file, instrumented_file, cfg_dot, ddg_dot)
                        futures.append((future, file_path))
                    
                    batch_results = []
                    for future, file_path in futures:
                        try:
                            result = future.result()
                            batch_results.append(result)
                            progress.update(batch_task, advance=1)
                            progress.update(overall_task, advance=1)
                        except Exception as e:
                            self.logger.error(f"Error processing {file_path}: {e}")
                            batch_results.append(ProcessingResult(
                                file_path=file_path,
                                success=False,
                                error_message=str(e)
                            ))
                            progress.update(batch_task, advance=1)
                            progress.update(overall_task, advance=1)
                        
                all_results.extend(batch_results)
                
                # Log progress
                successful = sum(1 for r in batch_results if r.success)
                progress.update(batch_task, description=f"[green]Batch {batch_num} completed: {successful}/{len(batch)} successful")
                
        return all_results

    def process_files_parallel(self, file_processor: FileProcessor, file_paths: List[str], db_file: str, output_path: str, start_index: int) -> List[ProcessingResult]:
        """Alternative processing method using map for better CPU utilization"""
        console.print(f"[yellow]Using parallel processing with {self.num_processes} workers")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            total, success_t = len(file_paths), 0
            
            task = progress.add_task(
                f"[green]Processing {len(file_paths)} files", 
                total=len(file_paths)
            )

            chunk_size = 500  # Smaller chunks for better load balancing

            result_queue = multiprocessing.Manager().Queue()
            pd_writer = PDWriter(
                queue=result_queue,
                output_path=output_path,
                bin_size=chunk_size*20,
                start_index=start_index
            )
            
            pd_process = multiprocessing.Process(
                target=pd_writer.start
            )
            pd_process.start()
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit work in chunks for better load balancing
                future_to_files = {}
                for i in range(0, len(file_paths), chunk_size):
                    chunk = file_paths[i:i + chunk_size]
                    future = executor.submit(self._process_chunk, file_processor, chunk, db_file, result_queue)
                    future_to_files[future] = chunk
                
                for future in as_completed(future_to_files):
                    try:
                        chunk_results = future.result()
                        success, total = chunk_results[0], chunk_results[1]
                        success_t += success
                        progress.update(task, advance=total)
                    except Exception as e:
                        chunk = future_to_files[future]
                        self.logger.error(f"Error processing chunk: {e}")
                        # Add failed results for the chunk
                        progress.update(task, advance=len(chunk))
                    finally:
                        future_to_files.pop(future)
                        
            result_queue.put('STOP')  # Signal end of processing
            pd_process.join()  # Wait for the PDWriter process to finish
            
        console.print(f"[green]Parallel processing completed. Results written to {output_path}")
        return success_t, total

    def _process_chunk(self, file_processor: FileProcessor, file_paths: List[str], db_file: str, queue: multiprocessing.Queue) -> List[int]:
        """Process a chunk of files in a single process"""
        results = []
        with sqlite3.connect(db_file, uri=True) as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(file_paths))
            sql_query = f"SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot FROM results WHERE input_path IN ({placeholders})"
            cursor.execute(sql_query, file_paths)
            file_paths = cursor.fetchall()
            conn.commit()

        for file_path in file_paths:
            input_file, purified_file, instrumented_file, cfg_dot, ddg_dot = file_path
            result = file_processor.process_single_file(input_file=input_file,
                                                        purified_file=purified_file,
                                                        instrumented_file=instrumented_file,
                                                        cfg_dot=cfg_dot,
                                                        ddg_dot=ddg_dot)
            results.append(result)
        total_len = len(results)
        success = sum(1 for r in results if r.success)
        
        results = [result.to_dict() for result in results if result.success]
        queue.put(results)
        return [success, total_len] 


def create_hf_dataset_from_files(
    file_paths: List[str],
    db_path: str,
    tokenizer_path: str,
    num_processes: int,
    cleanup_temp_files: bool = True
) -> 'datasets.Dataset':
    """Create HuggingFace dataset from files using standalone functions"""
    
    def standalone_generator():
        """Standalone generator that doesn't capture any class instances"""
        print(f"Starting parallel processing generator for HF with {num_processes} workers", file=sys.stderr)
        
        chunk_size = 400
        
        # Use tqdm progress bar which is more serialization-friendly
        with tqdm(total=len(file_paths), desc="Processing files for HF dataset", file=sys.stderr, dynamic_ncols=True) as pbar:
            with ProcessPool(max_workers=num_processes) as executor:
                future_to_files = {}
                for i in range(0, len(file_paths), chunk_size):
                    chunk = file_paths[i:i + chunk_size]
                    future = executor.schedule(
                        process_chunk_standalone,
                        args=(
                            chunk, 
                            db_path,
                            tokenizer_path, 
                            cleanup_temp_files
                        )
                    )
                    future_to_files[future] = chunk
                
                processed_count = 0
                total_files = len(file_paths)
                for future in as_completed(future_to_files):
                    try:
                        chunk_results = future.result()
                        for result in chunk_results:
                            if result.success:
                                yield {
                                    'file_path': result.file_path,
                                    'ddg_graph': result.ddg_graph,
                                    'cfg_graph': result.cfg_graph,
                                    'input_ids': result.input_ids
                                }
                        processed_count += len(chunk_results)
                        pbar.update(len(chunk_results))
                        pbar.set_postfix({'processed': processed_count, 'total': total_files})
                    except Exception as e:
                        # --- 增强的错误日志 ---
                        chunk = future_to_files[future]
                        print(f"!! 子进程任务处理失败 !!", file=sys.stderr)
                        print(f"处理文件块 {chunk} 时发生严重错误: {e}", file=sys.stderr)
                        # 打印完整的错误堆栈跟踪，这至关重要！
                        traceback.print_exc(file=sys.stderr)
                        pbar.update(len(chunk))
    
    # Create dataset from the standalone generator with features
    dataset = datasets.Dataset.from_generator(
        standalone_generator,
        features=get_dataset_features()
    )
    return dataset
