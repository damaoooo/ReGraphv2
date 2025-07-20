"""
Utility functions for dataset building
"""
import os
import glob
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from Utils.utils import should_skip_this_file

console = Console()


def find_llvm_files(directory: str, pattern: str = "**/*.ll") -> List[str]:
    """Find all LLVM IR files in directory recursively"""
    return glob.glob(os.path.join(directory, pattern), recursive=True)


def cleanup_residual_temp_files(input_folder: str):
    """Clean up residual temporary files from previous interrupted runs"""
    console.print("[yellow]Cleaning up residual temporary files from previous runs...")
    
    cleaned_count = 0
    
    try:
        # Find and delete all temporary files
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if (file.endswith('_purified.ll') or 
                    file.endswith('_instrumented.ll') or 
                    file.endswith('.dot')):
                    
                    temp_file = os.path.join(root, file)
                    try:
                        os.remove(temp_file)
                        cleaned_count += 1
                    except Exception as e:
                        continue
                        
    except Exception as e:
         console.print(f"[red]Error cleaning directory {input_folder}: {e}")

    if cleaned_count > 0:
        console.print(f"[green]Cleaned up {cleaned_count} residual temporary files")
    else:
        console.print("[green]No residual temporary files found")


class FileFilter:
    """Handles file filtering operations"""
    
    def __init__(self, num_processes: int):
        self.num_processes = num_processes
        
    def _should_skip_file(self, file_path: str) -> bool:
        """检查单个文件是否应该跳过"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return should_skip_this_file(content)
        except Exception as e:
            return True  # 读取失败时跳过该文件
    
    def _filter_files_chunk(self, file_paths: List[str]) -> List[str]:
        """处理一批文件的预筛选"""
        valid_files = []
        for file_path in file_paths:
            if not self._should_skip_file(file_path):
                valid_files.append(file_path)
        return valid_files
    
    def filter_files_parallel(self, file_paths: List[str]) -> List[str]:
        """使用多核心并行预筛选文件"""
        console.print(f"[yellow]Starting parallel file filtering with {self.num_processes} workers")
        
        chunk_size = 200
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"[green]Filtering {len(file_paths)} files", 
                total=len(file_paths)
            )
            
            valid_files = []
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # 提交分块任务
                future_to_chunk = {}
                for i in range(0, len(file_paths), chunk_size):
                    chunk = file_paths[i:i + chunk_size]
                    future = executor.submit(self._filter_files_chunk, chunk)
                    future_to_chunk[future] = chunk
                
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_valid_files = future.result()
                        valid_files.extend(chunk_valid_files)
                        chunk = future_to_chunk[future]
                        progress.update(task, advance=len(chunk))
                    except Exception as e:
                        chunk = future_to_chunk[future]
                        progress.update(task, advance=len(chunk))
        
        console.print(f"[green]Filtering completed: {len(valid_files)}/{len(file_paths)} files passed the filter")
        return valid_files
