#!/usr/bin/env python3
"""
LLVM IR Corpus Builder for BPE Tokenization
Reads LLVM IR files, normalizes them, and saves as HuggingFace dataset for BPE training
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import argparse
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

from datasets import Dataset, DatasetDict
from datasets import load_dataset
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

from normalizer import LLVMIRNormalizer


def extract_optimization(filename: str) -> str:
    """Extract optimization level from filename"""
    for opt in ['O0', 'O1', 'O2', 'O3', 'Os', 'Oz']:
        if opt in filename:
            return opt
    return 'unknown'


class LLVMIRCorpusBuilder:
    def __init__(self, base_dir: str, output_dir: str = "llvm_corpus", num_processes: Optional[int] = None):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.normalizer = LLVMIRNormalizer()
        self.num_processes = num_processes or mp.cpu_count()
        self.console = Console()
        
        # Special tokens used by normalizer - ensure they are in BPE vocabulary
        self.special_tokens = [
            '<MED_INT>',
            '<LARGE_INT>', 
            '<HUGE_INT>',
            '<HEX_CONST>',
            '<FLOAT_CONST>',
            '<STRING_CONST>'
        ]
        
        # Statistics (需要线程安全)
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_functions': 0,
            'total_lines': 0,
            'failed_list': [],
            'special_tokens_added': len(self.special_tokens)
        }
        self.stats_lock = threading.Lock()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    def find_llvm_files(self) -> List[Tuple[dict, Path]]:
        """Find all LLVM IR files in the base directory"""
        ret: List[Tuple[dict, Path]] = []
        folders = os.listdir(self.base_dir)
        for folder in folders:
            folder_path = Path(self.base_dir, folder)
            if not folder_path.is_dir():
                continue

            binary_name = folder_path.name
            function_map_csv = folder_path / "function_map.csv"
            if not function_map_csv.is_file():
                continue

            # 使用csv库读取function_map.csv，生成{文件名: 函数名}映射
            file_func_map = {}
            with open(function_map_csv, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        func_name, file_name = row[0], row[1]
                        file_func_map[file_name] = func_name

            for file_name in os.listdir(folder_path):
                function_name = file_func_map.get(file_name, None)
                if not function_name or not file_name.endswith('.ll'):
                    continue

                file_path = folder_path / file_name
                if not file_path.is_file():
                    continue

                metadata = {
                    'file_name': binary_name,
                    'function_name': function_name,
                    'opt': extract_optimization(binary_name),
                }

                ret.append((metadata, file_path))
        return ret

    @staticmethod
    def process_single_file(args: Tuple[Path, dict]) -> Tuple[List[Dict], Dict]:
        """Static method to process a single file - used for multiprocessing"""
        file_path, metadata = args
        normalizer = LLVMIRNormalizer()
        
        # Statistics for this file
        file_stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_functions': 0,
            'total_lines': 0,
            'failed_list': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                ir_code = f.read()
            
            # Extract individual functions
            func_code = ir_code.strip()
            
            results = []

            if not func_code.strip():
                return results, file_stats

            # Reset normalizer for each function to get consistent naming
            normalizer.reset()

            # Normalize the function
            normalized_func = normalizer.normalize_ir(func_code)

            # Create metadata
            metadata['file_path'] = str(file_path)
            metadata['original_length'] = len(func_code)
            metadata['normalized_length'] = len(normalized_func)

            results.append({
                'text': normalized_func,
                'metadata': metadata
            })

            file_stats['total_functions'] += 1
            file_stats['total_lines'] += len(normalized_func.split('\n'))
            file_stats['processed_files'] += 1
            return results, file_stats
            
        except Exception as e:
            file_stats['failed_files'] += 1
            file_stats['failed_list'].append(str(file_path))
            print(f"Error processing {file_path}: {e}")
            return [], file_stats


    def process_file(self, file_path: Path, metadata: dict) -> List[Dict]:
        """Process a single LLVM IR file and return normalized function data (legacy method)"""
        results, file_stats = self.process_single_file((file_path, metadata))
        
        # Update main stats
        with self.stats_lock:
            for key in ['processed_files', 'failed_files', 'total_functions', 'total_lines']:
                self.stats[key] += file_stats[key]
            self.stats['failed_list'].extend(file_stats['failed_list'])
        
        return results

    def update_stats(self, file_stats: Dict):
        """Thread-safe method to update statistics"""
        with self.stats_lock:
            for key in ['processed_files', 'failed_files', 'total_functions', 'total_lines']:
                self.stats[key] += file_stats[key]
            self.stats['failed_list'].extend(file_stats['failed_list'])

    def build_corpus(self, max_files: Optional[int] = None) -> Union[Dataset, None]:
        """Build the corpus from all LLVM IR files using multiprocessing"""
        
        # Find all LLVM IR files
        ll_files = self.find_llvm_files()
        self.console.print(f"[bold green]Found {len(ll_files)} LLVM IR files[/bold green]")
        if max_files:
            ll_files = ll_files[:max_files]
        
        self.stats['total_files'] = len(ll_files)
        
        self.console.print(f"[bold green]Will process {len(ll_files)} LLVM IR files[/bold green]")
        self.console.print(f"[bold blue]Using {self.num_processes} processes[/bold blue]")
        
        # Prepare arguments for multiprocessing
        process_args = [(file_path, metadata) for metadata, file_path in ll_files]
        
        # Process all files with multiprocessing and rich progress bar
        all_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Processing files...", total=len(process_args))
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all tasks
                future_to_args = {
                    executor.submit(self.process_single_file, args): args 
                    for args in process_args
                }
                
                # Process completed tasks
                for future in as_completed(future_to_args):
                    try:
                        results, file_stats = future.result()
                        all_data.extend(results)
                        self.update_stats(file_stats)
                        
                    except Exception as e:
                        args = future_to_args[future]
                        file_path = args[0]
                        self.console.print(f"[red]Error processing {file_path}: {e}[/red]")
                        with self.stats_lock:
                            self.stats['failed_files'] += 1
                            self.stats['failed_list'].append(str(file_path))
                    
                    progress.advance(task)
        
        # Create dataset
        if all_data:
            dataset = Dataset.from_list(all_data)
            
            # Save statistics
            with open(self.output_dir / 'corpus_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            return dataset
        else:
            self.console.print("[red]No data to create dataset[/red]")
            return None
    
    def save_corpus(self, dataset: Dataset, name: str = "llvm_ir_corpus"):
        """Save the corpus as HuggingFace dataset"""
        
        # Save as parquet for efficiency
        dataset.save_to_disk(self.output_dir / name)
        
        # Also save as JSON for easy inspection
        with open(self.output_dir / f"{name}.jsonl", 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        # Save just the text for BPE training with special tokens included
        with open(self.output_dir / f"{name}_text_only.txt", 'w') as f:
            # First, add special tokens at the beginning to ensure they are in vocabulary
            for token in self.special_tokens:
                f.write(f"{token}\n")
            
            # Add a separator line
            f.write("\n")
            
            # Then add all the actual corpus data
            for item in dataset:
                f.write(item['text'] + '\n\n')
        
        # Save special tokens list for reference
        with open(self.output_dir / 'special_tokens.json', 'w') as f:
            json.dump({
                'special_tokens': self.special_tokens,
                'description': 'Special tokens used by LLVM IR normalizer that should be preserved in BPE vocabulary'
            }, f, indent=2)
        
        # Validate special tokens presence in corpus
        self._validate_special_tokens_in_corpus(dataset)
        
        print(f"Corpus saved to {self.output_dir}")
        print(f"Total functions: {len(dataset)}")
        print(f"Special tokens included: {self.special_tokens}")
        print(f"Statistics saved to {self.output_dir / 'corpus_stats.json'}")
        print(f"Special tokens saved to {self.output_dir / 'special_tokens.json'}")
    
    def _validate_special_tokens_in_corpus(self, dataset: Dataset):
        """Validate that special tokens appear in the corpus and report statistics"""
        token_counts = {token: 0 for token in self.special_tokens}
        
        for item in dataset:
            text = item['text']
            for token in self.special_tokens:
                token_counts[token] += text.count(token)
        
        # Save token statistics
        with open(self.output_dir / 'special_token_stats.json', 'w') as f:
            json.dump({
                'token_frequencies': token_counts,
                'total_samples': len(dataset),
                'validation_passed': all(count > 0 for count in token_counts.values())
            }, f, indent=2)
        
        # Print validation results
        self.console.print("\n[bold cyan]=== Special Token Validation ===[/bold cyan]")
        for token, count in token_counts.items():
            if count > 0:
                self.console.print(f"[green]✓ {token}:[/green] {count} occurrences")
            else:
                self.console.print(f"[red]✗ {token}:[/red] 0 occurrences (may need more diverse data)")
        
        missing_tokens = [token for token, count in token_counts.items() if count == 0]
        if missing_tokens:
            self.console.print(f"\n[yellow]Warning: {len(missing_tokens)} special tokens not found in corpus: {missing_tokens}[/yellow]")
            self.console.print("[yellow]These tokens are still added to the vocabulary file for BPE training[/yellow]")
        else:
            self.console.print("\n[green]✓ All special tokens found in corpus![/green]")
    
    def print_stats(self):
        """Print processing statistics using rich console"""
        self.console.print("\n[bold cyan]=== Corpus Building Statistics ===[/bold cyan]")
        self.console.print(f"[green]Total files found:[/green] {self.stats['total_files']}")
        self.console.print(f"[green]Successfully processed:[/green] {self.stats['processed_files']}")
        self.console.print(f"[red]Failed files:[/red] {self.stats['failed_files']}")
        self.console.print(f"[blue]Total functions extracted:[/blue] {self.stats['total_functions']}")
        self.console.print(f"[blue]Total lines of code:[/blue] {self.stats['total_lines']}")
        self.console.print(f"[yellow]Special tokens included:[/yellow] {self.stats['special_tokens_added']}")
        self.console.print(f"[yellow]Special tokens:[/yellow] {', '.join(self.special_tokens)}")
        
        if self.stats['failed_list']:
            self.console.print("\n[red]Failed files:[/red]")
            for failed_file in self.stats['failed_list'][:10]:  # Show first 10
                self.console.print(f"  [dim]- {failed_file}[/dim]")
            if len(self.stats['failed_list']) > 10:
                self.console.print(f"  [dim]... and {len(self.stats['failed_list']) - 10} more[/dim]")


def main():
    parser = argparse.ArgumentParser(description='Build LLVM IR corpus for BPE tokenization')
    parser.add_argument('--input_dir', '-i', required=True, 
                       help='Input directory containing LLVM IR files')
    parser.add_argument('--output_dir', '-o', default='llvm_corpus',
                       help='Output directory for corpus')
    parser.add_argument('--max_files', '-m', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--name', '-n', default='llvm_ir_corpus',
                       help='Name of the corpus dataset')
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='Number of processes to use (default: CPU count)')
    
    args = parser.parse_args()
    
    console = Console()
    
    # Create corpus builder
    builder = LLVMIRCorpusBuilder(
        args.input_dir, 
        args.output_dir, 
        num_processes=args.processes
    )
    
    # Build corpus
    console.print(f"[bold yellow]Building corpus with {builder.num_processes} processes...[/bold yellow]")
    dataset = builder.build_corpus(max_files=args.max_files)
    
    if dataset:
        # Save corpus
        builder.save_corpus(dataset, args.name)
        
        # Print statistics
        builder.print_stats()
        
        # Show a sample
        console.print("\n[bold cyan]=== Sample normalized function ===[/bold cyan]")
        console.print(f"[dim]{dataset[0]['text'][:500]}...[/dim]")
        console.print(f"\n[yellow]Metadata:[/yellow] {dataset[0]['metadata']}")
    else:
        console.print("[red]Failed to build corpus[/red]")


if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    main()
