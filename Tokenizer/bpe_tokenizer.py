#!/usr/bin/env python3
"""
BPE Tokenizer Training Script for LLVM IR
Trains a BPE tokenizer on the normalized LLVM IR corpus for machine learning
"""

import os
import time
import sys
from pathlib import Path
from typing import Optional, List
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import typer
from rich.console import Console
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import Dataset, load_from_disk

# Initialize rich console
console = Console()
app = typer.Typer(help="Train BPE tokenizer on LLVM IR corpus")



def _process_single_function(func: str) -> str:
    """Process a single function - used for multiprocessing"""
    # Add function boundary tokens

    func_with_tokens = f"<func> {func.strip()} </func>"
    
    # Add basic block boundary tokens (identify basic block labels)
    lines = func_with_tokens.split('\n')
    processed_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Basic block labels (end with ':')
        if stripped.endswith(':') and not any(op in stripped for op in ['=', 'call', 'load', 'store']):
            processed_lines.append(f"{line} <bb>")
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def _write_batch_to_file(batch_data: tuple) -> str:
    """Write a batch of functions to a temporary file - used for multiprocessing"""
    batch, batch_idx, output_dir = batch_data
    temp_file = Path(output_dir) / f"temp_batch_{batch_idx}.txt"
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        for func in batch:
            f.write(func + '\n')
    
    return str(temp_file)


class LLVMIRTokenizerTrainer:
    def __init__(self, corpus_path: str, corpus_name: str = "ir_corpus", output_dir: str = "llvm_tokenizer", num_workers: int = None):
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.corpus_name = corpus_name
        
        # Multiprocessing configuration
        self.num_workers = num_workers or min(mp.cpu_count(), 8)  # Limit to 8 to avoid too many processes
        console.print(f"[green]Using {self.num_workers} workers for parallel processing[/green]")
        
        # BPE Configuration
        self.vocab_size = 50000  # Large vocabulary for complex LLVM IR
        self.min_frequency = 2   # Minimum frequency for tokens
        
        # Special tokens for LLVM IR
        self.special_tokens = [
            "<pad>",     # Padding token
            "<unk>",     # Unknown token
            "<bos>",     # Beginning of sequence
            "<eos>",     # End of sequence
            "<mask>",    # Mask token for masked language modeling
            "<func>",    # Function boundary
            "<bb>",      # Basic block boundary
            "<var>",     # Variable token
            "<const>",   # Constant token
        ]
    
    def load_corpus_text(self) -> List[str]:
        """Load the corpus text for training"""
        
        with console.status("[bold green]Loading corpus...") as status:
            # Try to load from text_only file first (fastest)
            text_file = self.corpus_path / "{}_text_only.txt".format(self.corpus_name)
            if text_file.exists():
                status.update(f"[bold green]Loading corpus from text file: {text_file}")
                with open(text_file, 'r', encoding='utf-8') as f:
                    # Split by double newlines (function boundaries)
                    content = f.read()
                    functions = [func.strip() for func in content.split('\n\n') if func.strip()]
                    console.print(f"[green]✓ Loaded {len(functions)} functions from text file[/green]")
                    return functions
            
            # Try to load from HuggingFace dataset
            dataset_dir = self.corpus_path / self.corpus_name
            if dataset_dir.exists():
                status.update(f"[bold green]Loading corpus from HuggingFace dataset: {dataset_dir}")
                dataset = load_from_disk(str(dataset_dir))
                functions = [item['text'] for item in dataset]
                console.print(f"[green]✓ Loaded {len(functions)} functions from dataset[/green]")
                return functions
            
            # Try to load from JSONL file
            jsonl_file = self.corpus_path / "{}.jsonl".format(self.corpus_name)
            if jsonl_file.exists():
                status.update(f"[bold green]Loading corpus from JSONL file: {jsonl_file}")
                functions = []
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        functions.append(data['text'])
                console.print(f"[green]✓ Loaded {len(functions)} functions from JSONL file[/green]")
                return functions
        
        raise FileNotFoundError(f"No valid corpus found in {self.corpus_path}")
    
    def prepare_training_data(self, functions: List[str]) -> List[str]:
        """Prepare training data by adding special tokens and formatting"""
        
        console.print("[blue]Preparing training data...[/blue]")
        
        # Simple single-process preparation (string operations are too lightweight for multiprocessing)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing functions", total=len(functions))
            prepared_data = []
            for func in functions:
                prepared_data.append(_process_single_function(func))
                progress.advance(task)
            
            console.print(f"[green]✓ Processed {len(prepared_data)} functions[/green]")
        
        return prepared_data
    
    def create_tokenizer(self) -> Tokenizer:
        """Create and configure the BPE tokenizer"""

        # Initialize BPE model
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # Set pre-tokenizer (split on whitespace but preserve structure)
        tokenizer.pre_tokenizer = Whitespace()
        
        # Add post-processor for special tokens
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", 2),  # ID for <bos>
                ("<eos>", 3),  # ID for <eos>
            ],
        )
        
        return tokenizer
    
    def train_tokenizer(self, training_data: List[str], tokenizer: Tokenizer) -> Tokenizer:
        """Train the BPE tokenizer on the corpus using multiprocessing for file I/O"""
        
        console.print(f"[blue]Training BPE tokenizer with vocab size {self.vocab_size}[/blue]")
        
        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,  # Keep tokenizer's built-in progress bar
        )
        
        # Prepare batches for parallel file writing
        batch_size = max(1000, len(training_data) // (self.num_workers * 4))  # Adaptive batch size
        batches = []
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            batch_idx = i // batch_size
            batches.append((batch, batch_idx, str(self.output_dir)))
        
        console.print(f"[blue]Writing {len(batches)} batches using {self.num_workers} workers...[/blue]")
        
        # Use multiprocessing to write temporary files
        temp_files = []
        if len(batches) > 1 and self.num_workers > 1:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Writing batches", total=len(batches))
                
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_batch = {executor.submit(_write_batch_to_file, batch_data): batch_data for batch_data in batches}
                    
                    for future in as_completed(future_to_batch):
                        try:
                            temp_file = future.result()
                            temp_files.append(temp_file)
                            progress.advance(task)
                        except Exception as exc:
                            console.print(f"[red]Batch writing generated an exception: {exc}[/red]")
        else:
            # For small datasets, use single process
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Writing batches", total=len(batches))
                for batch_data in batches:
                    temp_file = _write_batch_to_file(batch_data)
                    temp_files.append(temp_file)
                    progress.advance(task)
        
        console.print(f"[green]✓ Generated {len(temp_files)} temporary files[/green]")
        
        # Give a moment for rich progress to clear before tokenizer training starts

        time.sleep(0.1)
        
        try:
            # Train the tokenizer - let it show its own progress
            console.print("[bold blue]Training tokenizer...[/bold blue]")
            tokenizer.train(temp_files, trainer)
            console.print("[green]✓ Tokenizer training completed[/green]")
            
        finally:
            # Clean up temporary files
            with console.status("[bold yellow]Cleaning up temporary files..."):
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            console.print("[green]✓ Temporary files cleaned up[/green]")
        
        return tokenizer
    
    def save_tokenizer(self, tokenizer: Tokenizer, name: str = "llvm_ir_bpe"):
        """Save the trained tokenizer"""
        
        with console.status("[bold blue]Saving tokenizer..."):
            # Save tokenizer
            tokenizer_path = self.output_dir / f"{name}.json"
            tokenizer.save(str(tokenizer_path))
            console.print(f"[green]✓ Tokenizer saved to: {tokenizer_path}[/green]")
            
            # Save vocabulary
            vocab = tokenizer.get_vocab()
            vocab_path = self.output_dir / f"{name}_vocab.json"
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✓ Vocabulary saved to: {vocab_path}[/green]")
            
            # Save configuration
            config = {
                "model_type": "BPE",
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "special_tokens": self.special_tokens,
                "tokenizer_path": str(tokenizer_path),
                "vocab_path": str(vocab_path),
            }
            
            config_path = self.output_dir / f"{name}_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            console.print(f"[green]✓ Configuration saved to: {config_path}[/green]")
        
        return tokenizer_path
    
    def test_tokenizer(self, tokenizer: Tokenizer, test_samples: List[str] = None):
        """Test the trained tokenizer with some samples"""
        
        if test_samples is None:
            # Default test samples
            test_samples = [
                'define i32 @func0(i32 %var0, i32 %var1) {',
                'bb0:',
                '  %var2 = add i32 %var0, %var1',
                '  ret i32 %var2',
                '}',
                '%var3 = load i32, i32* %var4, align 4',
                'store i32 <MED_INT>, i32* %var5, align 8',
                'br i1 %var6, label %bb1, label %bb2',
            ]
        
        console.print("\n[bold blue]=== Tokenizer Test ===[/bold blue]")
        
        # Create a table for better formatting
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Original", style="cyan", no_wrap=False)
        table.add_column("Tokens", style="yellow", no_wrap=False)
        table.add_column("IDs", style="green", no_wrap=False)
        table.add_column("Decoded", style="white", no_wrap=False)
        
        for sample in test_samples:
            encoded = tokenizer.encode(sample)
            decoded = tokenizer.decode(encoded.ids)
            
            # Truncate long outputs for table display
            tokens_str = str(encoded.tokens)
            if len(tokens_str) > 50:
                tokens_str = tokens_str[:47] + "..."
            
            ids_str = str(encoded.ids)
            if len(ids_str) > 30:
                ids_str = ids_str[:27] + "..."
            
            table.add_row(
                sample[:50] + "..." if len(sample) > 50 else sample,
                tokens_str,
                ids_str,
                decoded[:50] + "..." if len(decoded) > 50 else decoded
            )
        
        console.print(table)
    
    def get_tokenizer_stats(self, tokenizer: Tokenizer):
        """Get statistics about the trained tokenizer"""
        
        vocab = tokenizer.get_vocab()
        
        stats = {
            "vocab_size": len(vocab),
            "special_tokens": [token for token in self.special_tokens if token in vocab],
            "sample_tokens": list(vocab.keys())[:20],  # First 20 tokens
        }
        
        console.print("\n[bold blue]=== Tokenizer Statistics ===[/bold blue]")
        
        # Create a panel with statistics
        stats_text = f"""
[green]Vocabulary size:[/green] {stats['vocab_size']:,}
[green]Special tokens:[/green] {', '.join(stats['special_tokens'])}
[green]Sample tokens:[/green] {', '.join(stats['sample_tokens'][:10])}...
        """
        
        panel = Panel(stats_text.strip(), title="📊 Statistics", border_style="blue")
        console.print(panel)
        
        return stats
    
    def train_full_pipeline(self, name: str = "llvm_ir_bpe"):
        """Complete training pipeline"""
        
        console.print(Panel.fit("🚀 Starting LLVM IR BPE tokenizer training...", border_style="green"))
        
        # 1. Load corpus
        functions = self.load_corpus_text()
        
        # 2. Prepare training data
        console.print("[bold blue]Step 2/7: Preparing training data...[/bold blue]")
        training_data = self.prepare_training_data(functions)
        
        # 3. Create tokenizer
        console.print("[bold blue]Step 3/7: Creating tokenizer...[/bold blue]")
        tokenizer = self.create_tokenizer()
        
        # 4. Train tokenizer
        console.print("[bold blue]Step 4/7: Training tokenizer...[/bold blue]")
        tokenizer = self.train_tokenizer(training_data, tokenizer)
        
        # 5. Save tokenizer
        console.print("[bold blue]Step 5/7: Saving tokenizer...[/bold blue]")
        tokenizer_path = self.save_tokenizer(tokenizer, name)
        
        # 6. Test tokenizer
        console.print("[bold blue]Step 6/7: Testing tokenizer...[/bold blue]")
        self.test_tokenizer(tokenizer)
        
        # 7. Show statistics
        console.print("[bold blue]Step 7/7: Generating statistics...[/bold blue]")
        self.get_tokenizer_stats(tokenizer)
        
        # Final success message
        success_panel = Panel.fit(
            f"✅ Training Complete!\n\n"
            f"[green]Tokenizer saved to:[/green] {tokenizer_path}\n"
            f"[green]Ready for use in machine learning models![/green]",
            title="🎉 Success",
            border_style="green"
        )
        console.print(success_panel)
        
        return tokenizer, tokenizer_path


@app.command()
def train(
    corpus_path: str = typer.Argument(..., help="Path to the corpus directory"),
    output_dir: str = typer.Option("llvm_tokenizer", "--output-dir", "-o", help="Output directory for tokenizer"),
    vocab_size: int = typer.Option(50000, "--vocab-size", "-v", help="Vocabulary size"),
    min_freq: int = typer.Option(2, "--min-freq", "-f", help="Minimum frequency for tokens"),
    name: str = typer.Option("llvm_ir_bpe", "--name", "-n", help="Name for the tokenizer"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of workers for multiprocessing (default: auto)"),
    corpus_name: str = typer.Option("ir_corpus", "--corpus-name", "-cn", help="Name for the corpus directory")
):
    """
    Train a BPE tokenizer on LLVM IR corpus.
    
    This command trains a Byte-Pair Encoding (BPE) tokenizer specifically designed
    for LLVM IR code, with support for multiprocessing and rich progress visualization.
    """
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]LLVM IR BPE Tokenizer Trainer[/bold blue]\n"
        f"[yellow]Corpus:[/yellow] {corpus_path}\n"
        f"[yellow]Output:[/yellow] {output_dir}\n"
        f"[yellow]Vocab Size:[/yellow] {vocab_size:,}\n"
        f"[yellow]Workers:[/yellow] {workers or 'auto'}",
        title="🔧 Configuration",
        border_style="blue"
    ))
    
    try:
        # Create trainer
        trainer = LLVMIRTokenizerTrainer(
            corpus_path, 
            corpus_name=corpus_name, 
            output_dir=output_dir, 
            num_workers=workers
        )
        trainer.vocab_size = vocab_size
        trainer.min_frequency = min_freq
        
        # Train tokenizer
        tokenizer, tokenizer_path = trainer.train_full_pipeline(name)
        
    except FileNotFoundError as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        raise typer.Exit(1)


def main():
    # Ensure multiprocessing works on all platforms
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    app()


if __name__ == "__main__":
    main()
