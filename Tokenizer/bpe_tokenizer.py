#!/usr/bin/env python3
"""
BPE tokenizer training script for ASM corpora
"""

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import typer
from datasets import load_from_disk
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

console = Console()
app = typer.Typer(help="Train a BPE tokenizer on ASM corpus")


def _prepare_single_asm(function_text: str) -> str:
    """Trim each ASM sample before training."""
    return function_text.strip()


def _write_batch_to_file(batch_data: tuple) -> str:
    """Write a batch of samples to a temporary training file."""
    batch, batch_idx, output_dir = batch_data
    temp_file = Path(output_dir) / f"temp_batch_{batch_idx}.txt"

    with open(temp_file, "w", encoding="utf-8") as handle:
        for func in batch:
            handle.write(func + "\n")

    return str(temp_file)


class ASMTokenizerTrainer:
    def __init__(
        self,
        corpus_path: str,
        corpus_name: str = "asm_corpus",
        output_dir: str = "asm_tokenizer",
        num_workers: int = None,
    ):
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_name = corpus_name
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        console.print(f"[green]Using {self.num_workers} workers for parallel processing[/green]")

        self.vocab_size = 50000
        self.min_frequency = 2
        self.special_tokens = [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
            "<mask>",
        ]

    def load_corpus_text(self) -> List[str]:
        """Load plain-text ASM samples from a saved corpus."""
        with console.status("[bold green]Loading corpus...") as status:
            text_file = self.corpus_path / f"{self.corpus_name}_text_only.txt"
            if text_file.exists():
                status.update(f"[bold green]Loading corpus from text file: {text_file}")
                with open(text_file, "r", encoding="utf-8") as handle:
                    content = handle.read()
                    functions = [func.strip() for func in content.split("\n\n") if func.strip()]
                    console.print(f"[green]Loaded {len(functions)} functions from text file[/green]")
                    return functions

            dataset_dir = self.corpus_path / self.corpus_name
            if dataset_dir.exists():
                status.update(f"[bold green]Loading corpus from Hugging Face dataset: {dataset_dir}")
                dataset = load_from_disk(str(dataset_dir))
                functions = [item["text"] for item in dataset]
                console.print(f"[green]Loaded {len(functions)} functions from dataset[/green]")
                return functions

            jsonl_file = self.corpus_path / f"{self.corpus_name}.jsonl"
            if jsonl_file.exists():
                status.update(f"[bold green]Loading corpus from JSONL file: {jsonl_file}")
                functions = []
                with open(jsonl_file, "r", encoding="utf-8") as handle:
                    for line in handle:
                        functions.append(json.loads(line)["text"])
                console.print(f"[green]Loaded {len(functions)} functions from JSONL file[/green]")
                return functions

        raise FileNotFoundError(f"No valid corpus found in {self.corpus_path}")

    def prepare_training_data(self, functions: List[str]) -> List[str]:
        """Prepare ASM text for BPE training."""
        console.print("[blue]Preparing training data...[/blue]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing functions", total=len(functions))
            prepared_data = []
            for func in functions:
                prepared = _prepare_single_asm(func)
                if prepared:
                    prepared_data.append(prepared)
                progress.advance(task)

        console.print(f"[green]Processed {len(prepared_data)} functions[/green]")
        return prepared_data

    def create_tokenizer(self) -> Tokenizer:
        """Create the raw BPE tokenizer."""
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", self.special_tokens.index("<bos>")),
                ("<eos>", self.special_tokens.index("<eos>")),
            ],
        )
        return tokenizer

    def train_tokenizer(self, training_data: List[str], tokenizer: Tokenizer) -> Tokenizer:
        """Train the tokenizer from prepared ASM samples."""
        console.print(f"[blue]Training BPE tokenizer with vocab size {self.vocab_size}[/blue]")

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        batch_size = max(1000, len(training_data) // max(self.num_workers * 4, 1))
        batches = []
        for index in range(0, len(training_data), batch_size):
            batch = training_data[index : index + batch_size]
            batch_idx = index // batch_size
            batches.append((batch, batch_idx, str(self.output_dir)))

        console.print(f"[blue]Writing {len(batches)} batches using {self.num_workers} workers...[/blue]")

        temp_files = []
        if len(batches) > 1 and self.num_workers > 1:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Writing batches", total=len(batches))
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_batch = {
                        executor.submit(_write_batch_to_file, batch_data): batch_data
                        for batch_data in batches
                    }

                    for future in as_completed(future_to_batch):
                        temp_file = future.result()
                        temp_files.append(temp_file)
                        progress.advance(task)
        else:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Writing batches", total=len(batches))
                for batch_data in batches:
                    temp_files.append(_write_batch_to_file(batch_data))
                    progress.advance(task)

        console.print(f"[green]Generated {len(temp_files)} temporary files[/green]")
        time.sleep(0.1)

        try:
            console.print("[bold blue]Training tokenizer...[/bold blue]")
            tokenizer.train(temp_files, trainer)
            console.print("[green]Tokenizer training completed[/green]")
        finally:
            with console.status("[bold yellow]Cleaning up temporary files..."):
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            console.print("[green]Temporary files cleaned up[/green]")

        return tokenizer

    def save_tokenizer(self, tokenizer: Tokenizer, name: str = "asm_bpe"):
        """Persist tokenizer, vocab, and config."""
        with console.status("[bold blue]Saving tokenizer..."):
            tokenizer_path = self.output_dir / f"{name}.json"
            tokenizer.save(str(tokenizer_path))
            console.print(f"[green]Tokenizer saved to: {tokenizer_path}[/green]")

            vocab = tokenizer.get_vocab()
            vocab_path = self.output_dir / f"{name}_vocab.json"
            with open(vocab_path, "w", encoding="utf-8") as handle:
                json.dump(vocab, handle, indent=2, ensure_ascii=False)
            console.print(f"[green]Vocabulary saved to: {vocab_path}[/green]")

            config = {
                "domain": "asm",
                "model_type": "BPE",
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "special_tokens": self.special_tokens,
                "tokenizer_path": str(tokenizer_path),
                "vocab_path": str(vocab_path),
            }

            config_path = self.output_dir / f"{name}_config.json"
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
            console.print(f"[green]Configuration saved to: {config_path}[/green]")

        return tokenizer_path

    def test_tokenizer(self, tokenizer: Tokenizer, test_samples: List[str] = None):
        """Show tokenization examples for a few ASM samples."""
        if test_samples is None:
            test_samples = [
                "push rbp",
                "mov rbp, rsp",
                "sub rsp, 20h",
                "cmp eax, [rbp+var_4]",
                "jg 0x1305",
                "call 0x1150",
                "mov rax, cs:g_sink",
            ]

        console.print("\n[bold blue]=== Tokenizer Test ===[/bold blue]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Original", style="cyan", no_wrap=False)
        table.add_column("Tokens", style="yellow", no_wrap=False)
        table.add_column("IDs", style="green", no_wrap=False)
        table.add_column("Decoded", style="white", no_wrap=False)

        for sample in test_samples:
            encoded = tokenizer.encode(sample)
            decoded = tokenizer.decode(encoded.ids)

            tokens_str = str(encoded.tokens)
            if len(tokens_str) > 60:
                tokens_str = tokens_str[:57] + "..."

            ids_str = str(encoded.ids)
            if len(ids_str) > 40:
                ids_str = ids_str[:37] + "..."

            table.add_row(
                sample,
                tokens_str,
                ids_str,
                decoded[:60] + "..." if len(decoded) > 60 else decoded,
            )

        console.print(table)

    def get_tokenizer_stats(self, tokenizer: Tokenizer):
        """Display tokenizer statistics."""
        vocab = tokenizer.get_vocab()
        stats = {
            "vocab_size": len(vocab),
            "special_tokens": [token for token in self.special_tokens if token in vocab],
            "sample_tokens": list(vocab.keys())[:20],
        }

        console.print("\n[bold blue]=== Tokenizer Statistics ===[/bold blue]")
        panel = Panel(
            (
                f"[green]Vocabulary size:[/green] {stats['vocab_size']:,}\n"
                f"[green]Special tokens:[/green] {', '.join(stats['special_tokens'])}\n"
                f"[green]Sample tokens:[/green] {', '.join(stats['sample_tokens'][:10])}..."
            ),
            title="Statistics",
            border_style="blue",
        )
        console.print(panel)
        return stats

    def train_full_pipeline(self, name: str = "asm_bpe"):
        """Run the full ASM tokenizer training pipeline."""
        console.print(Panel.fit("Starting ASM BPE tokenizer training...", border_style="green"))

        functions = self.load_corpus_text()
        console.print("[bold blue]Step 2/7: Preparing training data...[/bold blue]")
        training_data = self.prepare_training_data(functions)

        console.print("[bold blue]Step 3/7: Creating tokenizer...[/bold blue]")
        tokenizer = self.create_tokenizer()

        console.print("[bold blue]Step 4/7: Training tokenizer...[/bold blue]")
        tokenizer = self.train_tokenizer(training_data, tokenizer)

        console.print("[bold blue]Step 5/7: Saving tokenizer...[/bold blue]")
        tokenizer_path = self.save_tokenizer(tokenizer, name)

        console.print("[bold blue]Step 6/7: Testing tokenizer...[/bold blue]")
        self.test_tokenizer(tokenizer)

        console.print("[bold blue]Step 7/7: Generating statistics...[/bold blue]")
        self.get_tokenizer_stats(tokenizer)

        console.print(
            Panel.fit(
                f"Training complete.\n\n[green]Tokenizer saved to:[/green] {tokenizer_path}",
                title="Success",
                border_style="green",
            )
        )
        return tokenizer, tokenizer_path


LLVMIRTokenizerTrainer = ASMTokenizerTrainer


@app.command()
def train(
    corpus_path: str = typer.Argument(..., help="Path to the ASM corpus directory"),
    output_dir: str = typer.Option("asm_tokenizer", "--output-dir", "-o", help="Output directory for tokenizer"),
    vocab_size: int = typer.Option(50000, "--vocab-size", "-v", help="Vocabulary size"),
    min_freq: int = typer.Option(2, "--min-freq", "-f", help="Minimum frequency for tokens"),
    name: str = typer.Option("asm_bpe", "--name", "-n", help="Name for the tokenizer"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of workers for multiprocessing"),
    corpus_name: str = typer.Option("asm_corpus", "--corpus-name", "-cn", help="Name for the corpus dataset"),
):
    """Train a BPE tokenizer on an ASM corpus."""
    console.print(
        Panel.fit(
            "[bold blue]ASM BPE Tokenizer Trainer[/bold blue]\n"
            f"[yellow]Corpus:[/yellow] {corpus_path}\n"
            f"[yellow]Output:[/yellow] {output_dir}\n"
            f"[yellow]Vocab Size:[/yellow] {vocab_size:,}\n"
            f"[yellow]Workers:[/yellow] {workers or 'auto'}",
            title="Configuration",
            border_style="blue",
        )
    )

    try:
        trainer = ASMTokenizerTrainer(
            corpus_path,
            corpus_name=corpus_name,
            output_dir=output_dir,
            num_workers=workers,
        )
        trainer.vocab_size = vocab_size
        trainer.min_frequency = min_freq
        trainer.train_full_pipeline(name)
    except FileNotFoundError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Unexpected error: {exc}[/red]")
        raise typer.Exit(1)


def main():
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    app()


if __name__ == "__main__":
    main()
