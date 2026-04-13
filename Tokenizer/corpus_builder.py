#!/usr/bin/env python3
"""
ASM corpus builder for BPE tokenization
Reads ASM files and saves them as a Hugging Face dataset for tokenizer training.
"""

import json
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import typer
from datasets import Dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def clean_asm_text(asm_text: str) -> str:
    """Normalize line endings and trim surrounding blank space."""
    lines = [line.rstrip() for line in asm_text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


class ASMCorpusBuilder:
    def __init__(
        self,
        base_dir: str,
        output_dir: str = "asm_corpus",
        num_processes: Optional[int] = None,
        batch_size: int = 1000,
    ):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.num_processes = num_processes or mp.cpu_count()
        self.console = Console()
        self.batch_size = batch_size

        self.special_tokens = [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
            "<mask>",
        ]

        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_functions": 0,
            "total_lines": 0,
            "failed_list": [],
            "special_tokens_added": len(self.special_tokens),
        }
        self.stats_lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_asm_files(self) -> List[Tuple[dict, Path]]:
        """Find all ASM files in the base directory."""
        if self.base_dir.is_file():
            asm_files = [self.base_dir] if self.base_dir.suffix == ".asm" else []
            relative_base = self.base_dir.parent
        else:
            asm_files = sorted(self.base_dir.rglob("*.asm"))
            relative_base = self.base_dir

        results: List[Tuple[dict, Path]] = []
        for file_path in asm_files:
            relative_path = file_path.relative_to(relative_base)
            metadata = {
                "function_name": file_path.stem,
                "group_name": file_path.parent.name,
                "relative_path": str(relative_path),
            }
            results.append((metadata, file_path))
        return results

    @staticmethod
    def process_single_file(args: Tuple[Path, dict]) -> Tuple[List[Dict], Dict]:
        """Process a single ASM file."""
        file_path, metadata = args

        file_stats = {
            "processed_files": 0,
            "failed_files": 0,
            "total_functions": 0,
            "total_lines": 0,
            "failed_list": [],
        }

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                asm_code = handle.read()

            asm_text = clean_asm_text(asm_code)
            if not asm_text:
                return [], file_stats

            metadata_copy = dict(metadata)
            metadata_copy["file_path"] = str(file_path)
            metadata_copy["original_length"] = len(asm_code)
            metadata_copy["normalized_length"] = len(asm_text)
            metadata_copy["line_count"] = len(asm_text.splitlines())

            result = {
                "text": asm_text,
                "metadata": metadata_copy,
            }

            file_stats["processed_files"] = 1
            file_stats["total_functions"] = 1
            file_stats["total_lines"] = metadata_copy["line_count"]
            return [result], file_stats

        except Exception as exc:
            file_stats["failed_files"] = 1
            file_stats["failed_list"].append(str(file_path))
            print(f"Error processing {file_path}: {exc}")
            return [], file_stats

    def update_stats(self, file_stats: Dict):
        """Thread-safe statistics aggregation."""
        with self.stats_lock:
            for key in ["processed_files", "failed_files", "total_functions", "total_lines"]:
                self.stats[key] += file_stats[key]
            self.stats["failed_list"].extend(file_stats["failed_list"])

    @staticmethod
    def process_batch(batch_args: List[Tuple[Any, ...]]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Process a batch of ASM files."""
        batch_results = []
        aggregated_stats = {
            "processed_files": 0,
            "failed_files": 0,
            "total_functions": 0,
            "total_lines": 0,
            "failed_list": [],
        }

        for args in batch_args:
            file_path, _ = args
            try:
                results, file_stats = ASMCorpusBuilder.process_single_file(args)
                batch_results.extend(results)
                for key in ["processed_files", "failed_files", "total_functions", "total_lines"]:
                    aggregated_stats[key] += file_stats[key]
                aggregated_stats["failed_list"].extend(file_stats["failed_list"])
            except Exception:
                aggregated_stats["failed_files"] += 1
                aggregated_stats["failed_list"].append(str(file_path))

        return batch_results, aggregated_stats

    def build_corpus(self, max_files: Optional[int] = None) -> Union[Dataset, None]:
        """Build corpus from all ASM files."""
        asm_files = self.find_asm_files()
        self.console.print(f"[bold green]Found {len(asm_files)} ASM files[/bold green]")
        if max_files:
            asm_files = asm_files[:max_files]

        self.stats["total_files"] = len(asm_files)

        if not asm_files:
            self.console.print("[yellow]No ASM files to process.[/yellow]")
            return None

        self.console.print(f"[bold green]Will process {len(asm_files)} ASM files[/bold green]")
        self.console.print(
            f"[bold blue]Using {self.num_processes} processes with a batch size of {self.batch_size}[/bold blue]"
        )

        process_args = [(file_path, metadata) for metadata, file_path in asm_files]
        process_batches = [
            process_args[index : index + self.batch_size]
            for index in range(0, len(process_args), self.batch_size)
        ]

        all_data = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing files...", total=len(process_args))

            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch, batch): batch for batch in process_batches
                }

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results, batch_stats = future.result()
                        all_data.extend(batch_results)
                        self.update_stats(batch_stats)
                    except Exception as exc:
                        self.console.print(
                            f"[bold red]A whole batch of {len(batch)} files failed: {exc}[/bold red]"
                        )
                        with self.stats_lock:
                            self.stats["failed_files"] += len(batch)
                            for args in batch:
                                self.stats["failed_list"].append(str(args[0]))

                    progress.advance(task, len(batch))

        if not all_data:
            self.console.print("[red]No data to create dataset[/red]")
            return None

        return Dataset.from_list(all_data)

    def save_corpus(self, dataset: Dataset, name: str = "asm_corpus"):
        """Save the corpus in Hugging Face, JSONL, and plain text formats."""
        dataset_path = self.output_dir / name
        dataset.save_to_disk(dataset_path)

        with open(self.output_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for item in dataset:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(self.output_dir / f"{name}_text_only.txt", "w", encoding="utf-8") as handle:
            for item in dataset:
                handle.write(item["text"] + "\n\n")

        with open(self.output_dir / f"{name}_special_tokens.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "special_tokens": self.special_tokens,
                    "description": "Special tokens reserved for ASM tokenizer training",
                },
                handle,
                indent=2,
            )

        with open(self.output_dir / f"{name}_stats.json", "w", encoding="utf-8") as handle:
            json.dump(self.stats, handle, indent=2)

        print(f"Corpus saved to {self.output_dir}")
        print(f"Total functions: {len(dataset)}")
        print(f"Statistics saved to {self.output_dir / f'{name}_stats.json'}")

    def print_stats(self):
        """Print processing statistics."""
        self.console.print("\n[bold cyan]=== Corpus Building Statistics ===[/bold cyan]")
        self.console.print(f"[green]Total files found:[/green] {self.stats['total_files']}")
        self.console.print(f"[green]Successfully processed:[/green] {self.stats['processed_files']}")
        self.console.print(f"[red]Failed files:[/red] {self.stats['failed_files']}")
        self.console.print(f"[blue]Total functions extracted:[/blue] {self.stats['total_functions']}")
        self.console.print(f"[blue]Total lines of code:[/blue] {self.stats['total_lines']}")
        self.console.print(f"[yellow]Reserved special tokens:[/yellow] {', '.join(self.special_tokens)}")

        if self.stats["failed_list"]:
            self.console.print("\n[red]Failed files:[/red]")
            for failed_file in self.stats["failed_list"][:10]:
                self.console.print(f"  [dim]- {failed_file}[/dim]")
            if len(self.stats["failed_list"]) > 10:
                self.console.print(
                    f"  [dim]... and {len(self.stats['failed_list']) - 10} more[/dim]"
                )


LLVMIRCorpusBuilder = ASMCorpusBuilder


def main(
    input_dir: str = typer.Option(..., "--input-dir", "-i", help="Input directory containing ASM files"),
    output_dir: str = typer.Option("asm_corpus", "--output-dir", "-o", help="Output directory for corpus"),
    max_files: Optional[int] = typer.Option(None, "--max-files", "-m", help="Maximum number of files to process"),
    name: str = typer.Option("asm_corpus", "--name", "-n", help="Name of the corpus dataset"),
    processes: Optional[int] = typer.Option(None, "--processes", "-p", help="Number of processes to use"),
):
    """Build ASM corpus for BPE tokenizer training."""
    console = Console()

    builder = ASMCorpusBuilder(
        input_dir,
        output_dir,
        num_processes=processes,
    )

    console.print(f"[bold yellow]Building corpus with {builder.num_processes} processes...[/bold yellow]")
    dataset = builder.build_corpus(max_files=max_files)

    if dataset:
        builder.save_corpus(dataset, name)
        builder.print_stats()
        console.print("\n[bold cyan]=== Sample ASM Function ===[/bold cyan]")
        console.print(f"[dim]{dataset[0]['text'][:500]}...[/dim]")
        console.print(f"\n[yellow]Metadata:[/yellow] {dataset[0]['metadata']}")
    else:
        console.print("[red]Failed to build corpus[/red]")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    typer.run(main)
