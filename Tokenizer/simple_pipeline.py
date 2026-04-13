#!/usr/bin/env python3
"""
ASM -> BPE tokenizer pipeline
"""

import multiprocessing as mp
import os
from pathlib import Path

import typer

from .bpe_tokenizer import ASMTokenizerTrainer
from .corpus_builder import ASMCorpusBuilder


def main(
    input_dir: str = typer.Argument(..., help="Input directory with ASM files"),
    output_prefix: str = typer.Argument(
        str((Path(__file__).resolve().parent / "asm_output").resolve()),
        help="Output directory prefix",
    ),
    vocab_size: int = typer.Option(65535, help="Vocabulary size for BPE tokenizer"),
    num_processes: int = typer.Option(os.cpu_count(), help="Number of processes"),
    corpus_name: str = typer.Option("asm_corpus", help="Name for the corpus dataset"),
    tokenizer_name: str = typer.Option("asm_bpe", help="Name for the tokenizer files"),
    start_from: int = typer.Option(1, help="Start from step (1: corpus building, 2: tokenizer training)"),
    max_files: int = typer.Option(None, help="Maximum number of ASM files to process"),
):
    """ASM to BPE tokenizer pipeline."""
    corpus_dir = f"{output_prefix}_corpus"
    tokenizer_dir = f"{output_prefix}_tokenizer"

    if num_processes is None:
        num_processes = mp.cpu_count()

    typer.echo(f"Starting pipeline from step {start_from} with {num_processes} processes...")

    if start_from <= 1:
        if not os.path.exists(input_dir):
            typer.echo(f"Directory not found: {input_dir}")
            raise typer.Exit(1)

        typer.echo("Step 1: Building ASM corpus...")
        builder = ASMCorpusBuilder(input_dir, corpus_dir, num_processes)
        dataset = builder.build_corpus(max_files=max_files)
        if dataset is None:
            typer.echo("Failed to build corpus.")
            raise typer.Exit(1)
        builder.print_stats()
        typer.echo(f"Corpus built with {len(dataset)} functions.")
        builder.save_corpus(dataset, corpus_name)
        typer.echo("Step 1 completed.")
    else:
        typer.echo("Skipping step 1 (corpus building)")
        if not os.path.exists(corpus_dir):
            typer.echo(f"Corpus directory not found: {corpus_dir}")
            typer.echo("Run with --start-from 1 to build corpus first")
            raise typer.Exit(1)

    if start_from <= 2:
        typer.echo("Step 2: Training ASM tokenizer...")
        trainer = ASMTokenizerTrainer(corpus_dir, corpus_name=corpus_name, output_dir=tokenizer_dir, num_workers=num_processes)
        trainer.vocab_size = vocab_size
        _, tokenizer_path = trainer.train_full_pipeline(name=tokenizer_name)
        typer.echo("Step 2 completed.")
        typer.echo(f"Done. Tokenizer saved at: {tokenizer_path}")
    else:
        typer.echo("Skipping step 2 (tokenizer training)")


if __name__ == "__main__":
    typer.run(main)
