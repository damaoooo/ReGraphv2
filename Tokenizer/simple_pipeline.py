#!/usr/bin/env python3
"""
LLVM IR -> BPE Tokenizer Pipeline
"""

import os
import typer
import multiprocessing as mp
from normalizer import LLVMIRNormalizer
from corpus_builder import LLVMIRCorpusBuilder  
from bpe_tokenizer import LLVMIRTokenizerTrainer

def main(
    input_dir: str = typer.Argument(..., help="Input directory with LLVM IR files"),
    output_dir: str = typer.Argument("output", help="Output directory prefix"),
    vocab_size: int = typer.Option(10000, help="Vocabulary size for BPE tokenizer"),
    num_processes: int = typer.Option(os.cpu_count(), help="Number of processes (default: CPU count)"),
    corpus_name: str = typer.Option("ir_corpus", help="Name for the corpus directory, defaults to 'ir_corpus' if not specified"),
    start_from: int = typer.Option(1, help="Start from step (1: corpus building, 2: tokenizer training)"),
    max_files: int = typer.Option(None, help="Maximum number of LLVM IR files to process (default: all files)"),
):
    """LLVM IR to BPE Tokenizer Pipeline"""
    
    corpus_dir = f"{output_dir}_corpus"
    tokenizer_dir = f"{output_dir}_tokenizer"
    
    # Set default num_processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    typer.echo(f"🚀 Starting pipeline from step {start_from} with {num_processes} processes...")
    
    # Step 1: Build corpus
    if start_from <= 1:
        if not os.path.exists(input_dir):
            typer.echo(f"❌ Directory not found: {input_dir}")
            raise typer.Exit(1)
            
        typer.echo("📦 Step 1: Building corpus...")
        builder = LLVMIRCorpusBuilder(input_dir, corpus_dir, num_processes)
        dataset = builder.build_corpus(max_files=max_files)
        builder.print_stats()
        typer.echo(f"📦 Corpus built with {len(dataset)} functions.")
        builder.save_corpus(dataset, corpus_name)
        typer.echo("✅ Step 1 completed!")
    else:
        typer.echo("⏭️  Skipping step 1 (corpus building)")
        if not os.path.exists(corpus_dir):
            typer.echo(f"❌ Corpus directory not found: {corpus_dir}")
            typer.echo("💡 Run with --start-from 1 to build corpus first")
            raise typer.Exit(1)
    
    # Step 2: Train tokenizer
    if start_from <= 2:
        typer.echo("🤖 Step 2: Training tokenizer...")
        trainer = LLVMIRTokenizerTrainer(corpus_dir, output_dir=tokenizer_dir, num_workers=num_processes)
        trainer.vocab_size = vocab_size
        tokenizer, tokenizer_path = trainer.train_full_pipeline()
        typer.echo("✅ Step 2 completed!")
        typer.echo(f"✅ Done! Tokenizer saved at: {tokenizer_path}")
    else:
        typer.echo("⏭️  Skipping step 2 (tokenizer training)")

if __name__ == "__main__":
    typer.run(main)
