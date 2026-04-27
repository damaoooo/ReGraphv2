"""
Command line interface for materializing fused Task3 parquet shards.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import datasets
import typer
from rich.console import Console

try:
    from .dataset_features import get_dataset_features
except ImportError:  # Support direct script execution.
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from DataProcess.dataset_features import get_dataset_features


app = typer.Typer(help="Materialize ReGraph fused parquet shards as HuggingFace datasets")
console = Console()
SPLIT_NAMES = ("train", "validation", "test")


@app.callback()
def main():
    """Materialize fused Task3 parquet shards."""


def list_parquet_files(directory: Path) -> List[str]:
    return [str(path) for path in sorted(directory.glob("*.parquet")) if path.is_file()]


def discover_parquet_inputs(input_parquet_dir: Path) -> List[Tuple[Optional[str], List[str]]]:
    root_files = list_parquet_files(input_parquet_dir)
    if root_files:
        return [(None, root_files)]

    split_inputs: List[Tuple[Optional[str], List[str]]] = []
    preferred_split_dirs = [input_parquet_dir / split for split in SPLIT_NAMES]
    if any(path.is_dir() for path in preferred_split_dirs):
        for split in SPLIT_NAMES:
            split_dir = input_parquet_dir / split
            if not split_dir.is_dir():
                continue
            files = list_parquet_files(split_dir)
            if files:
                split_inputs.append((split, files))
        return split_inputs

    for child in sorted(input_parquet_dir.iterdir()):
        if not child.is_dir():
            continue
        files = list_parquet_files(child)
        if files:
            split_inputs.append((child.name, files))
    return split_inputs


def output_path_for_split(output_dir: Path, split: Optional[str], single_dataset: bool) -> Path:
    if split is None or single_dataset:
        return output_dir
    return output_dir / f"{split}_dataset"


@app.command("parquet")
def parquet_to_hf(
    input_parquet_dir: str = typer.Option(
        ...,
        "--input-parquet-dir",
        "-i",
        help="Directory containing final parquet shards from fused Task3",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for HuggingFace dataset(s)",
    ),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="Optional HuggingFace datasets cache directory"),
):
    """Read final parquet shards and save HuggingFace dataset directories."""
    input_root = Path(input_parquet_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not input_root.exists() or not input_root.is_dir():
        console.print(f"[red]Parquet directory not found: {input_root}[/red]")
        raise typer.Exit(code=1)

    split_inputs = discover_parquet_inputs(input_root)
    if not split_inputs:
        console.print(f"[red]No parquet files found under: {input_root}[/red]")
        raise typer.Exit(code=1)

    output_root.mkdir(parents=True, exist_ok=True)
    single_dataset = len(split_inputs) == 1 and split_inputs[0][0] is None
    features = get_dataset_features()

    for split, files in split_inputs:
        label = split or "dataset"
        target_path = output_path_for_split(output_root, split, single_dataset)
        console.print(f"[yellow]Loading split={label}, parquet_files={len(files)}[/yellow]")
        dataset = datasets.load_dataset(
            "parquet",
            data_files=files,
            split="train",
            features=features,
            cache_dir=cache_dir,
        )
        if target_path.exists():
            import shutil

            shutil.rmtree(target_path)
        dataset.save_to_disk(str(target_path))
        console.print(f"[green]Saved split={label}, rows={len(dataset)}, path={target_path}[/green]")


if __name__ == "__main__":
    app()
