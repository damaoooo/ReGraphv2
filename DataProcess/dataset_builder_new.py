"""Build and publish the single supported Hugging Face dataset format."""
import os
from pathlib import Path
import tempfile

from .parallel_processor import create_hf_dataset


class DatasetBuilder:
    """Read current graph records and publish only a fully saved dataset."""

    def __init__(self, tokenizer_path: str, db_file: str, num_processes: int | None = None):
        self.tokenizer_path = str(Path(tokenizer_path).resolve())
        self.db_file = str(Path(db_file).resolve())
        self.num_processes = num_processes if num_processes is not None else (os.cpu_count() or 1)
        if self.num_processes < 1:
            raise ValueError("num_processes must be positive")

    def process_dataset(self, output_path: str) -> Path:
        output_directory = Path(output_path).resolve()
        if not Path(self.db_file).is_file():
            raise FileNotFoundError(self.db_file)
        if output_directory.exists() and (
            not output_directory.is_dir() or any(output_directory.iterdir())
        ):
            raise FileExistsError(f"Output must be a new or empty directory: {output_directory}")

        output_directory.parent.mkdir(parents=True, exist_ok=True)
        # A fresh Arrow cache prevents reuse after the database, IR, or tokenizer changes.
        with tempfile.TemporaryDirectory(
            prefix=f".{output_directory.name}-build-", dir=output_directory.parent
        ) as temporary_directory:
            workspace = Path(temporary_directory)
            dataset = create_hf_dataset(
                self.db_file, self.tokenizer_path, self.num_processes, str(workspace / "cache")
            )
            staged_output = workspace / "dataset"
            dataset.save_to_disk(str(staged_output))
            sample_count = len(dataset)
            del dataset
            # Do not overwrite a nonempty dataset, including one created during this build.
            if output_directory.exists():
                output_directory.rmdir()
            os.replace(staged_output, output_directory)

        print(f"Saved {sample_count} samples to {output_directory}")
        return output_directory
