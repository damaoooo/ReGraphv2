"""Content-bound completion records for the file-based pipeline."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from concurrent.futures import wait, FIRST_COMPLETED
from typing import Any


def iter_completed_tasks(executor, function, items, max_pending_tasks):
    """Yield (completed future, submitted item), bounding the submission queue."""
    iterator = iter(items)
    pending = {}
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < max_pending_tasks:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(function, item)] = item
        if pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future, pending.pop(future)


def compute_file_sha256(path: str | Path) -> str:
    """Hash a file without loading its entire contents into memory."""
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@lru_cache(maxsize=None)
def get_tool_identity(tool: str) -> list[str]:
    """Return the executable path and version output used to invalidate caches."""
    path = shutil.which(tool)
    if path is None:
        raise FileNotFoundError(tool)
    result = subprocess.run([path, '--version'], capture_output=True, text=True, check=True)
    return [os.path.realpath(path), result.stdout, result.stderr]


def build_input_signature(input_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Describe the input contents and configuration that produced an artifact."""
    return {'schema': 1, 'input_sha256': compute_file_sha256(input_path), 'config': config}


def is_cached_output_valid(
    marker_path: str | Path,
    expected_signature: dict[str, Any],
    output_path: str | Path,
) -> bool:
    """Require both a matching completion record and an unchanged output file."""
    try:
        completion_record = json.loads(Path(marker_path).read_text())
        return (
            completion_record['signature'] == expected_signature
            and os.path.getsize(output_path) > 0
            and completion_record['output_sha256'] == compute_file_sha256(output_path)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def create_temporary_output_path(output_path: str | Path) -> str:
    """Create a sibling temporary file, allowing publication via os.replace."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{output_path.name}.', suffix='.tmp', dir=output_path.parent,
    )
    os.close(descriptor)
    return temporary_path


def write_completion_record(
    marker_path: str | Path,
    input_signature: dict[str, Any],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Publish a completion record only after the output is ready."""
    temporary_marker_path = create_temporary_output_path(marker_path)
    completion_record = {
        'signature': input_signature,
        'output_sha256': compute_file_sha256(output_path),
        'metadata': metadata or {},
    }
    try:
        Path(temporary_marker_path).write_text(json.dumps(completion_record))
        os.replace(temporary_marker_path, marker_path)
    finally:
        Path(temporary_marker_path).unlink(missing_ok=True)
