"""The production dataset builder has one HF path for one or many workers."""
from pathlib import Path
import sqlite3
import sys
import subprocess

import datasets
import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DataProcess.cli import app
from DataProcess.builder import DatasetBuilder
from GraphBuilder.graph_generator import init_database, process_one_file
from Utils.utils import DEFAULT_TOKENIZER_PATH, DEFAULT_CFG_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH


@pytest.fixture
def graph_database(tmp_path):
    for plugin in (DEFAULT_CFG_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH):
        if not Path(plugin).exists():
            pytest.skip('build LLVMTools first')
    db = tmp_path / 'results.db'
    init_database(str(db))
    rows = []
    for name in ('zulu', 'alpha'):
        source = tmp_path / f'{name}.ll'
        source.write_text(f'define i32 @{name}(i32 %x) {{\nentry:\n  %v = add i32 %x, 1\n  ret i32 %v\n}}\n')
        row = process_one_file(str(source))
        assert row is not None
        rows.append(row)
    with sqlite3.connect(db) as conn:
        conn.executemany('INSERT INTO results (input_path,purify_path,instrumented_path,cfg_dot,ddg_dot) VALUES (?,?,?,?,?)', rows)
    return db


def test_same_hf_data_for_one_and_two_workers(graph_database, tmp_path):
    from DataProcess import builder as parallel_processor
    # Multiple chunks exercise ordered multiprocessing, not just a single task.
    original = parallel_processor.CHUNK_SIZE
    parallel_processor.CHUNK_SIZE = 1
    try:
        outputs = []
        for workers in (1, 2):
            output = tmp_path / f'hf_{workers}'
            DatasetBuilder(DEFAULT_TOKENIZER_PATH, str(graph_database), workers).process_dataset(str(output))
            outputs.append(datasets.load_from_disk(str(output)).to_dict())
        assert outputs[0] == outputs[1]
        assert len(outputs[0]['file_path']) == 2
        assert outputs[0]['file_path'] == sorted(outputs[0]['file_path'])
        assert all(outputs[0]['input_ids'])
        assert not list(tmp_path.glob('.*-build-*'))
    finally:
        parallel_processor.CHUNK_SIZE = original


def test_existing_dataset_is_not_overwritten(graph_database, tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        DatasetBuilder(DEFAULT_TOKENIZER_PATH, str(graph_database), 1).process_dataset(str(output))
    assert (output / 'sentinel').read_text() == 'keep'


def test_save_failure_propagates_and_does_not_publish(graph_database, tmp_path, monkeypatch):
    import DataProcess.builder as builder_module
    class BrokenDataset:
        def save_to_disk(self, path):
            raise OSError('disk full')
    monkeypatch.setattr(builder_module, 'create_hf_dataset', lambda *args: BrokenDataset())
    output = tmp_path / 'failed'
    with pytest.raises(OSError, match='disk full'):
        DatasetBuilder(DEFAULT_TOKENIZER_PATH, str(graph_database), 1).process_dataset(str(output))
    assert not output.exists()
    assert not list(tmp_path.glob('.*-build-*'))


def test_cli_only_exposes_single_output_mode():
    result = CliRunner().invoke(app, ['--help'])
    assert result.exit_code == 0
    for removed_option in ('--use-hf', '--parallel', '--no-cache', '--batch-size', '--no-cleanup'):
        assert removed_option not in result.stdout


def test_package_entrypoint():
    result = subprocess.run(
        [sys.executable, '-m', 'DataProcess', '--help'],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'INPUT_DIR OUTPUT_DIR' in result.stdout


def test_empty_database_fails_without_publishing(tmp_path):
    db = tmp_path/'results.db'
    init_database(str(db))
    output = tmp_path/'empty'
    with pytest.raises(Exception, match='generating|No valid samples'):
        DatasetBuilder(DEFAULT_TOKENIZER_PATH, str(db), 1).process_dataset(str(output))
    assert not output.exists()


def test_single_worker_initializes_once_for_multiple_chunks(monkeypatch):
    from DataProcess import builder as processor
    initializations = []
    monkeypatch.setattr(processor, 'initialize_worker', initializations.append)
    monkeypatch.setattr(processor, 'iter_database_chunks', lambda path: iter([[1], [2], [3]]))
    monkeypatch.setattr(processor, 'process_chunk', lambda rows: rows)
    assert list(processor.iter_processed_chunks('unused', 'tokenizer', 1)) == [[1], [2], [3]]
    assert initializations == ['tokenizer']


def test_parallel_submissions_are_bounded_and_ordered(monkeypatch):
    from DataProcess import builder as processor
    outstanding = 0
    peak = 0

    class FakeFuture:
        def __init__(self, rows):
            self.rows = rows

        def result(self):
            nonlocal outstanding
            outstanding -= 1
            return self.rows

    class FakeExecutor:
        def __init__(self, **kwargs):
            assert kwargs['initializer'] is processor.initialize_worker

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, rows):
            nonlocal outstanding, peak
            outstanding += 1
            peak = max(peak, outstanding)
            return FakeFuture(rows)

    chunks = [[index] for index in range(25)]
    monkeypatch.setattr(processor, 'iter_database_chunks', lambda path: iter(chunks))
    monkeypatch.setattr(processor, 'ProcessPoolExecutor', FakeExecutor)
    assert list(processor.iter_processed_chunks('unused', 'tokenizer', 2)) == chunks
    assert peak == 4
    assert outstanding == 0
