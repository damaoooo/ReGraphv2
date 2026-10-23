"""Default worker counts follow the CPUs allocated to the process, not the node's CPU count."""
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Utils.utils import DEFAULT_TOKENIZER_PATH, available_cpu_count


@pytest.mark.parametrize('affinity, slurm, expected', [
    (12, None, 12),       # affinity mask (cgroups / Slurm binding)
    (12, '4', 4),         # Slurm allocation caps it
    (12, '64', 12),       # never more than the process may use
    (12, 'abc', 12),      # malformed value ignored
    (None, None, 1),      # unknown: still at least one worker
    (None, '6', 6),
])
def test_available_cpu_count(monkeypatch, affinity, slurm, expected):
    monkeypatch.setattr(os, 'process_cpu_count', lambda: affinity, raising=False)
    if slurm is None:
        monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    else:
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', slurm)
    assert available_cpu_count() == expected


def test_falls_back_to_the_affinity_mask(monkeypatch):
    monkeypatch.delattr(os, 'process_cpu_count', raising=False)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1, 2})
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    assert available_cpu_count() == 3


def test_default_tokenizer_lives_in_the_repository():
    path = Path(DEFAULT_TOKENIZER_PATH)
    assert path.is_file() and ROOT in path.parents


def test_wash_rejects_a_non_positive_worker_count():
    import typer
    from typer.testing import CliRunner
    from DataProcess.wash import main
    app = typer.Typer()
    app.command()(main)
    result = CliRunner().invoke(app, ['--dataset-path', 'x', '--output-path', 'y', '--num-proc', '0'])
    assert result.exit_code != 0
