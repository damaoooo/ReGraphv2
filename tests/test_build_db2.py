"""Scripts/build_db2.sh resolves the caller's paths before it changes into the repository."""
from pathlib import Path
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'Scripts' / 'build_db2.sh'


def logging_runner(directory):
    """Stands in for the interpreter: records the working directory and one argument per line."""
    runner = directory / 'runner.sh'
    runner.write_text('#!/bin/bash\n{ echo "cwd=$PWD"; printf "arg=%s\\n" "$@"; echo; } >> "$RUNNER_LOG"\n')
    runner.chmod(0o755)
    return runner


def calls(log):
    return [block.splitlines() for block in log.read_text().strip().split('\n\n')]


def test_relative_paths_are_taken_from_the_callers_directory(tmp_path):
    caller = tmp_path / 'caller dir'
    (caller / 'data dir').mkdir(parents=True)
    tools = tmp_path / 'tools'
    tools.mkdir()
    log = tmp_path / 'calls.log'
    env = dict(os.environ, RUNNER_LOG=str(log), REGRAPH_PYTHON_CMD=f'bash {logging_runner(tools)}',
               TOKENIZER_PATH='tok dir/t.json')
    result = subprocess.run(['bash', str(SCRIPT), 'data dir', 'out dir', '2'], cwd=caller, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (caller / 'out dir').is_dir() and not (ROOT / 'out dir').exists()
    stages = calls(log)
    assert [lines[2] for lines in stages] == ['arg=GraphBuilder.graph_generator', 'arg=DataProcess',
                                              'arg=DataProcess.wash', 'arg=Pretrain.split_train_validation']
    paths = [line[4:] for lines in stages for line in lines[3:] if line[4:].startswith('/')]
    assert f'{caller}/data dir' in paths
    assert f'{caller}/out dir/db2_raw_dataset' in paths and f'{caller}/out dir/db2_final_set' in paths
    assert paths.count(f'{caller}/tok dir/t.json') == 2  # DataProcess and wash
    assert all(line == f'cwd={ROOT}' for lines in stages for line in lines[:1])


def test_default_interpreter_keeps_a_repository_path_with_spaces(tmp_path):
    repo = tmp_path / 're po'
    (repo / 'Scripts').mkdir(parents=True)
    shutil.copy(SCRIPT, repo / 'Scripts' / 'build_db2.sh')
    (tmp_path / 'data').mkdir()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    shutil.copy(logging_runner(bin_dir), bin_dir / 'uv')
    log = tmp_path / 'calls.log'
    env = dict(os.environ, RUNNER_LOG=str(log), PATH=f"{bin_dir}:{os.environ['PATH']}")
    env.pop('REGRAPH_PYTHON_CMD', None)
    result = subprocess.run(['bash', str(repo / 'Scripts' / 'build_db2.sh'), str(tmp_path / 'data'), str(tmp_path / 'out')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    first = calls(log)[0]
    assert first[1:4] == ['arg=run', 'arg=--project', f'arg={repo}']


def test_invalid_worker_count_creates_nothing(tmp_path):
    (tmp_path / 'data').mkdir()
    result = subprocess.run(['bash', str(SCRIPT), str(tmp_path / 'data'), str(tmp_path / 'out'), '0'],
                            capture_output=True, text=True)
    assert result.returncode == 2 and 'positive integer' in result.stderr
    assert not (tmp_path / 'out').exists()
