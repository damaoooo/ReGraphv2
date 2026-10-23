"""Small real-LLVM regressions for orchestration and restart safety."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Scripts'))
from pipeline_cache import is_cached_output_valid, write_completion_record, build_input_signature


def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / 'Scripts' / script), *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture
def source(tmp_path):
    for tool in ('opt', 'clang', 'llvm-dis', 'llvm-nm', 'llvm-extract', 'llvm-as'):
        if not shutil.which(tool):
            pytest.skip(f'{tool} unavailable')
    path = tmp_path / 'x64_probe.ll'
    write_ir(path, 1)
    return path


def write_ir(path, value):
    write_module(path, f'target triple = "x86_64-unknown-linux-gnu"\ndefine i32 @probe() {{ ret i32 {value} }}\n')


def write_module(path, text):
    """Write a hand-made module with the lift state an IDA lift would publish (every function in .text)."""
    path.write_text(text)
    functions = [{'name': name, 'llvm_name': name, 'status': 'lifted', 'segment': '.text'}
                 for name in re.findall(r'^define [^@]*@(\w+)\(', text, re.M)]
    state = {'schema': 1, 'status': 'ok', 'functions': functions,
             'output': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    Path(str(path) + '.lift.json').write_text(json.dumps(state))


def reopt(source, *args):
    result = run('task2_reoptimize.py', '--input-path', source.parent,
                 '--opt-level', 'Oc', '--workers', 1, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_pipeline_start3_and_start4_forwarding(source):
    reopt(source)
    result = run('pipeline.py', 'pipeline', '--output', source.parent, '--start-from', 3, '--workers', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    # Use an unrecognizable filename so the explicit arch must reach Task 4.
    source.with_suffix('.bc').rename(source.parent / 'unknown.bc')
    result = run('pipeline.py', 'pipeline', '--output', source.parent, '--start-from', 4,
                 '--enable-recompile', '--arch', 'm64', '--workers', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (source.parent / 'unknown.re').is_file()


@pytest.mark.parametrize('script,suffix', [('task2_reoptimize.py', '.ll'),
                                         ('task3_extract.py', '.bc'), ('task4_recompile.py', '.bc')])
def test_failure_exit(source, script, suffix):
    source.with_suffix(suffix).write_text('invalid LLVM')
    result = run(script, '--input-path', source.parent, '--workers', 1)
    assert result.returncode != 0
    assert 'Failed: 1' in result.stdout


def test_changed_source_and_damaged_output_invalidate_resume(source):
    reopt(source)
    cached = reopt(source, '--resume')
    assert 'Skipping 1' in cached.stdout
    write_ir(source, 2)
    reopt(source, '--resume')
    bc = source.with_suffix('.bc')
    text = subprocess.check_output(['llvm-dis', str(bc), '-o', '-'], text=True)
    assert 'ret i32 2' in text
    bc.write_bytes(b'broken')
    reopt(source, '--resume')
    assert 'ret i32 2' in subprocess.check_output(['llvm-dis', str(bc), '-o', '-'], text=True)


def test_option_switch_does_not_reuse_other_outputs(source):
    source.write_text('target triple = "x86_64-unknown-linux-gnu"\ndefine i32 @probe(i32 %x) { %a = add i32 %x, 0\nret i32 %a }\n')
    reopt(source)
    result = run('task2_reoptimize.py', '--input-path', source.parent, '--opt-level', 'O0', '--workers', 1)
    assert result.returncode == 0
    result = reopt(source, '--resume')
    assert 'Skipping 1' not in result.stdout


def test_extraction_refresh_partial_resume_and_removed_functions(source):
    reopt(source)
    folder = source.with_suffix('').with_name('x64_probe_functions')
    def extract(*args):
        result = run('task3_extract.py', '--input-path', source.parent, '--workers', 1, *args)
        assert result.returncode == 0, result.stdout + result.stderr
        return result
    extract()
    assert 'Skipping 1' in extract('--resume').stdout
    next(folder.glob('*.ll')).unlink()
    assert 'Skipping 1' not in extract('--resume').stdout
    (folder / 'obsolete.ll').write_text('old')
    write_ir(source, 3)
    reopt(source)
    extract()
    assert not (folder / 'obsolete.ll').exists()
    assert 'ret i32 3' in next(folder.glob('*.ll')).read_text()


def test_empty_module_extraction(source):
    write_module(source, 'target triple = "x86_64-unknown-linux-gnu"\n')
    reopt(source)
    result = run('task3_extract.py', '--input-path', source.parent, '--workers', 1)
    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_and_weak_functions_are_extracted(source):
    write_module(source, 'target triple = "x86_64-unknown-linux-gnu"\n'
                         'define internal i32 @hidden() { ret i32 1 }\n'
                         'define weak i32 @weak_func() { ret i32 2 }\n')
    reopt(source)
    result = run('task3_extract.py', '--input-path', source.parent, '--workers', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    folder = source.parent / 'x64_probe_functions'
    assert len(list(folder.glob('*.ll'))) == 2


def test_task4_rebuilds_after_bitcode_update(source):
    reopt(source)
    def compile(*args):
        result = run('task4_recompile.py', '--input-path', source.parent, '--workers', 1, *args)
        assert result.returncode == 0, result.stdout + result.stderr
        return result
    compile()
    assert 'Skipping 1' in compile('--resume').stdout
    old = source.with_suffix('.re').read_bytes()
    write_ir(source, 77)
    reopt(source)
    assert 'Skipping 1' not in compile('--resume').stdout
    assert old != source.with_suffix('.re').read_bytes()


def test_task1_worker_records_only_successful_nonempty_output(tmp_path, monkeypatch):
    import task1_lift as task1
    from types import SimpleNamespace
    src, out = tmp_path / 'input.i64', tmp_path / 'out.ll'
    src.write_text('fake database')
    monkeypatch.setattr(task1, 'LOG_PATH', str(tmp_path / 'lift.log'))
    def success(*args, **kwargs):
        out.write_text('fake lifted output')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(task1.subprocess, 'run', success)
    args = (str(src), str(out), 'ida', '', '', '', 'host', 1, 60, 300, False, False)
    config = {'lifter': 'v1'}
    assert task1.lift_single_file(*args, cache_config=config)[0]
    marker = str(out) + '.task1.done.json'
    assert is_cached_output_valid(marker, build_input_signature(src, config), out)
    assert not is_cached_output_valid(marker, build_input_signature(src, {'lifter': 'v2'}), out)
    monkeypatch.setattr(task1.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout='', stderr=''))
    assert not task1.lift_single_file(*args, cache_config=config)[0]
    assert not Path(marker).exists()


def test_timeout_kills_shell_children(tmp_path):
    from utils import run_command
    import time
    pidfile = tmp_path / 'child.pid'
    code = ('import subprocess,time; from pathlib import Path; '
            f'p=subprocess.Popen(["sleep","30"]); Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(30)')
    ok, _, error = run_command([sys.executable, '-c', code], timeout=0.5)
    assert not ok and 'timed out' in error
    pid = int(pidfile.read_text())
    for _ in range(50):
        stat = Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2] == 'Z':
            break
        time.sleep(0.02)
    else:
        pytest.fail('command descendant survived timeout')


def test_cache_binds_configuration_and_output(tmp_path):
    src, output, marker = [tmp_path / x for x in ('src', 'out', 'done')]
    src.write_text('input')
    output.write_text('output')
    expected = build_input_signature(src, {'version': 1})
    write_completion_record(marker, expected, output)
    assert is_cached_output_valid(marker, expected, output)
    assert not is_cached_output_valid(marker, build_input_signature(src, {'version': 2}), output)
    output.write_text('changed')
    assert not is_cached_output_valid(marker, expected, output)


@pytest.mark.parametrize('start', [0, 4, 5])
def test_invalid_start(source, start):
    result = run('pipeline.py', 'pipeline', '--output', source.parent, '--start-from', start)
    assert result.returncode != 0
