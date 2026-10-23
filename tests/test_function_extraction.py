"""Function extraction: both backends match llvm-extract; excluded IDA segments stay out of the samples."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Scripts'))
from split_functions import DEFAULT_EXTRACTOR, MODULE_ID, lift_state_path, list_defined_functions
import task3_extract
from task3_extract import build_extraction_signature, is_extraction_complete


@pytest.fixture
def bitcode(tmp_path):
    for tool in ('llvm-as', 'llvm-extract', 'llvm-nm'):
        if not shutil.which(tool):
            pytest.skip(f'{tool} unavailable')
    source = tmp_path / 'input.ll'
    lines = ['source_filename = "fixture.ll"', '@g = internal global i32 17',
             'declare i32 @external(i32)',
             'define internal i32 @helper(i32 %x) { %v = load i32, ptr @g\n %r = add i32 %x, %v\n ret i32 %r }',
             'define weak i32 @weak_func(i32 %x) { %r = call i32 @external(i32 %x)\n ret i32 %r }']
    for index in range(130):
        lines.append(f'define i32 @f{index:03}(i32 %x) {{ %r = call i32 @helper(i32 %x)\n ret i32 %r }}')
    source.write_text('\n'.join(lines) + '\n')
    output = source.with_suffix('.bc')
    subprocess.run(['llvm-as', str(source), '-o', str(output)], check=True)
    return output


@pytest.mark.parametrize('backend', ['cpp', 'cli'])
def test_extraction_matches_official_tool(bitcode, tmp_path, backend):
    if backend == 'cpp' and not DEFAULT_EXTRACTOR.exists():
        pytest.skip('build LLVMTools first')
    output = tmp_path / backend
    env = dict(os.environ, REGRAPH_PYTHON=sys.executable,
               REGRAPH_SPLIT_FUNCS=str(DEFAULT_EXTRACTOR if backend == 'cpp' else tmp_path/'missing'))
    command = ['bash', str(ROOT/'Scripts/split_llvm_ir.sh'), str(bitcode), str(output), '--no-segment-filter']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    names = list_defined_functions(bitcode)
    rows = ['OriginalFunctionName,HashedFileName']
    for name in names:
        filename = hashlib.sha1(name.encode()).hexdigest() + '.ll'
        rows.append(f'"{name}","{filename}"')
        baseline = subprocess.check_output(['llvm-extract', '-S', f'--func={name}', str(bitcode), '-o', '-'], text=True)
        actual = (output/filename).read_text()
        assert actual == MODULE_ID + baseline.partition('\n')[2], name
    assert (output/'function_map.csv').read_text() == '\n'.join(rows) + '\n'
    rerun = tmp_path / f'{backend}-rerun'
    subprocess.run(command[:-2] + [str(rerun), '--no-segment-filter'], env=env, check=True, capture_output=True)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {p.name: p.read_bytes() for p in rerun.iterdir()}


def test_signature_tracks_backend_binary(bitcode, tmp_path, monkeypatch):
    write_task1_record(bitcode, 'ghidra')
    binary = tmp_path/'split_funcs'
    monkeypatch.setenv('REGRAPH_SPLIT_FUNCS', str(binary))
    fallback = build_extraction_signature(str(bitcode))
    binary.write_bytes(b'build one')
    first_build = build_extraction_signature(str(bitcode))
    binary.write_bytes(b'build two')
    second_build = build_extraction_signature(str(bitcode))
    assert fallback != first_build != second_build


def test_mismatched_tool_fails_instead_of_silent_fallback(bitcode, tmp_path):
    tool = tmp_path/'split_funcs'
    tool.write_text('#!/bin/sh\necho "split_funcs LLVM 0.0.0"\n')
    tool.chmod(0o755)
    env = dict(os.environ, REGRAPH_SPLIT_FUNCS=str(tool), REGRAPH_PYTHON=sys.executable)
    result = subprocess.run(['bash', str(ROOT/'Scripts/split_llvm_ir.sh'), str(bitcode), str(tmp_path/'out'),
                             '--no-segment-filter'], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'versions differ' in result.stderr


def write_lift_state(bitcode, segments, **fields):
    """A Task 1 lift state for the fixture module; `segments` maps LLVM names to their IDA segment."""
    source = bitcode.with_suffix('.ll')
    state = {'schema': 1, 'status': 'ok',
             'output': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
             'functions': [{'name': name, 'llvm_name': name, 'status': 'lifted', 'segment': segment}
                           for name, segment in segments.items()]}
    state.update(fields)
    lift_state_path(bitcode).write_text(json.dumps(state))
    return lift_state_path(bitcode)


def write_task1_record(bitcode, backend, output_sha256=None):
    """Task 1's completion record of the module's source; by default it describes the current `X.ll`."""
    source = bitcode.with_suffix('.ll')
    record = {'signature': {'schema': 1, 'input_sha256': '0' * 64, 'config': {'backend': backend}},
              'output_sha256': output_sha256 or hashlib.sha256(source.read_bytes()).hexdigest()}
    bitcode.with_suffix('.ll.task1.done.json').write_text(json.dumps(record))


def run_split(bitcode, output, *arguments, backend='cli', tmp_path=None):
    extractor = DEFAULT_EXTRACTOR if backend == 'cpp' else Path('/nonexistent/split_funcs')
    env = dict(os.environ, REGRAPH_PYTHON=sys.executable, REGRAPH_SPLIT_FUNCS=str(extractor))
    return subprocess.run(['bash', str(ROOT/'Scripts/split_llvm_ir.sh'), str(bitcode), str(output), *arguments],
                          env=env, capture_output=True, text=True)


def extracted_names(output):
    return {line.split(',')[0].strip('"') for line in (output/'function_map.csv').read_text().splitlines()[1:]}


def text_segments(bitcode, **overrides):
    return {name: '.text' for name in list_defined_functions(bitcode)} | overrides


@pytest.mark.parametrize('backend', ['cpp', 'cli'])
def test_excluded_segments_are_not_extracted(bitcode, tmp_path, backend):
    if backend == 'cpp' and not DEFAULT_EXTRACTOR.exists():
        pytest.skip('build LLVMTools first')
    state = write_lift_state(bitcode, text_segments(bitcode, f000='.plt', f001='.plt.got', f002='.init'))
    output = tmp_path / 'out'
    result = run_split(bitcode, output, '--lift-state', state, backend=backend)
    assert result.returncode == 0, result.stderr
    kept = set(list_defined_functions(bitcode)) - {'f000', 'f001', 'f002'}
    assert extracted_names(output) == kept
    assert {p.name for p in output.glob('*.ll')} == {hashlib.sha1(n.encode()).hexdigest() + '.ll' for n in kept}
    assert 'Skipped 3 functions in excluded segments' in result.stderr


def test_segment_filter_must_be_chosen(bitcode, tmp_path):
    result = run_split(bitcode, tmp_path / 'out')
    assert result.returncode != 0 and '--lift-state' in result.stderr


@pytest.mark.parametrize('problem, fields, message', [
    ('failed lift', {'status': 'error'}, "lift status is 'error'"),
    ('other module', {'output': {'path': '/elsewhere.ll', 'sha256': '0' * 64}}, 'was not written for'),
    ('null segment', None, 'records no segment'),
    ('empty segment', '', 'records no segment'),
    ('unknown function', {'functions': [{'name': 'gone', 'llvm_name': 'gone', 'status': 'lifted', 'segment': '.text'}]},
     'not defined there'),
    ('not a lift state', {'functions': [{'status': 'lifted'}]}, 'not a Task 1 lift state'),
])
def test_lift_state_that_does_not_describe_the_module_is_rejected(bitcode, tmp_path, problem, fields, message):
    if fields is None or fields == '':
        state = write_lift_state(bitcode, text_segments(bitcode, f000=fields))
    else:
        state = write_lift_state(bitcode, text_segments(bitcode), **fields)
    result = run_split(bitcode, tmp_path / 'out', '--lift-state', state)
    assert result.returncode != 0 and message in result.stderr, (problem, result.stderr)


def test_non_empty_output_directory_is_refused(bitcode, tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'old.ll').write_text('earlier run')
    result = run_split(bitcode, output, '--no-segment-filter')
    assert result.returncode != 0 and 'not empty' in result.stderr
    assert [p.name for p in output.iterdir()] == ['old.ll']


def test_all_functions_excluded_leaves_only_the_map(bitcode, tmp_path):
    state = write_lift_state(bitcode, {name: '.plt' for name in list_defined_functions(bitcode)})
    output = tmp_path / 'out'
    assert run_split(bitcode, output, '--lift-state', state).returncode == 0
    assert [p.name for p in output.iterdir()] == ['function_map.csv']
    assert extracted_names(output) == set()


def run_task3(bitcode, monkeypatch):
    monkeypatch.setenv('REGRAPH_SPLIT_FUNCS', '/nonexistent/split_funcs')
    output = bitcode.with_name(bitcode.stem + '_functions')
    return task3_extract.extract_functions(str(bitcode), str(output)), output


def test_task3_filters_ida_modules_and_resumes_on_the_lift_state(bitcode, monkeypatch):
    write_task1_record(bitcode, 'ida')
    write_lift_state(bitcode, text_segments(bitcode, f000='.plt'))
    (success, _, stderr), output = run_task3(bitcode, monkeypatch)
    assert success, stderr
    assert 'f000' not in extracted_names(output)
    assert json.loads((output / '.complete.json').read_text())['segment_filter']['applied'] is True
    assert is_extraction_complete(str(bitcode), str(output))
    write_lift_state(bitcode, text_segments(bitcode, f001='.plt'))
    assert not is_extraction_complete(str(bitcode), str(output))


def test_task3_keeps_ghidra_modules_unfiltered_and_says_so(bitcode, monkeypatch):
    write_task1_record(bitcode, 'ghidra')
    (success, _, stderr), output = run_task3(bitcode, monkeypatch)
    assert success, stderr
    assert extracted_names(output) == set(list_defined_functions(bitcode))
    assert json.loads((output / '.complete.json').read_text())['segment_filter'] == {
        'applied': False, 'reason': 'ghidra backend records no segments'}


@pytest.mark.parametrize('backend', ['ida', None])
def test_task3_refuses_ida_or_unknown_modules_without_lift_state(bitcode, monkeypatch, backend):
    if backend:
        write_task1_record(bitcode, backend)
    (success, _, stderr), output = run_task3(bitcode, monkeypatch)
    assert not success and 're-run Task 1' in stderr
    assert not output.exists()


def test_lift_state_needs_the_source_module(bitcode, tmp_path):
    """Name membership alone is no provenance: without `X.ll` the state cannot be bound to the module."""
    state = write_lift_state(bitcode, text_segments(bitcode, f000='.plt'), output={'sha256': 'unrelated'})
    bitcode.with_suffix('.ll').unlink()
    result = run_split(bitcode, tmp_path / 'out', '--lift-state', state)
    assert result.returncode != 0 and 'is missing' in result.stderr


def test_task3_ignores_a_task1_record_of_another_source(bitcode, monkeypatch):
    write_task1_record(bitcode, 'ghidra', output_sha256='0' * 64)
    (success, _, stderr), output = run_task3(bitcode, monkeypatch)
    assert not success and 're-run Task 1' in stderr
    assert not output.exists()


def test_task3_resume_and_fresh_extraction_agree_on_a_changed_source(bitcode, monkeypatch):
    write_task1_record(bitcode, 'ida')
    write_lift_state(bitcode, text_segments(bitcode))
    (success, _, stderr), output = run_task3(bitcode, monkeypatch)
    assert success, stderr
    assert is_extraction_complete(str(bitcode), str(output))
    source = bitcode.with_suffix('.ll')
    source.write_text(source.read_text() + '\n; edited after the lift\n')
    assert not is_extraction_complete(str(bitcode), str(output))
    assert run_task3(bitcode, monkeypatch)[0][0] is False
