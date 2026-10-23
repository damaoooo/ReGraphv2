"""Compare both extraction backends against one official extraction per function."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Scripts'))
from split_functions import DEFAULT_EXTRACTOR, MODULE_ID, list_defined_functions
from task3_extract import build_extraction_signature


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
    command = ['bash', str(ROOT/'Scripts/split_llvm_ir.sh'), str(bitcode), str(output)]
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
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_signature_tracks_backend_binary(bitcode, tmp_path, monkeypatch):
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
    result = subprocess.run(['bash', str(ROOT/'Scripts/split_llvm_ir.sh'), str(bitcode), str(tmp_path/'out')],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'versions differ' in result.stderr
