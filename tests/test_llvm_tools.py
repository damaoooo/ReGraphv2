"""Smoke-test the unified LLVM build through the Python graph entrypoint."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_unified_plugins_generate_graph_artifacts(tmp_path):
    from Utils.utils import DEFAULT_CFG_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH
    from GraphBuilder.graph_generator import process_one_file

    if not shutil.which('opt'):
        pytest.skip('opt is required')
    for plugin in (DEFAULT_CFG_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH):
        assert Path(plugin).parent == ROOT / 'LLVMTools/build/lib'
        if not Path(plugin).is_file():
            pytest.skip('build LLVMTools first')

    source = tmp_path / 'probe.ll'
    source.write_text('define i32 @probe(i32 %x) {\nentry:\n'
                      '  %v = add i32 %x, 1\n  ret i32 %v\n}\n')
    result = process_one_file(str(source))
    assert result is not None
    for artifact in result:
        assert Path(artifact).is_file() and Path(artifact).stat().st_size > 0
    for ir_path in result[1:3]:
        subprocess.run(['opt', '-passes=verify', '-disable-output', ir_path], check=True)
    assert 'digraph' in Path(result[3]).read_text()
    assert 'digraph' in Path(result[4]).read_text()
