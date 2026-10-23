"""Graph generation correctness: DDG instruction positions, resumable runs, failure reporting."""
from pathlib import Path
import shutil
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from GraphBuilder.ddg_graph_builder import DataDependencyGraphBuilder, split_instructions

TWO_STORES = '''define i32 @probe(ptr %p, i1 %c, i32 %x) {
entry:
  %sum = add i32 %x, 1
  br i1 %c, label %left, label %right
left:
  store i32 %sum, ptr %p
  br label %done
right:
  store i32 %sum, ptr %p
  br label %done
done:
  %v = load i32, ptr %p
  ret i32 %v
}
'''

SWITCH = '''@table = global [4 x i32] zeroinitializer

define i32 @pick(i32 %x) {
entry:
  %slot = getelementptr [4 x i32], ptr @table, i32 0, i32 %x
  %base = load i32, ptr %slot
  switch i32 %x, label %other [
    i32 0, label %zero
    i32 1, label %one
  ]
zero:
  ret i32 %base
one:
  %plus = add i32 %base, 1
  ret i32 %plus
other:
  %sum = add i32 %base, %x
  ret i32 %sum
}
'''


@pytest.fixture(scope='module')
def tools():
    from Utils.utils import DEFAULT_CFG_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH, DEFAULT_TOKENIZER_PATH
    if not shutil.which('opt'):
        pytest.skip('opt is required')
    for plugin in (DEFAULT_CFG_SO_PATH, DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH):
        if not Path(plugin).is_file():
            pytest.skip('build LLVMTools first')
    from Tokenizer.ir_tokenizer import load_tokenizer
    return load_tokenizer(DEFAULT_TOKENIZER_PATH)


def write_function(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{name}.ll'
    path.write_text(text)
    return path


def test_split_instructions_keeps_switch_together_and_duplicates_apart():
    text = 'define i32 @f() {\n"@0":\nstore i32 1, ptr @g\nswitch i32 %x, label %"@1" [\ni32 0, label %"@2"\n]\n' \
           '"@1":\nstore i32 1, ptr @g\nret i32 0\n}\n\ndeclare i32 @h()'
    spans = split_instructions(text)
    assert [s[2] for s in spans] == ['store i32 1, ptr @g', 'switch i32 %x, label %"@1" [ i32 0, label %"@2" ]',
                                     'store i32 1, ptr @g', 'ret i32 0']
    assert spans[0][0] != spans[2][0]
    assert text[spans[1][0]:spans[1][1]].endswith(']')


def ddg_positions(tokenizer, function_file: Path):
    from GraphBuilder.graph_generator import process_one_file
    from Tokenizer.normalizer import normalize_file
    outputs = process_one_file(str(function_file))
    assert outputs is not None
    builder = DataDependencyGraphBuilder(tokenizer)
    edges = builder.generate_ddg_matrix(outputs[4], outputs[2], outputs[1])
    normalized = normalize_file(outputs[1])
    offsets = tokenizer(normalized, return_offsets_mapping=True)['offset_mapping']
    spans = {text: [] for _, _, text in split_instructions(normalized)}
    for start, end, text in split_instructions(normalized):
        first = next(i for i, (s, e) in enumerate(offsets) if s < e and e > start)
        spans[text].append(first)
    return edges, spans, builder


def test_identical_instructions_in_two_branches_get_their_own_positions(tools, tmp_path):
    edges, spans, builder = ddg_positions(tools, write_function(tmp_path / 'b_functions', 'probe', TWO_STORES))
    store_starts = next(starts for text, starts in spans.items() if text.startswith('store '))
    assert len(store_starts) == 2 and store_starts[0] != store_starts[1]
    mapped = sorted(r[0] for k, r in builder.id_range_map.items()
                    if builder.normalized_instrumented_dict[k].startswith('store '))
    assert mapped == sorted(store_starts)
    # both stores take the same value: each one has its own edge into the graph
    assert {e[2] for e in edges} >= set(store_starts) or {e[0] for e in edges} >= set(store_starts)


def test_switch_maps_to_the_switch_instruction(tools, tmp_path):
    _, spans, builder = ddg_positions(tools, write_function(tmp_path / 'b_functions', 'pick', SWITCH))
    switch_start = next(starts[0] for text, starts in spans.items() if text.startswith('switch '))
    switch_ids = [k for k, text in builder.normalized_instrumented_dict.items() if text.startswith('switch ')]
    assert len(switch_ids) == 1 and builder.id_range_map[switch_ids[0]][0] == switch_start


def test_instrumented_ir_that_does_not_match_fails(tools, tmp_path):
    from GraphBuilder.graph_generator import process_one_file
    outputs = process_one_file(str(write_function(tmp_path / 'b_functions', 'probe', TWO_STORES)))
    purified = Path(outputs[1])
    purified.write_text(purified.read_text().replace('%sum = add i32 %x, 1', '%sum = add i32 %x, 2'))
    with pytest.raises(ValueError, match='does not match'):
        DataDependencyGraphBuilder(tools).generate_ddg_matrix(outputs[4], outputs[2], outputs[1])


def test_missing_artifacts_are_not_a_successful_sample(tools, tmp_path):
    from DataProcess.file_processor import FileProcessor
    source = write_function(tmp_path / 'b_functions', 'probe', TWO_STORES)
    result = FileProcessor(tools).process_single_file(str(source), str(tmp_path / 'missing.ll'),
                                                      str(tmp_path / 'missing_inst.ll'),
                                                      str(tmp_path / 'missing_cfg.dot'), str(tmp_path / 'missing_ddg.dot'))
    assert result.success is False


def rows(db: Path):
    with sqlite3.connect(db) as conn:
        return {r[0]: r[1:] for r in conn.execute(
            'SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot, signature FROM results')}


def test_rerun_keeps_artifacts_and_only_redoes_what_changed(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    functions = tmp_path / 'bin_functions'
    first = write_function(functions, 'a', TWO_STORES)
    second = write_function(functions, 'b', SWITCH)
    gone = write_function(functions, 'c', TWO_STORES.replace('@probe', '@probe2'))
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=True, jobs=1)
    db = tmp_path / 'results.db'
    before = rows(db)
    assert set(before) == {str(first), str(second), str(gone)}
    # artifacts stay out of the function directory Task 3 publishes as an exact file set
    assert sorted(p.name for p in functions.iterdir()) == ['a.ll', 'b.ll', 'c.ll']
    for row in before.values():
        assert all(Path(p).parent == tmp_path / 'bin_graphs' and Path(p).is_file() for p in row[:4])

    def stamps(table):
        return {p: Path(p).stat().st_mtime_ns for row in table.values() for p in row[:4] if Path(p).exists()}

    # an unchanged rerun with the default cleanup deletes nothing and regenerates nothing
    before_stamps = stamps(before)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=True, jobs=1)
    assert rows(db) == before and stamps(before) == before_stamps

    # new content at the same path, a deleted artifact, and a removed function
    first.write_text(TWO_STORES.replace('add i32 %x, 1', 'add i32 %x, 3'))
    Path(before[str(second)][2]).unlink()
    gone.unlink()
    (functions / 'function_map.csv').write_text('changed\n')
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=True, jobs=1)
    after = rows(db)
    assert set(after) == {str(first), str(second)}
    assert after[str(first)][4] != before[str(first)][4]
    regenerated = stamps(after)
    assert all(regenerated[p] != before_stamps.get(p) for p in after[str(first)][:4] + after[str(second)][:4])
    assert not any(Path(p).exists() for p in before[str(gone)][:4])


def test_failed_function_leaves_no_row_and_no_artifacts(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    functions = tmp_path / 'bin_functions'
    good = write_function(functions, 'a', TWO_STORES)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1)
    stale = rows(tmp_path / 'results.db')[str(good)]
    good.write_text('define i32 @probe( {\n')  # no longer valid IR: opt fails
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1)
    assert rows(tmp_path / 'results.db') == {}
    assert not any(Path(p).exists() for p in stale[:4])


def test_chunk_error_fails_the_run(tools, tmp_path, monkeypatch):
    import GraphBuilder.graph_generator as generator
    write_function(tmp_path / 'bin_functions', 'a', TWO_STORES)

    def broken(path):
        raise RuntimeError('worker crashed')
    monkeypatch.setattr(generator, 'process_one_file', broken)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1) is False
