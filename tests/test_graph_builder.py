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
    """A worker exception (here: the function file vanishes after it was listed) fails the whole run."""
    import GraphBuilder.graph_generator as generator
    function = write_function(tmp_path / 'bin_functions', 'a', TWO_STORES)
    completed = generator._completed_inputs

    def remove_after_listing(db_path, valid_files):
        function.unlink()
        return completed(db_path, valid_files)
    # Patched in the parent, which lists and schedules; spawned workers do not inherit patches
    monkeypatch.setattr(generator, '_completed_inputs', remove_after_listing)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1) is False


MANY_EDGES = 'define i32 @many(ptr %p, i32 %x) {\nentry:\n' + ''.join(
    f'  %a{i} = add i32 %x, {i}\n  store i32 %a{i}, ptr %p\n' for i in range(40)) + '  %v = load i32, ptr %p\n  ret i32 %v\n}\n'


def test_ddg_edge_order_does_not_depend_on_the_hash_seed(tools, tmp_path):
    """Edges used to come out of a set, so their order changed with PYTHONHASHSEED from process to process."""
    import json
    import os
    import subprocess
    from GraphBuilder.graph_generator import process_one_file
    outputs = process_one_file(str(write_function(tmp_path / 'b_functions', 'many', MANY_EDGES)))
    script = ('import json, sys; sys.path.insert(0, sys.argv[1])\n'
              'from GraphBuilder.ddg_graph_builder import DataDependencyGraphBuilder\n'
              'from Tokenizer.ir_tokenizer import load_tokenizer\n'
              'from Utils.utils import DEFAULT_TOKENIZER_PATH\n'
              'b = DataDependencyGraphBuilder(load_tokenizer(DEFAULT_TOKENIZER_PATH))\n'
              'print(json.dumps(b.generate_ddg_matrix(*sys.argv[2:5])))\n')
    runs = []
    for seed in ('1', '2', '3'):
        result = subprocess.run([sys.executable, '-c', script, str(ROOT), outputs[4], outputs[2], outputs[1]],
                                env=dict(os.environ, PYTHONHASHSEED=seed), capture_output=True, text=True, check=True)
        runs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert len(runs[0]) > 40 and runs[0] == runs[1] == runs[2]


def test_cfg_binary_search_matches_a_linear_scan(tools):
    from GraphBuilder.cfg_graph_builder import CFGGraphBuilder
    text = 'define i32 @f() {\n"@0":\n  %a = add i32 1, 2\n  br label %"@1"\n"@1":\n  ret i32 %a\n}\n'
    offsets = tools(text, return_offsets_mapping=True)['offset_mapping']

    def linear(start_idx, end_idx):  # the previous implementation
        token_start = token_end = None
        for token_idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if token_start is None and end > start_idx:
                token_start = token_idx
            if start < end_idx:
                token_end = token_idx
            elif token_end is not None:
                break
        return None if token_start is None or token_end is None else (token_start, token_end)

    index = CFGGraphBuilder._index_tokens(offsets)
    for start in range(len(text) + 1):
        for end in range(start, len(text) + 2):
            assert CFGGraphBuilder._char_range_to_token_range(None, index, start, end) == linear(start, end), (start, end)


def test_file_processor_shares_one_encoding_with_both_graphs(tools, tmp_path):
    from DataProcess.file_processor import FileProcessor
    from GraphBuilder.cfg_graph_builder import CFGGraphBuilder
    from GraphBuilder.graph_generator import process_one_file
    from Tokenizer.normalizer import normalize_file
    source, purified, instrumented, cfg_dot, ddg_dot = process_one_file(
        str(write_function(tmp_path / 'b_functions', 'pick', SWITCH)))
    result = FileProcessor(tools).process_single_file(source, purified, instrumented, cfg_dot, ddg_dot)
    assert result.success
    assert result.input_ids == tools(normalize_file(purified))['input_ids']
    assert result.cfg_graph == CFGGraphBuilder(tools, purified, cfg_dot).build_cfg_edges()
    assert result.ddg_graph == DataDependencyGraphBuilder(tools).generate_ddg_matrix(ddg_dot, instrumented, purified)


def test_file_cache_sees_a_function_rewritten_in_place(tools, tmp_path):
    """Re-extraction can keep function_map.csv and the file count; the cache must still notice."""
    import GraphBuilder.graph_generator as generator
    functions = tmp_path / 'bin_functions'
    small = write_function(functions, 'a', 'define i32 @probe() {\nentry:\n  ret i32 0\n}\n')
    (functions / 'function_map.csv').write_text('OriginalFunctionName,HashedFileName\n"probe","a.ll"\n')
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=True, jobs=1)
    assert rows(tmp_path / 'results.db') == {}  # too small: filtered out
    small.write_text(TWO_STORES)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=True, jobs=1)
    assert set(rows(tmp_path / 'results.db')) == {str(small)}


def test_unreadable_function_fails_the_prefilter(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    write_function(tmp_path / 'bin_functions', 'a', TWO_STORES)
    (tmp_path / 'bin_functions' / 'b.ll').write_bytes(b'define i32 @f() {\n\xff\xfe\n}\n')
    with pytest.raises(ValueError, match='not UTF-8'):
        generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1)


# Passes the size prefilter, but opt rejects it (use of an undefined value)
INVALID = 'define i32 @broken(i32 %x) {\nentry:\n  %y = add i32 %x, %undefined\n  %z = mul i32 %y, 2\n  ret i32 %z\n}\n'


def failing_opt(tmp_path):
    """An opt that reports the real version but fails every pass run, like a plugin it cannot load."""
    real = shutil.which('opt')
    bin_dir = tmp_path / 'fake_bin'
    bin_dir.mkdir()
    script = bin_dir / 'opt'
    script.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then exec {real} --version; fi\n'
                      'echo "simulated plugin load failure" >&2\nexit 1\n')
    script.chmod(0o755)
    return bin_dir


def test_broken_toolchain_fails_before_touching_results(tools, tmp_path, monkeypatch, capsys):
    import os
    import GraphBuilder.graph_generator as generator
    functions = tmp_path / 'data' / 'bin_functions'
    keep = write_function(functions, 'a', TWO_STORES)
    edited = write_function(functions, 'b', SWITCH)
    assert generator.pipeline(str(tmp_path / 'data'), cleanup=True, use_cache=False, jobs=1)
    before = rows(tmp_path / 'data' / 'results.db')
    stamps = {p: Path(p).stat().st_mtime_ns for row in before.values() for p in row[:4]}
    edited.write_text(TWO_STORES.replace('add i32 %x, 1', 'add i32 %x, 9'))  # would be regenerated
    monkeypatch.setenv('PATH', f"{failing_opt(tmp_path)}:{os.environ['PATH']}")
    assert generator.pipeline(str(tmp_path / 'data'), cleanup=True, use_cache=False, jobs=1) is False
    assert rows(tmp_path / 'data' / 'results.db') == before
    assert {p: Path(p).stat().st_mtime_ns for p in stamps} == stamps
    output = capsys.readouterr().out
    assert 'Toolchain check failed' in output and 'simulated plugin load failure' in output


def test_missing_plugins_name_the_build_command(tmp_path, monkeypatch, capsys):
    import GraphBuilder.graph_generator as generator
    monkeypatch.setattr(generator, 'DEFAULT_CFG_SO_PATH', str(tmp_path / 'missing.so'))
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1) is False
    assert 'cmake --build LLVMTools/build' in capsys.readouterr().out.replace('\n', ' ')


def test_rejected_functions_are_skipped_and_reruns_still_succeed(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    functions = tmp_path / 'bin_functions'
    good = write_function(functions, 'a', TWO_STORES)
    write_function(functions, 'b', INVALID)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1)
    assert set(rows(tmp_path / 'results.db')) == {str(good)}
    # the invalid function is retried on every run; with the rest up to date that is not a failure
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1)


def test_every_attempted_function_failing_fails_the_run(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    write_function(tmp_path / 'bin_functions', 'b', INVALID)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1) is False
    assert rows(tmp_path / 'results.db') == {}


def test_empty_input_is_not_a_failure(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    (tmp_path / 'bin_functions').mkdir()
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1)


def test_unwritable_artifact_directory_fails_the_run(tools, tmp_path):
    import os
    import GraphBuilder.graph_generator as generator
    if os.geteuid() == 0:
        pytest.skip('root can write anywhere')
    write_function(tmp_path / 'bin_functions', 'a', TWO_STORES)
    artifacts = tmp_path / 'bin_graphs'
    artifacts.mkdir()
    artifacts.chmod(0o500)
    try:
        assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1) is False
    finally:
        artifacts.chmod(0o700)


def test_jobs_also_bound_the_prefilter(tools, tmp_path, monkeypatch):
    import GraphBuilder.graph_generator as generator
    budgets = []

    class RecordingFilter:
        def __init__(self, num_processes):
            budgets.append(num_processes)

        def filter_files_parallel(self, files):
            return files
    monkeypatch.setattr(generator, 'FileFilter', RecordingFilter)
    write_function(tmp_path / 'bin_functions', 'a', TWO_STORES)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=3)
    assert budgets == [3]


def test_output_write_failure_stops_the_run_and_keeps_the_old_row(tools, tmp_path):
    """A read-only existing artifact makes opt fail on valid IR: an output failure, not a rejected function."""
    import os
    import GraphBuilder.graph_generator as generator
    if os.geteuid() == 0:
        pytest.skip('root can write anywhere')
    functions = tmp_path / 'bin_functions'
    write_function(functions, 'a', TWO_STORES)
    b = write_function(functions, 'b', SWITCH)
    assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1)
    before = rows(tmp_path / 'results.db')
    purified = Path(before[str(b)][0])
    purified.chmod(0o444)
    b.write_text(SWITCH.replace('add i32 %base, 1', 'add i32 %base, 5'))  # b must be regenerated; a is up to date
    try:
        assert generator.pipeline(str(tmp_path), cleanup=True, use_cache=False, jobs=1) is False
    finally:
        purified.chmod(0o644)
    assert rows(tmp_path / 'results.db') == before  # the old row stays until a run can replace it


def test_pass_failures_are_rejections_only_for_invalid_ir(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    invalid = write_function(tmp_path / 'bin_functions', 'bad', INVALID)
    valid = write_function(tmp_path / 'bin_functions', 'good', TWO_STORES)
    assert generator._pass_failed('metadata stripping', str(invalid), 'detail') is None
    with pytest.raises(generator.PassExecutionError, match='failed on valid IR'):
        generator._pass_failed('metadata stripping', str(valid), 'Permission denied')


def test_a_killed_verifier_is_not_a_rejection(tools, tmp_path, monkeypatch):
    """returncode -9 (e.g. the OOM killer) says nothing about the input: stop the run instead of skipping."""
    import subprocess
    import GraphBuilder.graph_generator as generator
    invalid = write_function(tmp_path / 'bin_functions', 'bad', INVALID)
    monkeypatch.setattr(generator.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args, -9, '', ''))
    with pytest.raises(generator.PassExecutionError, match='killed by signal 9'):
        generator._pass_failed('DDG generation', str(invalid), 'pass failed')


def test_a_killed_pass_is_not_a_rejection_even_for_invalid_ir(tools, tmp_path):
    import GraphBuilder.graph_generator as generator
    invalid = write_function(tmp_path / 'bin_functions', 'bad', INVALID)
    with pytest.raises(generator.PassExecutionError, match='killed by signal 9'):
        generator._pass_failed('metadata stripping', str(invalid), 'terminated', returncode=-9)


def test_a_pass_killed_mid_run_keeps_the_rows(tools, tmp_path, monkeypatch):
    """An opt killed while regenerating b (sibling a up to date) must not delete b's row or report success."""
    import os
    import GraphBuilder.graph_generator as generator
    functions = tmp_path / 'data' / 'bin_functions'
    write_function(functions, 'a', TWO_STORES)
    b = write_function(functions, 'b', SWITCH)
    assert generator.pipeline(str(tmp_path / 'data'), cleanup=True, use_cache=False, jobs=1)
    before = rows(tmp_path / 'data' / 'results.db')
    b.write_text(SWITCH.replace('add i32 %base, 1', 'add i32 %base, 5'))
    bin_dir = tmp_path / 'killing_bin'
    bin_dir.mkdir()
    killer = bin_dir / 'opt'
    killer.write_text(f'#!/bin/sh\ncase "$*" in *b.ll*) kill -9 $$ ;; esac\nexec {shutil.which("opt")} "$@"\n')
    killer.chmod(0o755)
    monkeypatch.setenv('PATH', f"{bin_dir}:{os.environ['PATH']}")
    assert generator.pipeline(str(tmp_path / 'data'), cleanup=True, use_cache=False, jobs=1) is False
    assert rows(tmp_path / 'data' / 'results.db') == before
