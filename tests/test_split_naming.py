"""Positive pairs come from (origin binary, function name); origin is the binary name without its build variant."""
import hashlib
from pathlib import Path
import sys

import datasets
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Pretrain.split_train_validation import build_positive_indices, parse_binary_directory


@pytest.mark.parametrize('naming, name, expected', [
    ('dataset1', 'arm32-gcc-4.8-O0_clambc', ('arm32-gcc-4.8_clambc', 'O0')),
    ('dataset1', 'x64-clang-3.5-O3_openssl', ('x64-clang-3.5_openssl', 'O3')),
    ('dataset2', 'x86-64_coreutils-8.32-O0_pinky', ('x86-64_coreutils-8.32_pinky', 'O0')),
    ('dataset2', 'x86-64_coreutils-8.32-subobf_join', ('x86-64_coreutils-8.32_join', 'subobf')),
    ('dataset2', 'mips-32_ImageMagick-7.0.10-27-O2_convert', ('mips-32_ImageMagick-7.0.10-27_convert', 'O2')),
    ('binarycorp', 'agensgraph-git-bloom.so-Os-3ad76dc09f4f95595f6989901b7f4dd7', ('agensgraph-git-bloom.so', 'Os')),
])
def test_each_convention_removes_the_build_variant(naming, name, expected):
    assert parse_binary_directory(name, naming) == expected


@pytest.mark.parametrize('naming, name', [
    ('dataset2', 'aes.o'),                                   # irregular names in Dataset-2
    ('dataset2', 'libmicrohttpd_la-connection.o'),
    ('dataset1', 'x86-64_coreutils-8.32-O0_pinky'),          # another dataset's convention
    ('binarycorp', 'arm32-gcc-4.8-O0_clambc'),
    ('binarycorp', 'agensgraph-git-bloom.so-Os-not-a-hash'),
])
def test_names_off_the_convention_are_rejected(naming, name):
    assert parse_binary_directory(name, naming) is None


def test_positives_pair_one_function_across_variants_of_one_binary(tmp_path):
    binaries = {  # binary directory -> functions it defines
        'x86-64_coreutils-8.32-O0_pinky': ['main', 'usage'],
        'x86-64_coreutils-8.32-O3_pinky': ['main', 'usage'],
        'x86-64_coreutils-8.32-subobf_pinky': ['main'],
        'x86-64_coreutils-8.32-O0_fmt': ['main'],     # same name, other program: not a positive
        'arm-64_coreutils-8.32-O0_pinky': ['main'],   # other architecture: another origin
        'aes.o': ['main'],                            # irregular name: dropped
    }
    paths = []
    for binary, functions in binaries.items():
        directory = tmp_path / 'coreutils' / f'{binary}_functions'
        directory.mkdir(parents=True)
        rows = ['OriginalFunctionName,HashedFileName']
        for function in functions:
            hashed = hashlib.sha1(f'{binary}/{function}'.encode()).hexdigest() + '.ll'
            rows.append(f'"{function}","{hashed}"')
            paths.append(str(directory / hashed))
        (directory / 'function_map.csv').write_text('\n'.join(rows) + '\n')
    dataset = datasets.Dataset.from_dict({'file_path': paths})

    positive_map, info = build_positive_indices(dataset, str(tmp_path), 'dataset2')

    names = {row.original_idx: (row.origin_binary_name, row.function_name) for row in info.itertuples()}
    groups = {frozenset([anchor, *positives]) for anchor, positives in positive_map.items()}
    by_key = lambda key: frozenset(i for i, k in names.items() if k == key)
    assert groups == {by_key(('coreutils/x86-64_coreutils-8.32_pinky', 'main')),
                      by_key(('coreutils/x86-64_coreutils-8.32_pinky', 'usage'))}
    assert len(by_key(('coreutils/x86-64_coreutils-8.32_pinky', 'main'))) == 3
    assert not any('aes.o' in path for path in info['file_path'])
    assert len(info) == len(paths) - 1


def write_dataset2(root, programs=12):
    """`programs` binaries, each built at O0 and O3, each defining main and usage."""
    paths = []
    for index in range(programs):
        for variant in ('O0', 'O3'):
            directory = root / 'coreutils' / f'x86-64_coreutils-8.32-{variant}_prog{index}_functions'
            directory.mkdir(parents=True)
            rows = ['OriginalFunctionName,HashedFileName']
            for function in ('main', 'usage'):
                hashed = hashlib.sha1(f'{directory.name}/{function}'.encode()).hexdigest() + '.ll'
                rows.append(f'"{function}","{hashed}"')
                paths.append(str(directory / hashed))
            (directory / 'function_map.csv').write_text('\n'.join(rows) + '\n')
    datasets.Dataset.from_dict({'file_path': paths, 'input_ids': [[2, 3]] * len(paths)}).save_to_disk(str(root / 'raw'))


def run_split(root, seed):
    import json
    from typer.testing import CliRunner
    from Pretrain.split_train_validation import main
    import typer
    app = typer.Typer()
    app.command()(main)
    output = f'out_{seed}'
    # relative paths on purpose: the command changes into the output directory
    result = CliRunner().invoke(app, ['raw', '--base-path', '.', '--naming', 'dataset2', '--train-ratio', '0.5',
                                      '--random-seed', str(seed), '--output-dir', output])
    assert result.exit_code == 0, result.output
    return json.loads((root / output / 'split_indices.json').read_text())


def test_relative_paths_and_the_seed_decide_the_split(tmp_path, monkeypatch):
    write_dataset2(tmp_path)
    monkeypatch.chdir(tmp_path)
    first, again, other = run_split(tmp_path, 1), run_split(tmp_path, 1), run_split(tmp_path, 2)
    assert first == again
    assert first != other
    assert len(first['train']) + len(first['validation']) == 48


def write_binary(root, binary, functions, with_map=True, map_functions=None):
    directory = root / 'pkg' / f'{binary}_functions'
    directory.mkdir(parents=True)
    rows, paths = ['OriginalFunctionName,HashedFileName'], []
    for function in functions:
        hashed = hashlib.sha1(f'{binary}/{function}'.encode()).hexdigest() + '.ll'
        paths.append(str(directory / hashed))
        if map_functions is None or function in map_functions:
            rows.append(f'"{function}","{hashed}"')
    if with_map:
        (directory / 'function_map.csv').write_text('\n'.join(rows) + '\n')
    return paths


def test_missing_function_map_is_an_error(tmp_path):
    paths = write_binary(tmp_path, 'x64-gcc-4.8-O0_demo', ['f']) + write_binary(tmp_path, 'x64-gcc-4.8-O3_demo', ['f'], with_map=False)
    with pytest.raises(ValueError, match='1 binaries have no function_map.csv'):
        build_positive_indices(datasets.Dataset.from_dict({'file_path': paths}), None, 'dataset1')


def test_function_missing_from_its_map_is_an_error(tmp_path):
    paths = (write_binary(tmp_path, 'x64-gcc-4.8-O0_demo', ['f', 'g'])
             + write_binary(tmp_path, 'x64-gcc-4.8-O3_demo', ['f', 'g'], map_functions={'f'}))
    with pytest.raises(ValueError, match='1 functions are missing'):
        build_positive_indices(datasets.Dataset.from_dict({'file_path': paths}), None, 'dataset1')


def test_binaries_off_the_convention_need_no_map(tmp_path):
    paths = (write_binary(tmp_path, 'x64-gcc-4.8-O0_demo', ['f']) + write_binary(tmp_path, 'x64-gcc-4.8-O3_demo', ['f'])
             + write_binary(tmp_path, 'aes.o', ['f'], with_map=False))
    positive_map, info = build_positive_indices(datasets.Dataset.from_dict({'file_path': paths}), None, 'dataset1')
    assert len(info) == 2 and len(positive_map) == 2


def test_no_positive_pairs_is_an_error(tmp_path):
    paths = write_binary(tmp_path, 'x64-gcc-4.8-O0_demo', ['f']) + write_binary(tmp_path, 'x64-gcc-4.8-O3_other', ['f'])
    with pytest.raises(ValueError, match='No positive pairs'):
        build_positive_indices(datasets.Dataset.from_dict({'file_path': paths}), None, 'dataset1')


def test_function_names_are_read_verbatim(tmp_path):
    paths = write_binary(tmp_path, 'x64-gcc-4.8-O0_demo', ['NA', 'nan']) + write_binary(tmp_path, 'x64-gcc-4.8-O3_demo', ['NA', 'nan'])
    positive_map, info = build_positive_indices(datasets.Dataset.from_dict({'file_path': paths}), None, 'dataset1')
    assert sorted(set(info['function_name'])) == ['NA', 'nan'] and len(positive_map) == 4
