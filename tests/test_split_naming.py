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
