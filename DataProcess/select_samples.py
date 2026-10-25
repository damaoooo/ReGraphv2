"""Dataset-wide sample selection over the train/validation/test splits of one dataset, before
Pretrain.split_train_validation pairs positives. Follows the Dataset-1 construction of Marcelli et al.
(USENIX Security '22), whose Steps 1-2 (>= 5 basic blocks, opcode-hash dedup) Scripts/select_functions.py
applies at extraction time:

  IR dedup  keep only the first sample of each distinct untruncated input_ids sequence (the normalized IR)
            over all splits, in split-name then function-path order
  Step 5    drop function names shared across splits: from train those also in validation or test,
            from test those also in validation
  Step 6    drop samples whose (binary, function name) group has a single member in its split

A binary is <project>/<origin> under the --naming convention (dataset1-xm: one group per binary over every
architecture, compiler, version and optimization); function names come from the function_map.csv beside
each sample's file. Writes each split's selected rows to <output-dir>/<split>.

    python -m DataProcess.select_samples --naming dataset1-xm --output-dir sel \
        --split train=train_raw --split validation=validation_raw --split test=test_raw
"""
import hashlib
import os
os.environ.setdefault('USE_TORCH', '0')
from typing import Dict, List

import numpy as np
import pandas as pd
import typer
from Utils.utils import available_cpu_count


def function_names(file_paths: pd.Series) -> pd.Series:
    """Original function name of each sample, from the function_map.csv in its file's directory."""
    directories = file_paths.map(os.path.dirname)
    names = pd.Series(index=file_paths.index, dtype=object)
    for directory, rows in file_paths.groupby(directories):
        mapping = pd.read_csv(os.path.join(directory, 'function_map.csv'), dtype=str, keep_default_na=False)
        lookup = dict(zip(mapping['HashedFileName'], mapping['OriginalFunctionName']))
        names[rows.index] = rows.map(lambda path: lookup.get(os.path.basename(path)))
    missing = names.isna()
    if missing.any():
        raise ValueError(f'{missing.sum()} samples are missing from their function_map.csv, '
                         f'e.g. {file_paths[missing].iloc[0]}')
    return names


def binary_groups(file_paths: pd.Series, naming: str) -> pd.Series:
    """<project>/<origin> of each sample's binary under `naming`; NaN if its binary does not follow it."""
    from Pretrain.split_train_validation import parse_binary_directory

    parts = file_paths.str.split('/')
    directories = parts.str[-2].str.removesuffix('_functions')
    origins = {name: parse_binary_directory(name, naming) for name in directories.unique()}
    return parts.str[-3] + '/' + directories.map(lambda name: origins[name] and origins[name][0])


def ir_digest(input_ids) -> bytes:
    return hashlib.sha256(np.asarray(input_ids, dtype=np.int32).tobytes()).digest()


def select_samples(splits: Dict[str, pd.DataFrame], naming: str) -> Dict[str, np.ndarray]:
    """Boolean keep mask per split.

    Each frame holds one row per sample, in the split's row order, with `file_path` and `ir_digest`
    (ir_digest of the untruncated input_ids). Prints the effect of every step.
    """
    frames = {}
    for split, frame in splits.items():
        frame = frame[['file_path', 'ir_digest']].copy()
        frame['function_name'] = function_names(frame['file_path'])
        frame['binary'] = binary_groups(frame['file_path'], naming)
        frame['keep'] = True
        frames[split] = frame

    seen = set()
    for split in sorted(frames):
        frame = frames[split]
        order = frame['file_path'].sort_values(kind='stable').index
        first = np.zeros(len(frame), dtype=bool)
        positions = frame.index.get_indexer(order)
        for position, digest in zip(positions, frame['ir_digest'].to_numpy()[positions]):
            if digest not in seen:
                seen.add(digest)
                first[position] = True
        frame['keep'] &= first
    report('IR dedup', frames)

    names = {split: set(frame.loc[frame['keep'], 'function_name']) for split, frame in frames.items()}
    common = lambda a, b: names.get(a, set()) & names.get(b, set())
    for split, others in (('train', ('validation', 'test')), ('test', ('validation',))):
        if split in frames:
            shared = set().union(*(common(split, other) for other in others))
            frames[split]['keep'] &= ~frames[split]['function_name'].isin(shared)
    report('Step 5 (cross-split names)', frames)

    for frame in frames.values():
        kept = frame[frame['keep']]
        sizes = kept.groupby(['binary', 'function_name'])['file_path'].transform('size')
        frame.loc[kept.index[(sizes < 2).to_numpy() | kept['binary'].isna().to_numpy()], 'keep'] = False
    report('Step 6 (singletons)', frames)
    return {split: frame['keep'].to_numpy() for split, frame in frames.items()}


def report(step: str, frames: Dict[str, pd.DataFrame]) -> None:
    print(f'{step}: ' + ', '.join(f'{split} {frame["keep"].sum():,}/{len(frame):,}'
                                  for split, frame in sorted(frames.items())), flush=True)


def main(
    split: List[str] = typer.Option(..., help='NAME=PATH of a raw dataset split (DataProcess output); repeatable'),
    output_dir: str = typer.Option(..., help='receives <split> datasets of the selected rows'),
    naming: str = typer.Option(..., help='Pretrain.split_train_validation naming convention, e.g. dataset1-xm'),
    num_proc: int = typer.Option(available_cpu_count(), min=1, help='worker processes for hashing and saving'),
):
    import datasets

    loaded = {}
    for item in split:
        name, path = item.split('=', 1)
        if name not in {'train', 'validation', 'test'} or name in loaded:
            raise ValueError(f'Invalid or duplicate split: {name}')
        if os.path.exists(os.path.join(output_dir, name)):
            raise ValueError(f'Output already exists: {output_dir}/{name}')
        loaded[name] = datasets.load_from_disk(path)
    frames = {}
    for name, dataset in loaded.items():
        digests = dataset.select_columns(['input_ids']).map(
            lambda batch: {'ir_digest': [ir_digest(ids) for ids in batch['input_ids']]},
            batched=True, remove_columns=['input_ids'], num_proc=num_proc)['ir_digest']
        frames[name] = pd.DataFrame({'file_path': dataset['file_path'], 'ir_digest': digests})
    masks = select_samples(frames, naming)
    for name, dataset in loaded.items():
        dataset.select(np.flatnonzero(masks[name])).save_to_disk(os.path.join(output_dir, name),
                                                                 num_proc=num_proc)


if __name__ == '__main__':
    typer.run(main)
