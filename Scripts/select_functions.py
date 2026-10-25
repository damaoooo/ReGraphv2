#!/usr/bin/env python3
"""Dataset-wide function selection of Marcelli et al. (USENIX Security '22), Dataset-1 Steps 1-2.

Inputs, per binary <rel> (e.g. train/curl/x64-gcc-9-O2_curl):
  <flowchart_root>/<rel>.parquet   ea, bb_num, hashopcodes of every IDA function (ida_flowchart.py)
  <lift_root>/<rel>.ll.lift.json   the Task 1 lift state (IDA address -> LLVM name, status, segment)
Step 1 keeps the functions with at least --min-bb basic blocks; Step 2 keeps, for each hashopcodes value,
only its first function over the whole dataset (binaries in path order, functions in address order).
Only sample candidates take part: lifted functions outside split_functions.EXCLUDED_SEGMENTS, so a
duplicate is never kept in place of a function that could not become a sample anyway.
Writes <out_root>/<rel>.ll.select.json ({"functions": [llvm names]}, the --select of split_functions)
and <out_root>/selection_summary.json.

usage: select_functions.py FLOWCHART_ROOT LIFT_ROOT OUT_ROOT [--min-bb 5] [--workers N]
"""
import argparse
import json
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Utils.utils import available_cpu_count

from split_functions import EXCLUDED_SEGMENTS


def binary_candidates(flowchart_path: Path, lift_state_path: Path, min_bb: int) -> tuple[int, int, list]:
    """(IDA functions, sample candidates, [(ea, llvm_name, hashopcodes)] of candidates with >= min_bb blocks)."""
    import pyarrow.parquet as pq

    lifted = {}
    state = json.loads(lift_state_path.read_text())
    if state.get('status') != 'ok':
        raise ValueError(f'Incomplete lift state: {lift_state_path}')
    for record in state['functions']:
        if record.get('status') == 'lifted' and not record.get('segment'):
            raise ValueError(f'Missing segment: {lift_state_path}')
        if record.get('status') == 'lifted' and record.get('segment') not in EXCLUDED_SEGMENTS:
            lifted[int(record['ea'], 16)] = record['llvm_name']
    if flowchart_path.suffix == '.json':
        functions = json.loads(flowchart_path.read_text())['functions']
        rows = sorted((r['fva'], r['bb_num'], r['hashopcodes']) for r in functions)
    else:
        table = pq.read_table(flowchart_path, columns=['ea', 'bb_num', 'hashopcodes']).to_pydict()
        rows = sorted(zip(table['ea'], table['bb_num'], table['hashopcodes']))
    candidates = [(ea, lifted[ea], bb_num, digest) for ea, bb_num, digest in rows if ea in lifted]
    return len(rows), len(candidates), [(ea, name, digest) for ea, name, bb_num, digest in candidates
                                        if bb_num >= min_bb]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('flowchart_root', type=Path)
    parser.add_argument('lift_root', type=Path)
    parser.add_argument('out_root', type=Path)
    parser.add_argument('--min-bb', type=int, default=5, help='Step 1 threshold (default 5, as in the paper)')
    parser.add_argument('--workers', type=int, default=available_cpu_count())
    args = parser.parse_args()

    statistics = {}
    for pattern, suffix in (('*.parquet', '.parquet'), ('*.flowchart.json', '.flowchart.json')):
        for path in args.flowchart_root.rglob(pattern):
            rel = str(path.relative_to(args.flowchart_root))[:-len(suffix)]
            if rel in statistics:
                parser.error(f'Duplicate statistics for {rel}')
            statistics[rel] = path
    rels = sorted(statistics)
    if not rels:
        parser.error('No .parquet or .flowchart.json statistics found')
    missing = [rel for rel in rels if not (args.lift_root / (rel + '.ll.lift.json')).exists()]
    if missing:
        sys.exit(f'{len(missing)} binaries have no lift state, e.g. {missing[0]}')
    if args.workers < 1 or args.min_bb < 1:
        parser.error('workers and min-bb must be positive')
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(binary_candidates,
                                [statistics[rel] for rel in rels],
                                [args.lift_root / (rel + '.ll.lift.json') for rel in rels],
                                [args.min_bb] * len(rels), chunksize=16))

    seen = set()
    summary = {'binaries': len(rels), 'ida_functions': 0, 'candidates': 0, 'step1': 0, 'step2': 0}
    for rel, (ida_functions, candidates, step1) in zip(rels, results):
        selected = []
        for _, name, digest in step1:
            if digest not in seen:
                seen.add(digest)
                selected.append(name)
        path = args.out_root / (rel + '.ll.select.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'min_bb': args.min_bb, 'functions': selected}))
        summary['ida_functions'] += ida_functions
        summary['candidates'] += candidates
        summary['step1'] += len(step1)
        summary['step2'] += len(selected)
    (args.out_root / 'selection_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
