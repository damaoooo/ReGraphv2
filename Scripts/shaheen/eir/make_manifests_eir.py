"""S11-A step 3: rell.ida manifests for the coreutils EIR .re, the S2 make_manifests.py rules with variant "EIR".

Input layout: <root>/coreutils/<build>_<prog>.re (no ppc). idb_path = IDBs/coreutils/<build>_<prog>.i64, the same
identity as the S2 original / Oc groups. Shards are balanced by size (largest first into the lightest shard); files
over --big-mb go to "big" shards (fewer workers, more memory per worker).
Output: <out>/coreutils_EIR_{small,big}_<k>.json and <out>/coreutils_EIR_index.tsv.
usage: make_manifests_eir.py --root DIR --out DIR [--small-shards N] [--big-shards N] [--big-mb MB]
"""
import argparse, hashlib, heapq, json
from pathlib import Path


def shard(rows, n):
    heap = [(0, k, []) for k in range(n)]
    for r in sorted(rows, key=lambda r: -r['size']):
        load, k, lst = heapq.heappop(heap)
        lst.append(r)
        heapq.heappush(heap, (load + r['size'], k, lst))
    return [lst for _, _, lst in sorted(heap, key=lambda x: x[1]) if lst]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--small-shards', type=int, default=64); p.add_argument('--big-shards', type=int, default=32)
    p.add_argument('--big-mb', type=float, default=8)
    a = p.parse_args()
    rows = []
    for f in sorted((a.root / 'coreutils').glob('*.re')):
        name = f.name[:-3]
        if name.split('-')[0] in ('ppc32', 'ppc64'):
            continue
        rows.append({'path': str(f.resolve()), 'idb_path': f'IDBs/coreutils/{name}.i64', 'size': f.stat().st_size})
    a.out.mkdir(parents=True, exist_ok=True)
    big = [r for r in rows if r['size'] > a.big_mb * 2 ** 20]
    small = [r for r in rows if r['size'] <= a.big_mb * 2 ** 20]
    index = ['shard\tidb_path\tpath\tbytes\tsha256']
    for kind, part, n in (('small', small, a.small_shards), ('big', big, a.big_shards)):
        for k, lst in enumerate(shard(part, n)):
            name = f'coreutils_EIR_{kind}_{k:03d}'
            (a.out / f'{name}.json').write_text(json.dumps([{'path': r['path'], 'idb_path': r['idb_path'], 'variant': 'EIR'}
                                                            for r in lst], indent=1))
            for r in lst:
                index.append(f"{name}\t{r['idb_path']}\t{r['path']}\t{r['size']}\t{hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()}")
    (a.out / 'coreutils_EIR_index.tsv').write_text('\n'.join(index) + '\n')
    print(f'coreutils EIR: {len(rows)} inputs ({len(small)} small, {len(big)} big) -> {a.out}')


if __name__ == '__main__':
    main()
