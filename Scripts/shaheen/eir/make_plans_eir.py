"""S11-A step 4: CLAP / DEJINA export plans for the coreutils EIR .re: test-pool functions
(pkgA/coreutils_pool_functions_nonmips64.tsv) looked up BY NAME in the EIR IDB's functions.json (rell.ida output).
plans/<k>.json = {shard: {key: {idb_path, dataset, funcs: [[pool row, EIR ea], ...]}}}, one IDA shard per plan file (the
S7 / DEJINA layout). Pool rows that cannot be planned go to plans/missing.tsv (row, bin, function_name, reason):
  revng_failed       the build has no EIR (rev.ng lift failed)
  ida_failed         the EIR .re's rell.ida job is not complete
  name_not_found     no EIR function with that name
  name_ambiguous     more than one EIR function with that name
usage: make_plans_eir.py <rell.ida out (index/, features/)> <plans dir> <revng_failed.txt>
"""
import collections, csv, io, json, sys, tarfile
from pathlib import Path
import subprocess

ida, out, failed = Path(sys.argv[1]), Path(sys.argv[2]), set(Path(sys.argv[3]).read_text().split())
out.mkdir(parents=True, exist_ok=True)
ix = {}
for f in sorted((ida / 'index').glob('*.tsv')):
    for r in csv.DictReader(open(f), delimiter='\t'):
        ix['coreutils/' + r['idb_path'].rsplit('/', 1)[1][:-4]] = dict(r, shard=f.stem)
names = {}                                                     # key -> Counter-like {name: [ea, ...]}
for t in sorted((ida / 'features').glob('*.tar.zst')):
    raw = subprocess.run(['zstd', '-dc', str(t)], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for m in tf.getmembers():
            if m.name.endswith('/outputs/functions.json'):
                d = collections.defaultdict(list)
                for x in json.load(tf.extractfile(m)):
                    d[x['name']].append(int(x['fva'], 16))
                names[m.name.split('/', 1)[0]] = d
plan, miss, c = {}, [], collections.Counter()
for r in csv.DictReader(open(Path(__file__).resolve().parent / 'pkg/pkgA/coreutils_pool_functions_nonmips64.tsv'), delimiter='\t'):
    b = ix.get(r['bin'])
    if r['bin'] in failed:
        why = 'revng_failed'
    elif b is None or b['status'] != 'complete' or b['key'] not in names:
        why = 'ida_failed'
    else:
        eas = names[b['key']].get(r['function_name'], [])
        why = None if len(eas) == 1 else ('name_not_found' if not eas else 'name_ambiguous')
    if why:
        miss.append((r['row'], r['bin'], r['function_name'], why)); c[why] += 1; continue
    c['planned'] += 1
    e = plan.setdefault(b['shard'], {}).setdefault(b['key'], {'idb_path': b['idb_path'], 'dataset': 'coreutils', 'funcs': []})
    e['funcs'].append([int(r['row']), eas[0]])
for k, shard in enumerate(sorted(plan)):
    (out / f'{k:03d}.json').write_text(json.dumps({shard: plan[shard]}))
with open(out / 'missing.tsv', 'w') as f:
    f.write('row\tbin\tfunction_name\treason\n' + ''.join('\t'.join(m) + '\n' for m in miss))
summary = {'plans': len(plan), 'binaries': sum(map(len, plan.values())), **c}
(out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))
