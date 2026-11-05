"""EIR irnorm step 1: the distance subsets of the irsim (v1) and v2 coreutils pairs restricted to pairs whose two functions
are both non-mips64 and both have an EIR text (S11A/out_b/text: (module, LLVM function name)). Writes
<out>/{v1,v2}/pairs_hash.parquet (pair, kind, i, j, dist=True; pair ids of the original run) + universe.parquet (link),
so compute_dist_norm.py runs unchanged on them; and <out>/pairs_filter.json (kept / dropped counts per kind).
usage: make_pairs_eir.py <out>
"""
import glob, json, os, sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

out = sys.argv[1]
RUNS = {"v1": f"{DATA}/irsim/coreutils/pairs", "v2": f"{DATA}/irsim-v2/coreutils/pairs"}
have = set()
for f in glob.glob(f"{DATA}/S11A/out_b/text/coreutils/*.parquet"):
    m = "coreutils/" + os.path.basename(f)[:-8]
    have.update((m, n) for n in pq.read_table(f, columns=["function_name"]).column(0).to_pylist())
stats = {"eir_functions": len(have)}
for r, d in RUNS.items():
    u = pd.read_parquet(f"{d}/universe.parquet", columns=["fid", "module", "function_name", "arch"])
    ok = [(a != "mips64") and ((m, n) in have) for m, n, a in zip(u.module, u.function_name, u.arch)]
    ok = pd.Series(ok, index=u.fid).sort_index().to_numpy()
    p = pd.read_parquet(f"{d}/pairs_hash.parquet", columns=["pair", "kind", "i", "j", "dist"])
    p = p[p.dist]
    keep = ok[p.i.to_numpy()] & ok[p.j.to_numpy()]
    os.makedirs(f"{out}/{r}", exist_ok=True)
    p[keep].to_parquet(f"{out}/{r}/pairs_hash.parquet", index=False)
    if not os.path.exists(f"{out}/{r}/universe.parquet"):
        os.symlink(f"{d}/universe.parquet", f"{out}/{r}/universe.parquet")
    stats[r] = {k: {"distance_subset": int((p.kind == k).sum()), "kept": int(((p.kind == k) & keep).sum())}
                for k in sorted(p.kind.unique())}
json.dump(stats, open(f"{out}/pairs_filter.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
