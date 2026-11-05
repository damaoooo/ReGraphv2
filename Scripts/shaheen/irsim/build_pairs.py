"""Sample function pairs for the IR-similarity study and attach the hash-level metrics.

Universe: every function (binary, LLVM function name) whose normalized IR exists (purify ok) at every level in
--base-levels (O0, Oc, O3). No selection, no deduplication. Labels are parsed from the binary name with the
reference regex; positive groups are (project, library, function_name).

Sampling, numpy default_rng(42), as in reference_ir_distance.py:
  for kind in XA, XO, XC, XM, XV: walk anchors i in rng.permutation(n); candidates = the other members j of i's
  group that satisfy the kind's constraint; if any, take one uniformly (rng.integers) and record (i, j);
  stop at --n-hash pairs (or when all anchors are used). Then NEG: draw (i, j) = rng.integers(n, size=2) until
  --n-hash pairs with different groups.
The first --n-dist pairs of each kind (in sampling order) form the distance subset.

Outputs in --out:
  universe.parquet   fid, module, function_name, project, lib, arch, fam, bits, comp, ver, opt, group, hashopcodes
  pairs_hash.parquet pair, kind, i, j, dist (in the distance subset), same_hashopcodes, and per level L:
                     L_ident_raw, L_ident_canon, L_ident_ops, L_ntok_i, L_ntok_j, L_nblk_i, L_nblk_j
  counts.json        functions per level (rows, purify failed, in universe), sampling stats
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import os as _os
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
TEXT = _os.environ.get("IRSIM_TEXT", f"{DATA}/irsim/text")
FLOW = _os.environ.get("IRSIM_FLOW", f"{DATA}/Dataset-1-flowchart")
ARCH = {"arm32": ("arm", 32), "arm64": ("arm", 64), "mips32": ("mips", 32), "mips64": ("mips", 64), "x86": ("x", 32), "x64": ("x", 64),
        "ppc32": ("ppc", 32), "ppc64": ("ppc", 64)}
RE = r"^(?P<proj>[^/]+)/(?P<arch>[a-z]+\d+)-(?P<comp>gcc|clang)-(?P<ver>[\d.]+)-(?P<opt>O[0-3s])_(?P<lib>.+)$"
META = ["function_name", "purify_ok", "skip", "n_blocks", "ntok", "h_raw", "h_canon", "h_ops"]


def read_meta(args):
    level, rel = args
    t = pq.read_table(f"{TEXT}/{level}/{rel}.parquet", columns=META).to_pandas()
    t.insert(0, "module", rel)
    return t


def read_flow(rel):
    t = pq.read_table(f"{FLOW}/{rel}.parquet", columns=["llvm_name", "hashopcodes"]).to_pandas()
    t = t.dropna(subset=["llvm_name"]).drop_duplicates("llvm_name")
    t.insert(0, "module", rel)
    return t.rename(columns={"llvm_name": "function_name"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-levels", default="O0,Oc,O3")
    ap.add_argument("--extra-levels", default="O1,O2,Os,Oz")
    ap.add_argument("--n-hash", type=int, default=1_000_000)
    ap.add_argument("--n-dist", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--modules", default="**", help="glob under each level (testing: e.g. train/curl/*)")
    ap.add_argument("--no-split-level", action="store_true", help="modules are <proj>/<binary> (coreutils), not <split>/<proj>/<binary>")
    ap.add_argument("--kinds", default="XA,XO,XC,XM,XV")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    base = args.base_levels.split(",")
    extra = [L for L in args.extra_levels.split(",") if L and os.path.isdir(f"{TEXT}/{L}")]
    levels = base + extra
    rels = sorted(p[len(f"{TEXT}/{base[0]}/"):-len(".parquet")]
                  for p in glob.glob(f"{TEXT}/{base[0]}/{args.modules}.parquet" if args.modules != "**"
                                  else f"{TEXT}/{base[0]}/**/*.parquet", recursive=True))
    counts = {"modules": len(rels), "levels": {}}

    meta = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for L in levels:
            missing = [r for r in rels if not os.path.exists(f"{TEXT}/{L}/{r}.parquet")]
            if missing and L in base:
                raise SystemExit(f"{L}: {len(missing)} modules without text, e.g. {missing[0]}")
            df = pd.concat(ex.map(read_meta, [(L, r) for r in rels if r not in missing], chunksize=16), ignore_index=True)
            df["module"] = df["module"].astype("category")
            meta[L] = df.set_index(["module", "function_name"])
            counts["levels"][L] = {"functions": len(df), "purify_failed": int((~df.purify_ok).sum()),
                                   "skip_flag": int(df.skip.sum()), "modules_without_text": missing}
            print(f"{L}: {len(df):,} functions, {int((~df.purify_ok).sum()):,} purify failed  {time.time()-t0:.0f}s", flush=True)
        flow = pd.concat(ex.map(read_flow, rels, chunksize=16), ignore_index=True).set_index(["module", "function_name"])

    ok = None
    for L in base:
        s = set(meta[L].index[meta[L].purify_ok.to_numpy()])
        ok = s if ok is None else ok & s
    allkeys = set().union(*(set(meta[L].index) for L in base))
    uni = pd.DataFrame(sorted(ok), columns=["module", "function_name"])
    for L in levels:
        present = meta[L].index.isin(pd.MultiIndex.from_frame(uni))
        counts["levels"][L]["in_universe"] = int(present.sum())
    counts["universe"] = len(uni)
    counts["any_base_level"] = len(allkeys)
    counts["dropped_not_at_all_base_levels_or_purify_failed"] = len(allkeys) - len(uni)
    print(f"universe {len(uni):,} functions (of {len(allkeys):,} seen at any base level)", flush=True)

    p = (uni.module if args.no_split_level else uni.module.str.split("/", n=1).str[1]).str.extract(RE)  # <split>/<proj>/<binary>
    if p.isna().any().any():
        raise SystemExit("unparsed binary names: " + str(uni.module[p.isna().any(axis=1)].unique()[:3]))
    uni = pd.concat([uni, p.rename(columns={"proj": "project"})], axis=1)
    uni["fam"] = uni.arch.map(lambda a: ARCH[a][0])
    uni["bits"] = uni.arch.map(lambda a: ARCH[a][1])
    uni["group"] = pd.factorize(uni.project + "\0" + uni.lib + "\0" + uni.function_name)[0]
    uni["hashopcodes"] = flow.reindex(pd.MultiIndex.from_frame(uni[["module", "function_name"]])).hashopcodes.to_numpy()
    uni.insert(0, "fid", np.arange(len(uni)))
    counts["hashopcodes_missing"] = int(uni.hashopcodes.isna().sum())

    n = len(uni)
    grp = uni.group.to_numpy()
    fam = pd.factorize(uni.fam)[0]; bits = uni.bits.to_numpy(); comp = pd.factorize(uni.comp)[0]
    ver = pd.factorize(uni.comp + uni.ver)[0]; opt = pd.factorize(uni.opt)[0]
    order = np.argsort(grp, kind="stable")
    starts = np.searchsorted(grp[order], np.arange(grp.max() + 2))
    members = lambda g: order[starts[g]:starts[g + 1]]
    OK = {
        "XA": lambda i, m: (fam[m] != fam[i]) & (bits[m] != bits[i]) & (comp[m] == comp[i]) & (ver[m] == ver[i]) & (opt[m] == opt[i]),
        "XO": lambda i, m: (opt[m] != opt[i]) & (fam[m] == fam[i]) & (bits[m] == bits[i]) & (comp[m] == comp[i]) & (ver[m] == ver[i]),
        "XC": lambda i, m: (comp[m] != comp[i]) & (ver[m] != ver[i]) & (opt[m] != opt[i]) & (fam[m] == fam[i]) & (bits[m] == bits[i]),
        "XM": lambda i, m: m != i,
        "XV": lambda i, m: (ver[m] != ver[i]) & (opt[m] != opt[i]) & (comp[m] == comp[i]) & (fam[m] == fam[i]) & (bits[m] == bits[i]),
    }
    OK = {k: v for k, v in OK.items() if k in args.kinds.split(",")}
    rng = np.random.default_rng(42)
    rows, counts["sampling"] = [], {}
    for kind, ok_fn in OK.items():
        out, tried = [], 0
        for i in rng.permutation(n):
            tried += 1
            m = members(grp[i])
            c = m[ok_fn(i, m) & (m != i)]
            if len(c):
                out.append((i, c[rng.integers(len(c))]))
                if len(out) == args.n_hash:
                    break
        counts["sampling"][kind] = {"pairs": len(out), "anchors_tried": tried}
        rows += [(kind, i, j, k < args.n_dist) for k, (i, j) in enumerate(out)]
        print(f"{kind}: {len(out):,} pairs from {tried:,} anchors  {time.time()-t0:.0f}s", flush=True)
    neg, draws = [], 0
    while len(neg) < args.n_hash:
        i, j = rng.integers(n, size=2)
        draws += 1
        if grp[i] != grp[j]:
            neg.append((int(i), int(j)))
    counts["sampling"]["NEG"] = {"pairs": len(neg), "draws": draws}
    rows += [("NEG", i, j, k < args.n_dist) for k, (i, j) in enumerate(neg)]

    pairs = pd.DataFrame(rows, columns=["kind", "i", "j", "dist"])
    pairs.insert(0, "pair", np.arange(len(pairs)))
    hi, hj = uni.hashopcodes.to_numpy()[pairs.i], uni.hashopcodes.to_numpy()[pairs.j]
    pairs["same_hashopcodes"] = pd.array(np.where(pd.isna(hi) | pd.isna(hj), None, hi == hj), dtype="boolean")
    key = pd.MultiIndex.from_frame(uni[["module", "function_name"]])
    for L in levels:
        m = meta[L].reindex(key)
        for col, src in [("raw", "h_raw"), ("canon", "h_canon"), ("ops", "h_ops")]:
            h = m[src].to_numpy()
            a, b = h[pairs.i], h[pairs.j]
            valid = pd.notna(a) & pd.notna(b)
            pairs[f"{L}_ident_{col}"] = pd.array(np.where(valid, a == b, None), dtype="boolean")
        for col, src in [("ntok", "ntok"), ("nblk", "n_blocks")]:
            v = m[src].to_numpy()
            pairs[f"{L}_{col}_i"] = pd.array(v[pairs.i], dtype="Int64")
            pairs[f"{L}_{col}_j"] = pd.array(v[pairs.j], dtype="Int64")
    uni.to_parquet(f"{args.out}/universe.parquet")
    pairs.to_parquet(f"{args.out}/pairs_hash.parquet")
    counts["seconds"] = round(time.time() - t0)
    json.dump(counts, open(f"{args.out}/counts.json", "w"), indent=1)
    print(json.dumps(counts["sampling"]), f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
