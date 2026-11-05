"""Distances for the distance subset of the pairs, at one level, on untruncated whitespace tokens.

Per pair (i, j) and form (raw = normalized IR as stored, canon = local names renumbered, irsim_text.canon):
  edit   = rapidfuzz normalized Levenshtein distance on the token lists
  jac    = 1 - multiset Jaccard
  lenr   = |log(len_i / len_j)|
and the same three on the opcode sequence (tokens that are LLVM opcodes; renaming does not change it, so it is
computed once). Writes <out>/dist_<L>.parquet: pair, raw_edit, raw_jac, raw_lenr, canon_edit, canon_jac,
canon_lenr, ops_edit, ops_jac, ops_lenr, ntok_i, ntok_j, secs. Pairs missing a text at this level get nulls.
"""
import argparse
import multiprocessing
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from irsim_text import OPS, canon  # noqa: E402
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

TEXT = os.environ.get("IRSIM_TEXT", f"{DATA}/irsim/text")
RAW, CAN = {}, {}


def read_texts(args):
    level, module, names_fids = args
    path = f"{TEXT}/{level}/{module}.parquet"
    if not os.path.exists(path):  # no text at this level (e.g. the reopt crashed): its pairs get nulls
        return []
    t = pq.read_table(path, columns=["function_name", "text"]).to_pydict()
    want = dict(names_fids)
    return [(want[n], x) for n, x in zip(t["function_name"], t["text"]) if n in want and x is not None]


def canon_one(item):
    fid, text = item
    return fid, canon(text)


def three(a, b):
    from rapidfuzz.distance import Levenshtein

    ca, cb = Counter(a), Counter(b)
    inter, union = sum((ca & cb).values()), sum((ca | cb).values())
    return (Levenshtein.normalized_distance(a, b), 1 - inter / max(union, 1),
            abs(np.log(max(len(a), 1) / max(len(b), 1))))


def work(chunk):
    out = []
    for pair, i, j in chunk:
        if i not in RAW or j not in RAW:
            out.append((pair,) + (None,) * 9 + (None, None, 0.0))
            continue
        t0 = time.time()
        ra, rb = RAW[i].split(), RAW[j].split()
        r = three(ra, rb)
        c = three(CAN[i].split(), CAN[j].split())
        o = three([w for w in ra if w in OPS], [w for w in rb if w in OPS])
        out.append((pair,) + r + c + o + (len(ra), len(rb), time.time() - t0))
    return out


def main():
    global RAW, CAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    t0 = time.time()
    L = args.level
    uni = pd.read_parquet(f"{args.pairs_dir}/universe.parquet", columns=["fid", "module", "function_name"])
    pairs = pd.read_parquet(f"{args.pairs_dir}/pairs_hash.parquet", columns=["pair", "i", "j", "dist"])
    pairs = pairs[pairs.dist].drop(columns="dist")
    need = np.unique(np.concatenate([pairs.i.to_numpy(), pairs.j.to_numpy()]))
    sub = uni.iloc[need]
    jobs = [(L, m, list(zip(g.function_name, g.fid))) for m, g in sub.groupby("module", observed=True)]
    print(f"{L}: {len(pairs):,} pairs, {len(need):,} functions in {len(jobs)} modules", flush=True)

    with multiprocessing.get_context("fork").Pool(min(args.workers, 96)) as pool:
        for part in pool.imap_unordered(read_texts, jobs, chunksize=4):
            RAW.update(part)
    print(f"{L}: read {len(RAW):,} texts ({sum(map(len, RAW.values())) / 1e9:.1f} GB)  {time.time()-t0:.0f}s", flush=True)
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        CAN = dict(pool.imap_unordered(canon_one, RAW.items(), chunksize=256))
    print(f"{L}: canonicalized  {time.time()-t0:.0f}s", flush=True)

    # largest first so the long z3 functions do not form the tail
    size = np.array([len(RAW.get(i, "")) + len(RAW.get(j, "")) for i, j in zip(pairs.i, pairs.j)])
    order = np.argsort(-size, kind="stable")
    triples = list(zip(pairs.pair.to_numpy()[order], pairs.i.to_numpy()[order], pairs.j.to_numpy()[order]))
    chunks = [triples[k:k + 32] for k in range(0, len(triples), 32)]
    rows = []
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for k, part in enumerate(pool.imap_unordered(work, chunks)):
            rows += part
            if k % 2000 == 0:
                print(f"  {len(rows):,}/{len(triples):,} pairs  {time.time()-t0:.0f}s", flush=True)
    cols = ["pair", "raw_edit", "raw_jac", "raw_lenr", "canon_edit", "canon_jac", "canon_lenr",
            "ops_edit", "ops_jac", "ops_lenr", "ntok_i", "ntok_j", "secs"]
    df = pd.DataFrame(rows, columns=cols).sort_values("pair")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f"{args.pairs_dir}/dist_{L}.parquet")
    print(f"{L}: done {time.time()-t0:.0f}s; slowest pair {df.secs.max():.1f}s; "
          f"missing {df.raw_edit.isna().sum()}", flush=True)


if __name__ == "__main__":
    main()
