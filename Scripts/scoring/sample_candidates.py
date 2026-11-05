# CPU-only candidate sampler for the single-compiler subsets (talos_gcc / talos_clang) of the Talos-common Oc test set.
# No torch, no embeddings: every model is scored later on the same frozen lists (score_candidates.py for ReGraphv2).
#
# Protocol cpu_seed42_v1 (NumPy PCG64, independent streams so any block can be regenerated on its own):
#   queries   rng = default_rng([42, subset_id, task_id, 0]); permute the subset pool, keep anchors in that order that
#             have >= 1 valid positive (task constraint, same (proj, lib, function_name) group, other row), one positive
#             drawn uniformly per anchor with the same rng, at most 100,000 queries per task.
#   negatives block k of pool P: rng = default_rng([42, subset_id, task_id, P, k]); draw (n, P-1+64) pool positions
#             uniformly with replacement, drop draws in the anchor's group, keep the first P-1 per query (refill a
#             short row with further 64-draws from the same rng). Candidates per query = 1 positive + P-1 negatives.
# Row indices are rows of the full Oc test pool (keys_Oc_full_test_pool.parquet). Output is staged in
# <out_root>/<subset>.partial and atomically renamed to <out_root>/<subset> after MANIFEST.json is written.
# usage: sample_candidates.py <talos_gcc|talos_clang> <out_root>
import hashlib, json, platform, socket, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
KEYS = HERE / "keys_Oc_full_test_pool.parquet"
TALOS_ROWS = HERE / "full_test_pool_indices.npy"  # callers (e.g. the per-dataset sampler) set KEYS / TALOS_ROWS
PROTOCOL = "cpu_seed42_v1"
SEED, MAXQ, CHUNK, OVERSAMPLE, POOLS = 42, 100000, 8192, 64, (100, 10000)
SUBSET_ID = {"talos_gcc": 1, "talos_clang": 2}
TASK_ID = {"XA": 1, "XO": 2, "XV": 3, "XM": 4}  # also the task order
ARCH = {"arm32": ("arm", 32), "arm64": ("arm", 64), "mips32": ("mips", 32), "mips64": ("mips", 64), "x86": ("x", 32), "x64": ("x", 64)}


def load_subset(subset):
    d = pd.read_parquet(KEYS)
    p = d.bin.str.extract(r"^(?P<proj>[^/]+)/(?P<arch>[a-z]+\d+)-(?P<comp>gcc|clang)-(?P<ver>[\d.]+)-(?P<opt>O[0-3s])_(?P<lib>.+)$")
    assert p.notna().all().all()
    ctx = {"subset": subset, "fam": np.array([ARCH[x][0] for x in p.arch]), "bits": np.array([ARCH[x][1] for x in p.arch]),
           "comp": p.comp.to_numpy(), "ver": p.ver.to_numpy(), "opt": p.opt.to_numpy(),
           "grp": pd.factorize(p.proj + "\0" + p.lib + "\0" + d.function_name)[0].astype(np.int32)}
    keep = np.zeros(len(d), bool); keep[np.load(TALOS_ROWS)] = True
    keep &= ctx["comp"] == subset.split("_")[1]
    ctx["pool"] = np.flatnonzero(keep).astype(np.uint32)
    members = defaultdict(list)
    for i in ctx["pool"]: members[ctx["grp"][i]].append(int(i))
    ctx["members"] = members
    return ctx


def task_ok(ctx, t, i, j):
    f, b, v, o = ctx["fam"], ctx["bits"], ctx["ver"], ctx["opt"]  # compiler is constant within a subset
    return {"XA": lambda: f[i] != f[j] and b[i] != b[j] and v[i] == v[j] and o[i] == o[j],
            "XO": lambda: o[i] != o[j] and f[i] == f[j] and b[i] == b[j] and v[i] == v[j],
            "XV": lambda: v[i] != v[j] and o[i] != o[j] and f[i] == f[j] and b[i] == b[j],
            "XM": lambda: True}[t]()


def sample_queries(ctx, t):
    rng = np.random.default_rng([SEED, SUBSET_ID[ctx["subset"]], TASK_ID[t], 0])
    anchors, pos = [], []
    for i in rng.permutation(ctx["pool"]):
        i = int(i); c = [j for j in ctx["members"][ctx["grp"][i]] if j != i and task_ok(ctx, t, i, j)]
        if c: anchors.append(i); pos.append(c[rng.integers(len(c))])
        if len(anchors) == MAXQ: break
    return np.array(anchors, np.uint32), np.array(pos, np.uint32)


def sample_negatives(ctx, t, P, k, anchors):
    """Negatives for queries anchors[k*CHUNK:(k+1)*CHUNK] at pool size P: uint32 [n, P-1]."""
    rng = np.random.default_rng([SEED, SUBSET_ID[ctx["subset"]], TASK_ID[t], P, k])
    pool, grp = ctx["pool"], ctx["grp"]
    a = anchors[k * CHUNK:(k + 1) * CHUNK]
    rows = pool[rng.integers(0, len(pool), size=(len(a), P - 1 + OVERSAMPLE))]
    ok = grp[rows] != grp[a][:, None]
    out = rows[:, :P - 1].copy()
    for r in np.flatnonzero(~ok[:, :P - 1].all(1)):
        v = rows[r][ok[r]]
        while len(v) < P - 1:
            extra = pool[rng.integers(0, len(pool), size=OVERSAMPLE)]
            v = np.concatenate([v, extra[grp[extra] != grp[a[r]]]])
        out[r] = v[:P - 1]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""): h.update(block)
    return h.hexdigest()


def main():
    subset, root = sys.argv[1], Path(sys.argv[2])
    final = root / subset
    assert not final.exists(), f"{final} already published; published lists are immutable"
    out = root / f"{subset}.partial"
    out.mkdir(parents=True, exist_ok=False)
    ctx = load_subset(subset)
    np.save(out / "pool_rows.npy", ctx["pool"])
    summary = {"protocol": PROTOCOL, "subset": subset, "pool_rows": int(len(ctx["pool"])), "tasks": {}}
    for t in TASK_ID:
        anchors, pos = sample_queries(ctx, t)
        td = out / t; td.mkdir()
        np.save(td / "anchors.npy", anchors); np.save(td / "positives.npy", pos)
        nblk = -(-len(anchors) // CHUNK)
        for P in POOLS:
            for k in range(nblk):
                np.save(td / f"neg_p{P}_{k:03d}.npy", sample_negatives(ctx, t, P, k, anchors))
        summary["tasks"][t] = {"queries": int(len(anchors)), "pools": {str(P): {"neg_files": nblk, "neg_per_query": P - 1} for P in POOLS}}
        print(f"{subset} {t}: {len(anchors)} queries, {nblk} blocks per pool", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    assert "torch" not in sys.modules
    env = {"numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(), "host": socket.gethostname(),
           "platform": platform.platform(), "torch_imported": False,
           "regraphv2_git_head": subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    protocol = {"name": PROTOCOL, "bit_generator": "numpy PCG64 via default_rng(seed_sequence)", "seed": SEED,
                "query_stream": "default_rng([42, subset_id, task_id, 0])", "negative_stream": "default_rng([42, subset_id, task_id, P, block])",
                "subset_id": SUBSET_ID, "task_id_and_order": TASK_ID, "max_queries": MAXQ, "block_queries": CHUNK,
                "pool_sizes": list(POOLS), "oversample": OVERSAMPLE, "negatives_with_replacement": True,
                "candidates_per_query": "1 positive + P-1 negatives (anchor never a candidate)",
                "negative_exclusion": "anchor's (proj, lib, function_name) group"}
    inputs = {k: {"path": str(v), "sha256": sha256(v)} for k, v in {"keys": KEYS, "talos_common_rows": TALOS_ROWS, "script": Path(__file__).resolve()}.items()}
    files = {}
    for f in sorted(out.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(out)); files[rel] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
            if f.suffix == ".npy":
                m = np.load(f, mmap_mode="r"); files[rel].update(dtype=str(m.dtype), shape=list(m.shape))
    (out / "MANIFEST.json").write_text(json.dumps({"subset": subset, "complete": True, "protocol": protocol, "env": env,
                                                    "inputs": inputs, "files": files}, indent=1))
    for f in out.rglob("*"):
        if f.is_file(): f.chmod(0o444)
    out.rename(final)  # atomic publish
    for dd in [final, *[x for x in final.iterdir() if x.is_dir()]]: dd.chmod(0o555)
    print("published", final, flush=True)


if __name__ == "__main__":
    main()
