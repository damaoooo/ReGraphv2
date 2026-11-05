# Score one model on a frozen, published candidate subset (independent of sampling; consumes no RNG).
# Embeddings: .npy [rows of the full Oc test pool, dim], L2-normalized; score = dot product in fp32.
# rank = 1 + #(negatives scoring strictly above the positive), R@1 = mean(rank == 1), MRR = mean(1/rank) over all P.
# Output staged in <out_root>/<label>/<subset>.partial, atomically renamed after MANIFEST.json.
# usage: score_candidates.py <list_root>/<subset> <embeddings.npy> <out_root> <label> [--device cuda|cpu]
import hashlib, json, platform, socket, sys
from pathlib import Path
import numpy as np, torch

lists, emb, root, label = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
SUB = 64
lman = json.loads((lists / "MANIFEST.json").read_text()); assert lman["complete"]
summary = json.loads((lists / "summary.json").read_text())
final = root / label / lists.name
assert not final.exists(), f"{final} already published"
out = root / label / f"{lists.name}.partial"; out.mkdir(parents=True, exist_ok=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""): h.update(block)
    return h.hexdigest()


E = torch.from_numpy(np.load(emb)).to(dev, torch.float32)
res = {"label": label, "subset": lists.name, "protocol": lman["protocol"]["name"], "tasks": {}}
print(f"{label} on {lists.parent.name}/{lists.name}: task | queries | R@1/MRR pool 100 | R@1/MRR pool 10000")
for t, info in summary["tasks"].items():
    a = torch.from_numpy(np.load(lists / t / "anchors.npy").astype(np.int64)).to(dev)
    q = torch.from_numpy(np.load(lists / t / "positives.npy").astype(np.int64)).to(dev)
    (out / t).mkdir(); res["tasks"][t] = {"queries": len(a)}; line = []
    for P in (100, 10000):
        ranks, s = [], 0
        for k in range(info["pools"][str(P)]["neg_files"]):
            n = torch.from_numpy(np.load(lists / t / f"neg_p{P}_{k:03d}.npy").astype(np.int64))
            for u in range(0, len(n), SUB):
                ea = E[a[s + u:s + u + SUB]]; sp = (ea * E[q[s + u:s + u + SUB]]).sum(1)
                sn = torch.einsum("bd,bpd->bp", ea, E[n[u:u + SUB].to(dev)])
                ranks.append(((sn > sp[:, None]).sum(1) + 1).cpu())
            s += len(n)
        r = torch.cat(ranks).numpy().astype(np.uint32); assert len(r) == len(a)
        np.save(out / t / f"rank_p{P}.npy", r)
        r1, mrr = float((r == 1).mean() * 100), float((1 / r.astype(np.float64)).mean() * 100)
        res["tasks"][t][f"pool{P}"] = {"R1": round(r1, 4), "MRR": round(mrr, 4)}; line.append(f"{r1:.1f}/{mrr:.1f}")
    print(f"{t} | {len(a)} | {' | '.join(line)}", flush=True)
(out / "results.json").write_text(json.dumps(res, indent=1))
prop = torch.cuda.get_device_properties(0) if dev == "cuda" else None
files = {str(f.relative_to(out)): {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in sorted(out.rglob("*")) if f.is_file()}
(out / "MANIFEST.json").write_text(json.dumps({
    "complete": True, "label": label, "lists": str(lists), "lists_manifest_sha256": sha256(lists / "MANIFEST.json"),
    "embeddings": {"path": str(emb), "sha256": sha256(emb)}, "score": "fp32 dot product of L2-normalized embeddings, sub-batch 64",
    "env": {"torch": torch.__version__, "device": prop.name if prop else dev, "host": socket.gethostname(), "python": platform.python_version()},
    "script_sha256": sha256(Path(__file__).resolve()), "files": files}, indent=1))
for f in out.rglob("*"):
    if f.is_file(): f.chmod(0o444)
out.rename(final)
for dd in [final, *[x for x in final.iterdir() if x.is_dir()]]: dd.chmod(0o555)
print("published", final, flush=True)
