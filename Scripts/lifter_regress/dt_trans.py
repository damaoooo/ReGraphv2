"""dt_trans.py DTDIR : transitions of per-function verdicts between dir 0 and dir 1."""
import sys, json, glob, os, collections
c = collections.Counter(); ex = collections.defaultdict(list)


def verdict(p):
    if p["err"]: return "err"
    if p["disagree"]: return "bad"
    if p["agree"]: return "ok"
    return "untestable"


for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    if f.endswith("bad_examples.json"): continue
    d = json.load(open(f))
    for fn, r in d["results"].items():
        if "per_dir" not in r: continue
        v = tuple(verdict(p) for p in r["per_dir"])
        c[v] += 1
        if len(ex[v]) < 12: ex[v].append((os.path.basename(f)[:-5], fn))
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(k, v)
for k in ex:
    if k[0] == "ok" and k[-1] == "bad":
        print("REGRESSIONS", k, ex[k])
