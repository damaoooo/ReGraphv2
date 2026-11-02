"""dt_all.py OUTDIR DIR1 [DIR2 ...] [-j N] [--max-funcs N] [--only-changed IRCMP_JSON] : run difftest --auto over the sample."""
import sys, os, subprocess, json, collections
from concurrent.futures import ThreadPoolExecutor
args = sys.argv[1:]
J = 16; maxf = "300"; only = None
if "-j" in args: i = args.index("-j"); J = int(args[i + 1]); del args[i:i + 2]
if "--max-funcs" in args: i = args.index("--max-funcs"); maxf = args[i + 1]; del args[i:i + 2]
outdir, dirs = args[0], args[1:]
os.makedirs(outdir, exist_ok=True)
T = os.environ.get("LR_WORK", os.getcwd())
PY = os.environ.get("LR_PY", sys.executable)  # needs unicorn + pyelftools
bins = [l.strip() for l in open(os.path.join(T, "sample.txt")) if l.strip()]


def one(b):
    name = os.path.basename(b)
    out = os.path.join(outdir, name + ".json")
    if not os.path.exists(out):
        p = subprocess.run([PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), "difftest.py"), os.path.join(T, b)] + [os.path.join(T, d) for d in dirs]
                           + ["--auto", "--max-funcs", maxf, "--out", out], capture_output=True, text=True, timeout=7200)
        if p.returncode:
            return name, "FAIL " + p.stderr[-400:]
    return name, "ok"


with ThreadPoolExecutor(J) as ex:
    for name, st in ex.map(one, bins):
        if st != "ok": print(name, st)
tot = [collections.Counter() for _ in dirs]
byarch = collections.defaultdict(lambda: [collections.Counter() for _ in dirs])
bad_examples = [[] for _ in dirs]
for b in bins:
    name = os.path.basename(b)
    f = os.path.join(outdir, name + ".json")
    if not os.path.exists(f): continue
    d = json.load(open(f))
    for fn, r in d["results"].items():
        if "per_dir" not in r: continue
        for k, p in enumerate(r["per_dir"]):
            if p["err"]: c = "compile_err"
            elif p["disagree"]: c = "disagree"
            elif p["agree"]: c = "agree"
            else: c = "untestable"
            tot[k][c] += 1; byarch[name.split("-")[0]][k][c] += 1
            if c == "disagree" and len(bad_examples[k]) < 400: bad_examples[k].append((name, fn, p.get("example")))
for k, d in enumerate(dirs):
    print(d, dict(tot[k]))
    for a, cs in sorted(byarch.items()):
        print("   ", a, dict(cs[k]))
json.dump(bad_examples, open(os.path.join(outdir, "bad_examples.json"), "w"), indent=1)
