"""ircmp.py BASE_DIR NEW_DIR [-j N] : for functions lifted by both, compare raw IR and Oc-normalized IR.
Writes NEW_DIR/ircmp.json with per-function categories."""
import sys, json, os, re, collections, glob, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
PASSES = "sroa,mem2reg,simplifycfg,early-cse,sccp,correlated-propagation,jump-threading,simplifycfg,reassociate,gvn,dce,bdce,adce,simplifycfg"
base, new = sys.argv[1], sys.argv[2]
J = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 16


STRONG = PASSES + ",instcombine<no-verify-fixpoint>,dse,sroa,instcombine<no-verify-fixpoint>,simplifycfg,adce,gvn,instcombine<no-verify-fixpoint>,simplifycfg"


def oc(path, strong=False):
    out = path + (".st.ll" if strong else ".oc.ll")
    if not os.path.exists(out):
        p = subprocess.run(["opt", "-S", "-passes=" + (STRONG if strong else PASSES), path, "-o", out], capture_output=True, text=True)
        if p.returncode:
            print("opt failed", path, p.stderr[:300]); return None
    return out


def funcs(path):
    out, cur, buf = {}, None, []
    if path is None: return out
    for line in open(path):
        if line.startswith("define "):
            m = re.search(r'@"?([^"(]+)"?\(', line); cur, buf = m.group(1), [line]
        elif cur is not None:
            buf.append(line)
            if line.startswith("}"):
                out[cur] = "".join(buf); cur = None
    return out


TOK = re.compile(r'%(?:"[^"]*"|[-\w.$]+)')


def norm(body, name):
    body = re.sub(r", align \d+", "", body)
    body = body.replace('@"%s"' % name, "@SELF").replace("@%s(" % name, "@SELF(")
    m = {}
    def sub(t):
        k = t.group(0)
        if k not in m: m[k] = "%%v%d" % len(m)
        return m[k]
    body = TOK.sub(sub, body)
    # block labels
    body = re.sub(r'^("[^"]*"|[-\w.$]+):', lambda t: "L:", body, flags=re.M)
    return body


def recs(path):
    return {f["ea"]: f for f in json.load(open(path))["functions"]}


def one(bj):
    name = os.path.basename(bj)[:-len(".ll.lift.json")]
    nj = os.path.join(new, os.path.basename(bj))
    br, nr = recs(bj), recs(nj)
    bll, nll = bj[:-len(".lift.json")], nj[:-len(".lift.json")]
    braw, nraw = funcs(bll), funcs(nll)
    boc, noc = funcs(oc(bll)), funcs(oc(nll))
    res = []
    bst = nst = None
    for ea, b in br.items():
        n = nr.get(ea)
        if not n or b["status"] != "lifted" or n["status"] != "lifted": continue
        fn = b["llvm_name"]
        if fn not in braw or fn not in nraw: continue
        raw_same = braw[fn] == nraw.get(fn)
        x, y = boc.get(fn), noc.get(fn)
        oc_same = x is not None and y is not None and norm(x, fn) == norm(y, fn)
        if raw_same:
            cat = "same"
        elif oc_same:
            cat = "raw_only"
        else:
            if bst is None:
                bst, nst = funcs(oc(bll, True)), funcs(oc(nll, True))
            x, y = bst.get(fn), nst.get(fn)
            cat = "strong_same" if x is not None and y is not None and norm(x, fn) == norm(y, fn) else "oc_changed"
        res.append((name, fn, cat))
    return res


allres = []
with ThreadPoolExecutor(J) as ex:
    for r in ex.map(one, sorted(glob.glob(os.path.join(base, "*.lift.json")))):
        allres.extend(r)
c = collections.Counter(cat for _, _, cat in allres)
print("functions lifted by both:", len(allres), dict(c))
byarch = collections.defaultdict(collections.Counter)
for b, fn, cat in allres:
    byarch[b.split("-")[0]][cat] += 1
for a, cc in sorted(byarch.items()):
    print(f"  {a:8s} {dict(cc)}")
json.dump(allres, open(os.path.join(new, "ircmp.json"), "w"))
