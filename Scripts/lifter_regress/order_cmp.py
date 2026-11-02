"""order_cmp.py DIR_A DIR_B BUILD...: compare function status and normalized Oc IR between two lifts of the same builds."""
import sys, os, json, subprocess, collections
sys.path.insert(0, os.path.dirname(__file__))
from ircmp_lib import funcs, norm
PASSES = "sroa,mem2reg,simplifycfg,early-cse,sccp,correlated-propagation,jump-threading,simplifycfg,reassociate,gvn,dce,bdce,adce,simplifycfg"
a, b, builds = sys.argv[1], sys.argv[2], sys.argv[3:]


def oc(p):
    o = p + ".oc.ll"
    if not os.path.exists(o):
        subprocess.run(["opt", "-S", "-passes=" + PASSES, p, "-o", o], check=True, capture_output=True)
    return o


tot = collections.Counter()
for bd in builds:
    ra = {r["ea"]: r for r in json.load(open(f"{a}/{bd}.ll.lift.json"))["functions"]}
    rb = {r["ea"]: r for r in json.load(open(f"{b}/{bd}.ll.lift.json"))["functions"]}
    fa, fb = funcs(oc(f"{a}/{bd}.ll")), funcs(oc(f"{b}/{bd}.ll"))
    c = collections.Counter(); ex = []
    for ea, x in ra.items():
        y = rb.get(ea)
        if y is None:
            c["missing"] += 1; continue
        if (x["status"], x.get("reason"), tuple(x.get("approx", ()))) != (y["status"], y.get("reason"), tuple(y.get("approx", ()))):
            c["status_diff"] += 1; ex.append(("status", x["name"], x["status"], x.get("reason"), y["status"], y.get("reason")))
            continue
        if x["status"] == "lifted":
            n = x["llvm_name"]
            if norm(fa.get(n, ""), n) != norm(fb.get(n, ""), n):
                c["ir_diff"] += 1; ex.append(("ir", x["name"]))
            else:
                c["same"] += 1
        else:
            c["same_status"] += 1
    tot.update(c)
    print(bd, dict(c), ex[:6])
print("TOTAL", dict(tot))
