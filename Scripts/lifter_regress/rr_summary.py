"""rr_summary.py TAG builds... : aggregate rr classification over builds, .text functions only."""
import json, sys, collections, os
T = os.environ.get("LR_WORK", os.getcwd())
tag, builds = sys.argv[1], sys.argv[2:]
SECOND_RET = {"rdx", "edx", "dl", "$v1", "x1", "w1", "r1", "r4", "$v1"}
tot = collections.Counter(); per = {}
for b in builds:
    seg = {f["ea"]: f["segment"] for f in json.load(open(f"{T}/h_v3/{b}.ll.lift.json"))["functions"]}
    r = json.load(open(f"{T}/dt/rr{tag}_{b}.json"))
    c = collections.Counter()
    for x in r:
        if seg.get(x["ea"]) != ".text": continue
        kinds = set()
        for v in x.get("dirty", {}).values():
            for rr in v["reasons"]:
                if rr[0] == "call":
                    k = "call_" + rr[2].split("@")[0]
                    if rr[2].startswith("local") and v["reg"] in SECOND_RET: k = "call_local_2nd_retreg"
                    kinds.add(k)
                else:
                    kinds.add(rr[0])
        if not kinds: cat = "none(other dataflow)"
        elif kinds <= {"call_local", "call_local_LIB"}: cat = "callee-saved-by-IPA? (local callee only)"
        elif kinds <= {"call_local", "call_local_2nd_retreg", "call_local_LIB"}: cat = "local callee, 2nd return register"
        elif kinds <= {"partial_self"}: cat = "partial self-write only"
        elif kinds <= {"call_thunk", "call_extern"}: cat = "import callee only"
        elif kinds <= {"other_lvar"}: cat = "other lvar wrote register"
        elif "call_indirect" in kinds: cat = "involves indirect call"
        else: cat = "mixed:" + ",".join(sorted(kinds))
        c[cat] += 1
    per[b] = c; tot.update(c)
for k, v in tot.most_common():
    print(f"{v:5d}  {k}")
print()
for b, c in per.items():
    print(b, dict(c.most_common(6)))
