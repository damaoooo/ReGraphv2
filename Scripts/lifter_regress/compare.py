"""compare.py BASE_DIR NEW_DIR [--detail] : per-build + total lift outcome diff and IR-change stats"""
import sys, json, os, re, collections, glob
base, new = sys.argv[1], sys.argv[2]
detail = "--detail" in sys.argv


def funcs(path):
    out, cur, buf = {}, None, []
    for line in open(path):
        if line.startswith("define "):
            m = re.search(r'@"?([^"(]+)"?\(', line); cur, buf = m.group(1), [line]
        elif cur is not None:
            buf.append(line)
            if line.startswith("}"):
                out[cur] = "".join(buf); cur = None
    return out


def recs(path):
    d = json.load(open(path))
    return d, {(f["ea"]): f for f in d["functions"]}


T = collections.Counter(); TR = collections.Counter(); TRn = collections.Counter()
rows = []
changed_examples = []
newly = []; lost = []
for bj in sorted(glob.glob(os.path.join(base, "*.lift.json"))):
    name = os.path.basename(bj)[:-len(".ll.lift.json")]
    nj = os.path.join(new, os.path.basename(bj))
    if not os.path.exists(nj):
        print("missing", name); continue
    bd, br = recs(bj); nd, nr = recs(nj)
    bf = funcs(bj[:-len(".lift.json")]) if bd["status"] == "ok" else {}
    nf = funcs(nj[:-len(".lift.json")]) if nd["status"] == "ok" else {}
    c = collections.Counter()
    for ea, b in br.items():
        n = nr.get(ea)
        if n is None:
            c["missing_rec"] += 1; continue
        bs, ns = b["status"], n["status"]
        c[f"b_{bs}"] += 1; c[f"n_{ns}"] += 1
        if bs == "rejected": TR[b["reason"]] += 1
        if ns == "rejected": TRn[n["reason"]] += 1
        if bs == "lifted" and ns == "lifted":
            c["both"] += 1
            x, y = bf.get(b.get("llvm_name")), nf.get(n.get("llvm_name"))
            if x != y:
                c["changed"] += 1
                changed_examples.append((name, b["name"]))
            if bool(b.get("approx")) != bool(n.get("approx")): c["approx_flip"] += 1
        elif bs != "lifted" and ns == "lifted":
            c["newly_lifted"] += 1; newly.append((name, b["name"], b.get("reason"), n.get("approx")))
        elif bs == "lifted" and ns != "lifted":
            c["lost"] += 1; lost.append((name, b["name"], n.get("reason"), n.get("detail", "")[:100]))
    T.update(c)
    rows.append((name, c))
print(f"{'build':42s} {'lift_b':>6s} {'lift_n':>6s} {'rej_b':>5s} {'rej_n':>5s} {'both':>5s} {'chg':>5s} {'new':>4s} {'lost':>4s}")
for name, c in rows:
    if detail or c["newly_lifted"] or c["lost"]:
        print(f"{name:42s} {c['b_lifted']:6d} {c['n_lifted']:6d} {c['b_rejected']:5d} {c['n_rejected']:5d} {c['both']:5d} {c['changed']:5d} {c['newly_lifted']:4d} {c['lost']:4d}")
c = T
print(f"{'TOTAL':42s} {c['b_lifted']:6d} {c['n_lifted']:6d} {c['b_rejected']:5d} {c['n_rejected']:5d} {c['both']:5d} {c['changed']:5d} {c['newly_lifted']:4d} {c['lost']:4d}  approx_flip={c['approx_flip']}")
print("rejections base -> new:")
for k in sorted(set(TR) | set(TRn), key=lambda k: -TR.get(k, 0)):
    print(f"  {k:55s} {TR.get(k,0):6d} -> {TRn.get(k,0):6d}")
json.dump({"newly": newly, "lost": lost, "changed": changed_examples}, open(os.path.join(new, "compare_vs_base.json"), "w"))
print("lost examples:", lost[:15])
