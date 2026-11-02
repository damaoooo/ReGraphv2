"""stubs.py DIR [DIR...]: per dir, count lifted functions with >=5 lifted blocks whose Oc IR is a constant-return
stub, and functions (>=5 blocks) that share identical normalized Oc IR with another function of the same build."""
import sys, os, glob, re, collections, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from ircmp_lib import funcs, norm

for d in sys.argv[1:]:
    stub = collisions = total = 0
    ex = []
    for ll in sorted(glob.glob(os.path.join(d, "*.ll"))):
        if ll.endswith(".oc.ll") or ll.endswith(".st.ll"): continue
        oc = ll + ".oc.ll"
        if not os.path.exists(oc): continue
        raw, opt = funcs(ll), funcs(oc)
        groups = collections.defaultdict(list)
        for fn, body in raw.items():
            nblk = len(re.findall(r'^"@\d+":', body, flags=re.M))
            if nblk < 5 or fn not in opt: continue
            total += 1
            o = opt[fn]
            lines = [l.strip() for l in o.splitlines()[1:] if l.strip() and not l.strip().endswith(":") and l.strip() != "}"
                     and not l.strip().startswith('"@')]
            if len(lines) == 1 and re.fullmatch(r"ret (i\d+|ptr) (-?\d+|true|false|null|undef|poison)", lines[0]):
                stub += 1
                if len(ex) < 12: ex.append((os.path.basename(ll), fn, lines[0]))
            groups[hashlib.sha1(norm(o, fn).encode()).hexdigest()].append(fn)
        collisions += sum(len(v) for v in groups.values() if len(v) > 1)
    print(f"{d}: functions(>=5 lifted blocks) {total}  constant-return stubs {stub}  in identical-IR groups {collisions}")
    print("   e.g.", ex[:8])
