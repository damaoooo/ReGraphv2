"""showdiff.py BASE_DIR NEW_DIR build func [raw]: unified diff of one function (Oc by default)."""
import sys, os, difflib
sys.path.insert(0, os.path.dirname(__file__))
from ircmp_lib import funcs, norm
base, new, build, fn = sys.argv[1:5]
raw = len(sys.argv) > 5
suf = ".ll" if raw else ".ll.oc.ll"
a = funcs(os.path.join(base, build + suf)).get(fn, "")
b = funcs(os.path.join(new, build + suf)).get(fn, "")
if not raw:
    a, b = norm(a, fn), norm(b, fn)
sys.stdout.writelines(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "base", "new", n=2))
