import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
T = os.environ.get("LR_WORK", os.getcwd())
L = os.environ.get("LIFTER_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.environ.get("HDIR", "h_v3")
TAG = os.environ.get("TAG", "")


def one(b):
    p = subprocess.run([os.environ.get("LIFT_PY", sys.executable), os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_classify.py"), L, f"{T}/ida/rr_{b}",
                        f"{T}/{H}/{b}.ll.lift.json", f"{T}/dt/rr{TAG}_{b}.json"], capture_output=True, text=True,
                       env=dict(os.environ, IDADIR=os.environ["IDADIR"]))
    return b, "\n".join(l for l in p.stdout.splitlines()[:12]) + p.stderr[-300:] * (p.returncode != 0)


with ThreadPoolExecutor(12) as ex:
    for b, out in ex.map(one, sys.argv[1:]):
        print("=====", b); print(out)
