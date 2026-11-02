"""Lift binaries in parallel with a given lifter. usage: runlift.py LIFTER OUTDIR -j N BIN..."""
import sys, os, subprocess, shutil, tempfile, argparse
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("lifter"); ap.add_argument("outdir"); ap.add_argument("-j", type=int, default=8)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--timeout", type=int, default=7200)
ap.add_argument("bins", nargs="+")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
def one(b):
    name = os.path.basename(b)
    out = os.path.join(os.path.abspath(a.outdir), name + ".ll")
    if os.path.exists(out + ".lift.json"):
        return name, "cached"
    d = tempfile.mkdtemp(dir=os.path.abspath(a.outdir), prefix=".w_")
    shutil.copy(b, os.path.join(d, name))
    env = dict(os.environ, IDADIR=os.environ.get("IDADIR", "/home/damaoooo/ida-pro-9.4"))
    for kv in a.env:
        k, v = kv.split("=", 1); env[k] = v
    try:
        p = subprocess.run([os.environ.get("LIFT_PY", "/home/damaoooo/miniconda3/envs/ReLL/bin/python"), a.lifter, "-f", os.path.join(d, name), "-o", out],
                           env=env, capture_output=True, text=True, timeout=a.timeout)
        tail = [l for l in p.stderr.splitlines() if l.startswith("lift stats") or l.startswith("Error")]
        with open(out + ".log", "w") as f: f.write(p.stderr[-20000:])
        return name, " ".join(tail)
    except subprocess.TimeoutExpired:
        return name, "TIMEOUT"
    finally:
        shutil.rmtree(d, ignore_errors=True)
with ThreadPoolExecutor(a.j) as ex:
    for name, msg in ex.map(one, a.bins):
        print(name, msg, flush=True)
