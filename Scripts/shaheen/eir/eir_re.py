"""S11-A step 2: one EIR .re per coreutils binary from its rev.ng isolate archive.

Per binary: every function module -> llvm-dis (rev.ng's LLVM 16) -> eir_transform.agreed (block labels, helpers as
declarations) -> @local_<addr> renamed to the S2 IDA name (offset rule of eir_text.py) -> global variable definitions
weak -> llvm-link (LLVM 18, no opt pass) -> 3 functions spot-checked against their own module (instructions equal
up to metadata) -> clang -m32|-m64 -c -fno-inline -fno-pic -fno-pie (the .re recipe of task4_re.py, no -O).

usage: eir_re.py <lift shard .tar (uncompressed)> <tag> <stage re dir> <status.jsonl> <ida_names.json> <llvm-dis> [--workers N] [--tmp DIR]
writes <stage re dir>/<bin>.re and the status file (one record per binary); eir_re.sbatch packs the .re files.
"""
import argparse, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eir_transform as T

M32 = {"arm32", "mips32", "x86"}
STRIP_META = re.compile(r"(, !\S+ !\d+)+|\s+#\d+|\s+!\S+ !\d+")


def body(text, fname):
    """Instruction lines of the definition of fname, metadata attachments and attribute refs removed."""
    lines = text.split("\n"); out = []; on = False
    for line in lines:
        if line.startswith("define ") and (T.ir_name(fname) + "(") in line:
            on = True; continue
        if on:
            if line.startswith("}"):
                break
            out.append(STRIP_META.sub("", line).rstrip())
    return [l for l in out if l and not l.lstrip().startswith(";")]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=7200, **kw)


def one(job):
    name, data, names, dis, tmp = job["name"], job["isolate"], job["names"], job["dis"], job["tmp"]
    t0 = time.time(); key = "coreutils/" + name.split("__", 1)[1]
    arch = key.rsplit("/", 1)[1].split("-")[0]
    rec = {"binary": key, "arch": arch, "ok": False}
    w = Path(tempfile.mkdtemp(prefix="eirre_", dir=tmp))
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            mem = [m for m in t.getmembers() if m.isfile() and "/function/" in "/" + m.name]
            addrs = T.local_addresses([m.name for m in mem])
            ida = {int(k): v for k, v in names.items()}
            off = T.load_offset(addrs, ida)
            thumb = T.thumb_addresses(job["binary_path"]) if arch == "arm32" else set()
            chosen, dups = T.choose_members([m.name for m in mem], ida, off, thumb)
            nm = {T.local_name(m): ida[ea] for m, ea in chosen.items()}       # local_<addr>_<isa> -> IDA name
            rec.update(functions=len(mem), ida_functions=len(ida), load_offset=hex(off), isa_duplicate_addresses=dups)
            defs, lls, own = {}, [], {}
            for m in mem:
                ln = T.local_name(m.name)
                bc = w / f"{ln}.bc"; bc.write_bytes(t.extractfile(m).read())
                r = run([dis, str(bc), "-o", "-"])
                if r.returncode:
                    rec["error"] = f"llvm-dis {m.name}: {r.stderr[-300:]}"; return rec
                txt = T.agreed(r.stdout)
                txt, _, _ = T.rename_locals(txt, nm)
                txt, d = T.weak_globals(txt)
                for g, b in d.items():
                    defs.setdefault(g, set()).add(b)
                ll = w / f"{ln}.ll"; ll.write_text(txt); lls.append(str(ll)); bc.unlink()
                own[ln] = ll
            rec.update(renamed=len(nm), unmatched_local=len(mem) - len(nm),
                       global_names=len(defs), global_conflicts=sorted(g for g, b in defs.items() if len(b) > 1)[:20],
                       n_global_conflicts=sum(len(b) > 1 for b in defs.values()))
        (w / "list.txt").write_text("\n".join(lls) + "\n")
        r = run(["/usr/lib/llvm-18/bin/llvm-link", "-o", str(w / "merged.bc"), "@" + str(w / "list.txt")])
        if r.returncode:
            rec["error"] = "llvm-link: " + r.stderr[-500:]; return rec
        # spot check: first, middle, last renamed function
        cand = sorted(nm)
        picks = [cand[0], cand[len(cand) // 2], cand[-1]] if cand else []
        merged_txt = run(["llvm-dis", str(w / "merged.bc"), "-o", "-"]).stdout
        checks = []
        for a in dict.fromkeys(picks):
            own_txt = run(["bash", "-c", f"llvm-as {own[a]} -o - | llvm-dis -o -"]).stdout  # same LLVM 18 printer as merged
            b1, b2 = body(own_txt, nm[a]), body(merged_txt, nm[a])
            checks.append({"function": nm[a], "instructions": len(b1), "equal": b1 == b2 and len(b1) > 0})
        rec["spot_check"] = checks
        out_re = Path(job["out_re"]) / (key.rsplit("/", 1)[1] + ".re")
        flag = "-m32" if arch in M32 else "-m64"
        r = run(["clang", flag, "-c", "-fno-inline", "-fno-pic", "-fno-pie", str(w / "merged.bc"), "-o", str(out_re)])
        if r.returncode or not out_re.exists() or out_re.stat().st_size == 0:
            rec["error"] = "clang: " + r.stderr[-500:]; return rec
        rec.update(ok=True, re_bytes=out_re.stat().st_size, clang_flag=flag)
        return rec
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[-500:]; return rec
    finally:
        rec["secs"] = round(time.time() - t0, 1)
        shutil.rmtree(w, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard"); ap.add_argument("tag"); ap.add_argument("re_dir"); ap.add_argument("status"); ap.add_argument("names"); ap.add_argument("dis")
    ap.add_argument("--workers", type=int, default=64); ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    names = json.load(open(a.names))
    re_dir = Path(a.re_dir); re_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"re_{a.tag}_", dir=a.tmp))
    data = open(a.shard, "rb").read()
    jobs = []
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            if m.name.endswith(".isolate.tar"):
                nmb = Path(m.name).name[:-len(".isolate.tar")]
                key = "coreutils/" + nmb.split("__", 1)[1]
                jobs.append({"name": nmb, "isolate": t.extractfile(m).read(), "names": names.get(key, {}), "dis": a.dis,
                             "tmp": str(stage), "out_re": str(re_dir), "binary_path": f"{REPO}/Binaries/Coreutils/{key}"})
    del data
    recs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs):
            recs.append(r); print(json.dumps({k: r.get(k) for k in ("binary", "ok", "functions", "renamed", "n_global_conflicts", "secs", "error")}), flush=True)
    with open(a.status, "w") as f:
        for r in sorted(recs, key=lambda r: r["binary"]):
            f.write(json.dumps(r) + "\n")
    shutil.rmtree(stage, ignore_errors=True)
    print(f"{a.tag}: {sum(r['ok'] for r in recs)}/{len(recs)} .re")


if __name__ == "__main__":
    main()
