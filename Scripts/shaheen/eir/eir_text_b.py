"""S11 part B text, rule set (b) (9950X 2026-11-04 21:36), for one lift shard (coreutils): every rev.ng function that
maps to an IDA function (eir_transform.choose_members, the rule of eir_re.py / eir_b.py) ->
llvm-dis (rev.ng's LLVM 16) -> agreed (1 block labels, 2 helpers as declarations) -> strip_metadata (B3) ->
_opt_initial_purify + normalize_file = text_b; text_b_bodyfirst = body_first(text_b) (the define block first, nothing
removed); text_main = normalize_file(main_form(purified IR)) (M3 + M4: type and global variable definitions removed). -> <text_root>/coreutils/<bin>.parquet (function_name = the lift's LLVM name, ida_name, fva, text_b,
text_b_bodyfirst, text_main) and <status>/<tag>.jsonl (per binary; failed functions listed with their step).
usage: eir_text_b.py <lift shard .tar> <tag> <text root> <status dir> <ida_names.json> <llvm-dis> [--workers N] [--tmp DIR]
"""
import argparse, io, json, os, shutil, subprocess, sys, tarfile, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
sys.path[:0] = [str(Path(__file__).resolve().parent), f"{REPO}", f"{REPO}/Scripts"]
os.environ["USE_TORCH"] = "0"
import eir_transform as T


def one(job):
    from GraphBuilder.graph_generator import _opt_initial_purify
    from Tokenizer.normalizer import normalize_file
    import pyarrow as pa, pyarrow.parquet as pq
    key, data, names, dis, tmp = job["key"], job["isolate"], job["names"], job["dis"], job["tmp"]
    b = key.rsplit("/", 1)[1]; t0 = time.time()
    w = Path(tempfile.mkdtemp(prefix="eirtb_", dir=tmp))
    st = {"binary": key, "ok": False, "failed": []}
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            members = {m.name: m for m in t.getmembers() if m.isfile() and "/function/" in "/" + m.name}
            ida = {int(k): v for k, v in names.items()}
            lift = json.load(open(f"{DATA}/Coreutils-ll/{key}.ll.lift.json"))["functions"]
            llvm = {int(x["ea"], 16): x["llvm_name"] for x in lift if x.get("status") == "lifted" and x.get("llvm_name")}
            off = T.load_offset(T.local_addresses(members), ida)
            thumb = T.thumb_addresses(f"{REPO}/Binaries/Coreutils/{key}") if b.startswith("arm32-") else set()
            chosen, dups = T.choose_members(list(members), ida, off, thumb)
            st.update(revng_functions=len(members), ida_functions=len(ida), matched=len(chosen), load_offset=hex(off),
                      isa_duplicate_addresses=dups)
            rows = []
            for n, ea in sorted(chosen.items(), key=lambda x: x[1]):
                nm = ida[ea]
                bc = w / "f.bc"; bc.write_bytes(t.extractfile(members[n]).read())
                r = subprocess.run([dis, str(bc), "-o", "-"], capture_output=True, text=True, timeout=3600)
                if r.returncode:
                    st["failed"].append([nm, hex(ea), "llvm_dis"]); continue
                ll = w / f"{ea:x}.ll"; ll.write_text(T.strip_metadata(T.agreed(r.stdout)))
                gw = w / "g"; gw.mkdir(exist_ok=True)
                pur = _opt_initial_purify(str(ll), str(gw))
                if not pur:
                    st["failed"].append([nm, hex(ea), "purify"]); ll.unlink(); continue
                try:
                    txt = normalize_file(pur)
                    mf = w / "main.ll"; mf.write_text(T.main_form(open(pur).read()))
                    main = normalize_file(str(mf)); mf.unlink()
                finally:
                    os.remove(pur); ll.unlink()
                if not txt or not main:
                    st["failed"].append([nm, hex(ea), "normalize"]); continue
                rows.append((llvm.get(ea, nm), nm, hex(ea), txt, T.body_first(txt), main))
        tdir = Path(job["text_root"]) / "coreutils"; tdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"function_name": [x[0] for x in rows], "ida_name": [x[1] for x in rows],
                                 "fva": [x[2] for x in rows], "text_b": pa.array([x[3] for x in rows], pa.large_string()),
                                 "text_b_bodyfirst": pa.array([x[4] for x in rows], pa.large_string()),
                                 "text_main": pa.array([x[5] for x in rows], pa.large_string())}), str(tdir / f"{b}.parquet"))
        st.update(ok=True, texts=len(rows), residual_bang=sum("!" in x[3] for x in rows), main_residual_defs=sum(
                      any(l.startswith("%") and " = type " in l for l in x[5].split("\n")) for x in rows),
                  bodyfirst_unmoved=sum(x[3] == x[4] for x in rows))
    except Exception as e:  # noqa: BLE001
        st["error"] = repr(e)[-500:]
    finally:
        st["secs"] = round(time.time() - t0, 1); shutil.rmtree(w, ignore_errors=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    for x in ("shard", "tag", "text_root", "status", "names", "dis"):
        ap.add_argument(x)
    ap.add_argument("--workers", type=int, default=96); ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    names = json.load(open(a.names))
    stage = Path(tempfile.mkdtemp(prefix=f"tb_{a.tag}_", dir=a.tmp)); Path(a.status).mkdir(parents=True, exist_ok=True)
    jobs = []
    with tarfile.open(a.shard) as t:
        for m in t.getmembers():
            if m.name.endswith(".isolate.tar"):
                key = "coreutils/" + Path(m.name).name[:-len(".isolate.tar")].split("__", 1)[1]
                jobs.append({"key": key, "isolate": t.extractfile(m).read(), "names": names.get(key, {}), "dis": a.dis,
                             "tmp": str(stage), "text_root": a.text_root})
    stats = []
    with ProcessPoolExecutor(a.workers) as ex:
        for s in ex.map(one, jobs):
            stats.append(s)
            print(json.dumps({k: s.get(k) for k in ("binary", "ok", "matched", "texts", "residual_bang", "secs", "error")}), flush=True)
    with open(f"{a.status}/{a.tag}.jsonl", "w") as f:
        for s in sorted(stats, key=lambda s: s["binary"]): f.write(json.dumps(s) + "\n")
    shutil.rmtree(stage, ignore_errors=True)
    print(f"{a.tag}: {sum(s['ok'] for s in stats)}/{len(stats)} binaries, texts {sum(s.get('texts', 0) for s in stats)}, "
          f"failed functions {sum(len(s['failed']) for s in stats)}")


if __name__ == "__main__":
    main()
