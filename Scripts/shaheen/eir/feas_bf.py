"""Bodyfirst fused feasibility on the 30 ncat functions of feas.py: per function, llvm-dis -> eir_transform.agreed ->
strip_metadata -> process_one_file (purify, DDG, CFG) -> [bodyfirst: the define block moved to the front in the purified
and the instrumented IR, i.e. after the last opt output] -> FileProcessor.process_single_file (normalize, tokenize,
graph spans) -> wash.truncate_example at 2048. Prints per variant: ok, median input_ids, functions with CFG / DDG
before and after the 2048 cut, <unk> share.
usage: feas_bf.py <isolate.tar> <llvm-dis>"""
import os, subprocess, sys, tarfile, tempfile, statistics as stt
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
sys.path[:0] = [str(Path(__file__).resolve().parent), f"{REPO}", f"{REPO}/Scripts"]; os.environ["USE_TORCH"] = "0"
import eir_transform as T
from GraphBuilder.graph_generator import process_one_file
from DataProcess.file_processor import FileProcessor
from DataProcess.wash import truncate_example
from Tokenizer.ir_tokenizer import load_tokenizer
tok = load_tokenizer(f"{REPO}/Tokenizer/output_tokenizer/llvm_ir_bpe.json")
proc = FileProcessor(tok, cleanup_temp_files=False)
tarp, DIS = sys.argv[1], sys.argv[2]


def rewrite_bodyfirst(path):
    with open(path) as f:
        t = f.read()
    with open(path, "w") as f:
        f.write(T.body_first(t))


res = {"b": [], "bodyfirst": []}
with tempfile.TemporaryDirectory() as td, tarfile.open(tarp) as t:
    ms = [m for m in t.getmembers() if m.isfile() and "/function/" in "/" + m.name][:30]
    for variant in res:
        for m in ms:
            a = m.name.rsplit("/", 1)[-1].split(":")[0]; d = f"{td}/{variant}"; os.makedirs(d, exist_ok=True)
            bc, ll = f"{d}/{a}.bc", f"{d}/{a}.ll"
            open(bc, "wb").write(t.extractfile(m).read())
            r = subprocess.run([DIS, bc, "-o", "-"], capture_output=True, text=True)
            open(ll, "w").write(T.strip_metadata(T.agreed(r.stdout)))
            art = process_one_file(ll)
            if art is None:
                res[variant].append(None); continue
            if variant == "bodyfirst":
                rewrite_bodyfirst(art[1]); rewrite_bodyfirst(art[2])
            out = proc.process_single_file(*art)
            if not out.success:
                print(variant, a, "sample failed", out.error_message); res[variant].append(None); continue
            dd = out.to_dict(); n, c0, d0 = len(dd["input_ids"]), len(dd["cfg_graph"]), len(dd["ddg_graph"])
            unk = dd["input_ids"].count(tok.unk_token_id)
            ex = truncate_example({"input_ids": list(dd["input_ids"]), "cfg_graph": [list(e) for e in dd["cfg_graph"]],
                                   "ddg_graph": [list(e) for e in dd["ddg_graph"]]}, 2048, tok.eos_token_id)
            res[variant].append((n, c0, d0, len(ex["cfg_graph"]), len(ex["ddg_graph"]), unk))
            if variant == "bodyfirst" and len(res[variant]) == 1:
                print("first tokens:", tok.decode(dd["input_ids"][:40]))
for v, rs in res.items():
    ok = [x for x in rs if x]
    print(f"{v}: ok {len(ok)}/30, median ids {stt.median(x[0] for x in ok)}, CFG non-empty {sum(x[1] > 0 for x in ok)} "
          f"-> after 2048 cut {sum(x[3] > 0 for x in ok)}, DDG non-empty {sum(x[2] > 0 for x in ok)} -> after cut "
          f"{sum(x[4] > 0 for x in ok)}, CFG edges kept {sum(x[3] for x in ok)}/{sum(x[1] for x in ok)}, "
          f"DDG edges kept {sum(x[4] for x in ok)}/{sum(x[2] for x in ok)}, unk {sum(x[5] for x in ok)}")
