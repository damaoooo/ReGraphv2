"""IDA function names of the coreutils original binaries from S2 (functions.json in each feature shard):
writes S11A/work/ida_names.json = {"coreutils/<bin>": {"<fva int>": name}}."""
import csv, glob, io, json, subprocess, tarfile
import os
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
F = f"{DATA}/S2/out/ida-features/coreutils_original"
key2bin = {}
for f in glob.glob(f"{F}/index/*.tsv"):
    for r in csv.DictReader(open(f), delimiter="\t"):
        key2bin[r["key"]] = r["idb_path"].replace("IDBs/", "")[:-len(".i64")]   # coreutils/<bin>
out = {}
for f in sorted(glob.glob(f"{F}/features/*.tar.zst")):
    data = subprocess.run(["zstd", "-dc", f], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            if m.name.endswith("/outputs/functions.json"):
                key = m.name.split("/")[0]
                out[key2bin[key]] = {str(int(x["fva"], 16)): x["name"] for x in json.load(t.extractfile(m))}
json.dump(out, open(f"{DATA}/S11A/work/ida_names.json", "w"))
print(len(out), "binaries,", sum(len(v) for v in out.values()), "functions")
