"""EIR text for irnorm from the part B texts (same text as the delivered rell): usage text_b_view.py [COLUMN LEVEL]
(default text_b EIR) -> text_b/<LEVEL>/coreutils/<bin>.parquet with columns function_name, text."""
import glob, os, sys
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
COL, LEVEL = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("text_b", "EIR")
SRC, DST = f"{DATA}/S11A/out_tb/text/coreutils", f"{DATA}/S11A/irnorm/text_b/{LEVEL}/coreutils"
os.makedirs(DST, exist_ok=True)
def one(f):
    t = pq.read_table(f, columns=["function_name", COL]).rename_columns(["function_name", "text"])
    pq.write_table(t, f"{DST}/{os.path.basename(f)}"); return t.num_rows
with ProcessPoolExecutor(64) as ex: print(sum(ex.map(one, sorted(glob.glob(f"{SRC}/*.parquet")))))
