"""ircmp.py BASE_DIR NEW_DIR [-j N] : for functions lifted by both, compare raw IR and Oc-normalized IR.
Writes NEW_DIR/ircmp.json with per-function categories."""
import sys, json, os, re, collections, glob, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
PASSES = "sroa,mem2reg,simplifycfg,early-cse,sccp,correlated-propagation,jump-threading,simplifycfg,reassociate,gvn,dce,bdce,adce,simplifycfg"


def oc(path):
    out = path + ".oc.ll"
    if not os.path.exists(out):
        p = subprocess.run(["opt", "-S", "-passes=" + PASSES, path, "-o", out], capture_output=True, text=True)
        if p.returncode:
            print("opt failed", path, p.stderr[:300]); return None
    return out


def funcs(path):
    out, cur, buf = {}, None, []
    if path is None: return out
    for line in open(path):
        if line.startswith("define "):
            m = re.search(r'@"?([^"(]+)"?\(', line); cur, buf = m.group(1), [line]
        elif cur is not None:
            buf.append(line)
            if line.startswith("}"):
                out[cur] = "".join(buf); cur = None
    return out


TOK = re.compile(r'%(?:"[^"]*"|[-\w.$]+)')


def norm(body, name):
    body = re.sub(r", align \d+", "", body)
    body = body.replace('@"%s"' % name, "@SELF").replace("@%s(" % name, "@SELF(")
    m = {}
    def sub(t):
        k = t.group(0)
        if k not in m: m[k] = "%%v%d" % len(m)
        return m[k]
    body = TOK.sub(sub, body)
    # block labels
    body = re.sub(r'^("[^"]*"|[-\w.$]+):', lambda t: "L:", body, flags=re.M)
    return body


