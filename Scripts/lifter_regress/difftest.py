"""
Differential test of lifted functions against the original machine code (Unicorn).

usage: difftest.py BINARY LIFT_DIR [LIFT_DIR ...] --funcs f1,f2|--auto [--trials N] [--out JSON]

For each selected function: llvm-extract it from <LIFT_DIR>/<build>.ll.oc.ll, compile with llc for the
module's own (source) triple, link with ld.lld, then emulate the original function (at its address in
BINARY) and each compiled lifted version on the same random integer arguments and compare the return
value (masked to the lifted return type width).
--auto selects functions that are 'pure' in every LIFT_DIR's Oc IR: no load/store/alloca/non-intrinsic call,
integer/pointer args (<= register width), integer return, lifted without approx.
"""
import sys, os, re, json, random, subprocess, tempfile, argparse, struct
from elftools.elf.elffile import ELFFile
import unicorn as U
from unicorn import x86_const as X, arm_const as A, arm64_const as A64, mips_const as M, ppc_const as P

ap = argparse.ArgumentParser()
ap.add_argument("binary"); ap.add_argument("liftdirs", nargs="+")
ap.add_argument("--funcs"); ap.add_argument("--auto", action="store_true")
ap.add_argument("--trials", type=int, default=48); ap.add_argument("--out")
ap.add_argument("--limit", type=int, default=100000)
ap.add_argument("--max-funcs", type=int, default=400)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--ret-mask-bits", type=int, default=0)
a = ap.parse_args()
build = os.path.basename(a.binary)
arch = build.split("-")[0]
TMP = tempfile.mkdtemp(prefix="dt_", dir=os.environ.get("DT_TMP", os.getcwd()))

SENT = 0x0dead000
STACK, STACK_SZ = 0x7f000000, 0x100000


def read_funcs(path):
    out, cur, buf = {}, None, []
    for line in open(path):
        if line.startswith("define "):
            m = re.search(r'@("[^"]+"|[-\w.$]+)\(', line); cur, buf = m.group(1).strip('"'), [line]
        elif cur is not None:
            buf.append(line)
            if line.startswith("}"):
                out[cur] = "".join(buf); cur = None
    return out


def sig(body):
    head = body.splitlines()[0]
    m = re.match(r'define\s+(?:[\w]+\s+)*?(i\d+|ptr|void|float|double|\{.*?\}|\[.*?\])\s+@("[^"]+"|[-\w.$]+)\((.*)\)', head)
    if not m: return None
    ret, args = m.group(1), m.group(3)
    args = [x.strip().split()[0] for x in args.split(",") if x.strip()] if args.strip() else []
    return ret, args


PTRBITS = {"x64": 64, "x86": 32, "arm32": 32, "arm64": 64, "mips32": 32, "mips64": 64, "ppc32": 32, "ppc64": 64}[arch]


def bits(t):
    return PTRBITS if t == "ptr" else int(t[1:]) if re.fullmatch(r"i\d+", t) else None


def is_pure(body):
    for line in body.splitlines()[1:]:
        s = re.sub(r'%?"@\d+"', "", line.strip())
        if re.search(r"\b(load|store|alloca|va_arg|indirectbr|landingpad|invoke|fence|atomicrmw|cmpxchg)\b", s):
            return False
        if re.search(r"\bcall\b", s) and not re.search(r"@llvm\.(fshl|fshr|ctpop|ctlz|cttz|bswap|bitreverse|umin|umax|smin|smax|abs|[us](add|sub|mul)\.with\.overflow|[us](add|sub)\.sat)", s):
            return False
        if re.search(r"inttoptr|ptrtoint|getelementptr|@[\w\"]", s) and "@llvm." not in s:
            return False
        if re.search(r"\b(float|double|fp128|x86_fp80|half)\b", s):
            return False
    return True


# ---------------------------------------------------------------- ELF loading
def load_elf(uc, path, mapped):
    with open(path, "rb") as f:
        e = ELFFile(f)
        for seg in e.iter_segments():
            if seg["p_type"] != "PT_LOAD": continue
            va, msz = seg["p_vaddr"], seg["p_memsz"]
            lo, hi = va & ~0xfff, (va + msz + 0xfff) & ~0xfff
            for page in range(lo, hi, 0x1000):
                if page not in mapped:
                    uc.mem_map(page, 0x1000); mapped.add(page)
            uc.mem_write(va, seg.data())
        syms = {}
        st = e.get_section_by_name(".symtab")
        if st:
            for s in st.iter_symbols():
                if s["st_info"]["type"] == "STT_FUNC" and s["st_value"]:
                    syms[s.name] = s["st_value"]
        opd = e.get_section_by_name(".opd")
        opd = (opd["sh_addr"], opd.data()) if opd else None
        return syms, opd


def new_uc():
    m = {
        "x64": (U.UC_ARCH_X86, U.UC_MODE_64), "x86": (U.UC_ARCH_X86, U.UC_MODE_32),
        "arm32": (U.UC_ARCH_ARM, U.UC_MODE_ARM), "arm64": (U.UC_ARCH_ARM64, U.UC_MODE_ARM),
        "mips32": (U.UC_ARCH_MIPS, U.UC_MODE_MIPS32 | U.UC_MODE_BIG_ENDIAN),
        "mips64": (U.UC_ARCH_MIPS, U.UC_MODE_MIPS64 | U.UC_MODE_BIG_ENDIAN),
        "ppc32": (U.UC_ARCH_PPC, U.UC_MODE_PPC32 | U.UC_MODE_BIG_ENDIAN),
        "ppc64": (U.UC_ARCH_PPC, U.UC_MODE_PPC64 | U.UC_MODE_BIG_ENDIAN),
    }[arch]
    uc = U.Uc(*m)
    if arch == "mips64":
        with __import__("contextlib").suppress(Exception):
            uc.ctl_set_cpu_model(M.UC_CPU_MIPS64_MIPS64R2_GENERIC)
    if arch == "ppc64":
        with __import__("contextlib").suppress(Exception):
            uc.ctl_set_cpu_model(P.UC_CPU_PPC64_POWER9_V2_0)
    if arch == "arm32":
        with __import__("contextlib").suppress(Exception):
            uc.ctl_set_cpu_model(A.UC_CPU_ARM_CORTEX_A15)
    return uc


class Image:
    def __init__(self, path):
        self.uc = new_uc()
        self.mapped = set()
        self.syms, self.opd = load_elf(self.uc, path, self.mapped)
        self.uc.mem_map(STACK, STACK_SZ)
        self.uc.mem_map(SENT, 0x1000)
        if arch == "arm64":
            # enable FP/SIMD (CPACR_EL1.FPEN)
            with __import__("contextlib").suppress(Exception):
                self.uc.reg_write(A64.UC_ARM64_REG_CPACR_EL1, 0x300000)

    def toc_for(self, entry):
        if not self.opd: return None
        base, data = self.opd
        for i in range(0, len(data) - 15, 8):
            if struct.unpack(">Q", data[i:i + 8])[0] == entry:
                return struct.unpack(">Q", data[i + 8:i + 16])[0]
        return None


def setup_and_run(img, entry, ret_bits, args, toc=None):
    uc = img.uc
    sp = STACK + STACK_SZ - 0x1000
    uc.mem_write(STACK, b"\0" * STACK_SZ)
    W = PTRBITS // 8
    def wreg(r, v): uc.reg_write(r, v & ((1 << PTRBITS) - 1))
    # argument registers / stack, return value read-back
    if arch == "x64":
        for r, (v, b) in zip([X.UC_X86_REG_RDI, X.UC_X86_REG_RSI, X.UC_X86_REG_RDX, X.UC_X86_REG_RCX, X.UC_X86_REG_R8, X.UC_X86_REG_R9], args):
            wreg(r, v)
        sp -= 8; uc.mem_write(sp, struct.pack("<Q", SENT)); wreg(X.UC_X86_REG_RSP, sp)
        pc = entry
    elif arch == "x86":
        blob = b""
        for v, b in args:
            blob += struct.pack("<Q", v & (2**64 - 1)) if b == 64 else struct.pack("<I", v & 0xffffffff)
        sp -= 64
        uc.mem_write(sp, struct.pack("<I", SENT) + blob); wreg(X.UC_X86_REG_ESP, sp)
        pc = entry
    elif arch == "arm32":
        regs = [A.UC_ARM_REG_R0, A.UC_ARM_REG_R1, A.UC_ARM_REG_R2, A.UC_ARM_REG_R3]
        i = 0
        for v, b in args:
            if b == 64:
                i += i & 1
                if i + 1 >= 4: return "skip"
                wreg(regs[i], v & 0xffffffff); wreg(regs[i + 1], (v >> 32) & 0xffffffff); i += 2
            else:
                if i >= 4: return "skip"
                wreg(regs[i], v); i += 1
        wreg(A.UC_ARM_REG_SP, sp); wreg(A.UC_ARM_REG_LR, SENT)
        pc = entry
    elif arch == "arm64":
        for i, (v, b) in enumerate(args):
            wreg(A64.UC_ARM64_REG_X0 + i, v)
        wreg(A64.UC_ARM64_REG_SP, sp); wreg(A64.UC_ARM64_REG_X30, SENT)
        pc = entry
    elif arch == "mips32":
        regs = [M.UC_MIPS_REG_4, M.UC_MIPS_REG_5, M.UC_MIPS_REG_6, M.UC_MIPS_REG_7]
        i = 0
        for v, b in args:
            if b == 64:
                i += i & 1
                if i + 1 >= 4: return "skip"
                wreg(regs[i], (v >> 32) & 0xffffffff); wreg(regs[i + 1], v & 0xffffffff); i += 2
            else:
                if i >= 4: return "skip"
                wreg(regs[i], v); i += 1
        sp -= 32; wreg(M.UC_MIPS_REG_SP, sp); wreg(M.UC_MIPS_REG_RA, SENT); wreg(M.UC_MIPS_REG_T9, entry)
        pc = entry
    elif arch == "mips64":
        for i, (v, b) in enumerate(args):
            if b < 64:  # n64: 32-bit values are kept sign-extended in registers
                v &= (1 << b) - 1
                if b == 32 and v >> 31: v -= 1 << 32
            wreg(M.UC_MIPS_REG_4 + i, v)
        wreg(M.UC_MIPS_REG_SP, sp); wreg(M.UC_MIPS_REG_RA, SENT); wreg(M.UC_MIPS_REG_T9, entry)
        pc = entry
    elif arch in ("ppc32", "ppc64"):
        i = 0
        for v, b in args:
            if arch == "ppc32" and b == 64:
                i += (i & 1)
                if i + 1 >= 8: return "skip"
                wreg(P.UC_PPC_REG_3 + i, (v >> 32) & 0xffffffff); wreg(P.UC_PPC_REG_3 + i + 1, v & 0xffffffff); i += 2
            else:
                if i >= 8: return "skip"
                wreg(P.UC_PPC_REG_3 + i, v); i += 1
        sp -= 256
        wreg(P.UC_PPC_REG_1, sp); wreg(P.UC_PPC_REG_LR, SENT)
        if toc is not None: wreg(P.UC_PPC_REG_2, toc)
        wreg(P.UC_PPC_REG_12, entry)
        pc = entry
    try:
        uc.emu_start(pc, SENT, count=a.limit)
    except U.UcError as e:
        return f"emu_error:{e}"
    cur = uc.reg_read({"x64": X.UC_X86_REG_RIP, "x86": X.UC_X86_REG_EIP, "arm32": A.UC_ARM_REG_PC, "arm64": A64.UC_ARM64_REG_PC,
                       "mips32": M.UC_MIPS_REG_PC, "mips64": M.UC_MIPS_REG_PC, "ppc32": P.UC_PPC_REG_PC, "ppc64": P.UC_PPC_REG_PC}[arch])
    if (cur & ~1) != SENT:
        return "no_return"
    if arch == "x64": r = uc.reg_read(X.UC_X86_REG_RAX); r |= uc.reg_read(X.UC_X86_REG_RDX) << 64
    elif arch == "x86": r = uc.reg_read(X.UC_X86_REG_EAX) | (uc.reg_read(X.UC_X86_REG_EDX) << 32)
    elif arch == "arm32": r = uc.reg_read(A.UC_ARM_REG_R0) | (uc.reg_read(A.UC_ARM_REG_R1) << 32)
    elif arch == "arm64": r = uc.reg_read(A64.UC_ARM64_REG_X0)
    elif arch == "mips32": r = (uc.reg_read(M.UC_MIPS_REG_V1) | (uc.reg_read(M.UC_MIPS_REG_V0) << 32)) if ret_bits == 64 else uc.reg_read(M.UC_MIPS_REG_V0)
    elif arch == "mips64": r = uc.reg_read(M.UC_MIPS_REG_V0)
    elif arch == "ppc32": r = (uc.reg_read(P.UC_PPC_REG_4) | (uc.reg_read(P.UC_PPC_REG_3) << 32)) if ret_bits == 64 else uc.reg_read(P.UC_PPC_REG_3)
    elif arch == "ppc64": r = uc.reg_read(P.UC_PPC_REG_3)
    return r & ((1 << ret_bits) - 1)


def compile_lifted(liftdir, fn):
    src = os.path.join(liftdir, build + ".ll.oc.ll")
    d = tempfile.mkdtemp(dir=TMP)
    f_ll, f_o, f_elf = os.path.join(d, "f.ll"), os.path.join(d, "f.o"), os.path.join(d, "f.elf")
    p = subprocess.run(["llvm-extract", "-S", "--func=" + fn, src, "-o", f_ll], capture_output=True, text=True)
    if p.returncode: return None, "extract:" + p.stderr[:200]
    triple = re.search(r'target triple = "([^"]+)"', open(f_ll).read()).group(1)
    extra = []
    if arch == "mips64": extra = ["-target-abi", "n64"]
    p = subprocess.run(["llc", "-O1", "-mtriple=" + triple, "-relocation-model=static", "-filetype=obj", f_ll, "-o", f_o] + extra,
                       capture_output=True, text=True)
    if p.returncode: return None, "llc:" + p.stderr[:200]
    p = subprocess.run(["ld.lld", "-static", "--no-dynamic-linker", "-e", "0", "--section-start=.text=0x400000", "-o", f_elf, f_o],
                       capture_output=True, text=True)
    if p.returncode: return None, "ld:" + p.stderr[:200]
    return f_elf, None


def gen_args(argbits, rng):
    out = []
    for b in argbits:
        regw = PTRBITS
        pool = [0, 1, 2, 3, 7, 8, 9, 10, 31, 32, 33, 47, 48, 57, 64, 65, 90, 91, 96, 97, 122, 123, 127, 128, 255, 256,
                rng.randrange(0, 128), rng.randrange(0, 1 << min(b, 16)), rng.randrange(0, 1 << b)]
        if b >= regw or arch == "mips64":
            pool += [(1 << b) - 1, 1 << (b - 1), (1 << (b - 1)) - 1, rng.randrange(0, 1 << b), (1 << b) - rng.randrange(1, 300)]
        else:
            # narrower than the register: keep the top bit clear so sign/zero extension by the caller agree
            pool += [(1 << (b - 1)) - 1, rng.randrange(0, 1 << (b - 1))]
        v = rng.choice(pool) & ((1 << b) - 1)
        if b < regw and arch != "mips64":
            v &= (1 << (b - 1)) - 1
        out.append((v, b))
    return out


def main():
    rng = random.Random(a.seed)
    bodies = [read_funcs(os.path.join(d, build + ".ll.oc.ll")) for d in a.liftdirs]
    recs = []
    for d in a.liftdirs:
        recs.append({f.get("llvm_name"): f for f in json.load(open(os.path.join(d, build + ".ll.lift.json")))["functions"] if f.get("status") == "lifted"})
    if a.auto:
        cands = []
        for fn, body in bodies[-1].items():
            if not all(fn in b for b in bodies) or not all(fn in r for r in recs): continue
            if any(r[fn].get("approx") for r in recs): continue
            if not all(is_pure(b[fn]) for b in bodies): continue
            s = sig(bodies[-1][fn])
            if not s: continue
            ret, args = s
            if bits(ret) is None or any(bits(x) is None or bits(x) > PTRBITS for x in args) or bits(ret) > 64: continue
            if all(sig(b[fn]) != s for b in bodies[:-1]) and len(bodies) > 1: pass
            cands.append(fn)
        rng.shuffle(cands)
        names = cands[:a.max_funcs]
    else:
        names = a.funcs.split(",")
    orig = Image(a.binary)
    results = {}
    for fn in names:
        rec = recs[-1].get(fn) or {}
        ea = int(rec["ea"], 16) if rec else orig.syms.get(fn)
        if ea is None:
            results[fn] = {"error": "no ea"}; continue
        s = sig(bodies[-1][fn]); ret_bits = bits(s[0]); argbits = [bits(x) for x in s[1]]
        thumb = arch == "arm32" and (orig.syms.get(rec.get("name", fn), 0) & 1)
        o_entry = ea | 1 if thumb else ea
        toc = orig.toc_for(ea) if arch == "ppc64" else None
        imgs = []
        for d in a.liftdirs:
            sd = sig(bodies[a.liftdirs.index(d)][fn])
            if sd is None or sd[1] != s[1] or bits(sd[0]) is None:
                imgs.append(("sigdiff", None, None)); continue
            elf, err = compile_lifted(d, fn)
            if err: imgs.append((err, None, None)); continue
            img = Image(elf)
            entry = img.syms.get(fn)
            ltoc = None
            if arch == "ppc64" and entry is not None and img.opd:
                base, data = img.opd
                off = entry - base
                ltoc = struct.unpack(">Q", data[off + 8:off + 16])[0]
                entry = struct.unpack(">Q", data[off:off + 8])[0]
            imgs.append((None, img, (entry, ltoc, bits(sd[0]))))
        per = [{"agree": 0, "disagree": 0, "skip": 0, "err": e} for e, _, _ in imgs]
        ex = [None] * len(imgs)
        for t in range(a.trials):
            args = gen_args(argbits, rng)
            want = setup_and_run(orig, o_entry, ret_bits, args, toc)
            if isinstance(want, str):
                for p in per: p["skip"] += 1
                continue
            for k, (err, img, ent) in enumerate(imgs):
                if err: continue
                rb = min(ret_bits, ent[2])
                if a.ret_mask_bits: rb = min(rb, a.ret_mask_bits)
                got = setup_and_run(img, ent[0], rb, args, ent[1])
                if isinstance(got, int) and got == want & ((1 << rb) - 1): per[k]["agree"] += 1
                else:
                    per[k]["disagree"] += 1
                    if ex[k] is None: ex[k] = {"args": [hex(v) for v, _ in args], "want": want if isinstance(want, str) else hex(want), "got": got if isinstance(got, str) else hex(got)}
        for k, p in enumerate(per):
            if ex[k]: p["example"] = ex[k]
        results[fn] = {"ea": hex(ea), "sig": s, "per_dir": per}
    if a.out:
        json.dump({"binary": a.binary, "dirs": a.liftdirs, "results": results}, open(a.out, "w"), indent=1)
    # summary
    for k, d in enumerate(a.liftdirs):
        ok = sum(1 for r in results.values() if "per_dir" in r and r["per_dir"][k]["agree"] and not r["per_dir"][k]["disagree"])
        bad = sum(1 for r in results.values() if "per_dir" in r and r["per_dir"][k]["disagree"])
        noorig = sum(1 for r in results.values() if "per_dir" in r and not r["per_dir"][k]["agree"] and not r["per_dir"][k]["disagree"])
        print(f"{build} {d}: tested {len(results)} agree {ok} disagree {bad} untestable {noorig}")


main()
