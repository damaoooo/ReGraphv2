import idapro  # 必须是第一个 import，用于 idalib
"""
Ablation lifter: translate Hex-Rays' EARLIEST microcode (MMAT_GENERATED) to LLVM IR.

This is a separate translator for the "lifting without any Hex-Rays optimization" ablation tier
(user decision 2026-11-10). The production lifter `ida2llvm.py` is NOT modified: this file imports it
read-only and reuses its module/target setup, data globals, symbol resolution and streaming writer;
only function bodies are produced here. At MMAT_GENERATED the microcode is an instruction-by-instruction
translation: no local variables, no types, no call arguments, no stack variables, no CFG edges yet.

Translation model (pure translation, no inference):
- Every microregister (incl. flags and Hex-Rays temporaries) lives in a module-level global. Overlapping
  views (al/ax/eax/rax, or a pair) are merged into one global per connected byte range found in a first
  pass over all functions of the binary; partial accesses are shifts/masks on that integer (microregister
  numbering is little-endian by definition, independent of the target's memory endianness).
- All functions are `void ()`; calls carry no arguments (Hex-Rays recovers them only later). Values flow
  through the register globals, as on the machine.
- ldx/stx -> load/store through inttoptr(address); the segment operand is ignored, except fs/gs which
  address the lifter's `virtual_fs` buffer (same approximation as production).
- Integer constants that IDA's disassembler cross-references from the instruction (data or code xref) are
  emitted as symbol addresses (ptrtoint @sym), so the recompiled object keeps relocations; mop_v / mop_a
  operands are symbols by construction. Nothing else is symbolized.
- Branch targets: mop_b = block serial, mop_v = address of a block start in this function (direct jumps to
  another function's start = tail call). ijmp at a return instruction (is_ret_insn) = `ret void`; ijmp
  with IDA switch info = `switch` on the target address over the case blocks; any other ijmp = indirect
  tail call. These use only disassembler-level information.
- No production microcode hooks (ARM carry, AArch64 division, rotate) are installed: the microcode is
  taken exactly as Hex-Rays generates it.
"""
import logging
import os
import sys
import time
import gc
import json
import hashlib
from collections import Counter
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ida2llvm as L  # noqa: E402  (production lifter, used read-only)

import ida_auto  # noqa: E402
import ida_funcs  # noqa: E402
import ida_hexrays as h  # noqa: E402
import ida_idaapi  # noqa: E402
import ida_idp  # noqa: E402
import ida_nalt  # noqa: E402
import ida_xref  # noqa: E402
import idautils  # noqa: E402
import typer  # noqa: E402
from llvmlite import ir  # noqa: E402

MODE = "hexrays-generated"
_HERE = Path(__file__).resolve()
LIFTER_ID = hashlib.sha256(_HERE.read_bytes() + (_HERE.parent / "ida2llvm.py").read_bytes()).hexdigest()[:16]
VOID_FT = ir.FunctionType(ir.VoidType(), [])
MAX_CLUSTER_BYTES = 1 << 16  # byte-array clusters have no width cost; the per-view fallback is a safety valve only

_COND = {  # opcode -> (icmp kind, signed)
    h.m_setz: ("==", False), h.m_setnz: ("!=", False), h.m_setae: (">=", False), h.m_setb: ("<", False),
    h.m_seta: (">", False), h.m_setbe: ("<=", False), h.m_setg: (">", True), h.m_setge: (">=", True),
    h.m_setl: ("<", True), h.m_setle: ("<=", True),
    h.m_jz: ("==", False), h.m_jnz: ("!=", False), h.m_jae: (">=", False), h.m_jb: ("<", False),
    h.m_ja: (">", False), h.m_jbe: ("<=", False), h.m_jg: (">", True), h.m_jge: (">=", True),
    h.m_jl: ("<", True), h.m_jle: ("<=", True),
}
_JCC = {h.m_jz, h.m_jnz, h.m_jae, h.m_jb, h.m_ja, h.m_jbe, h.m_jg, h.m_jge, h.m_jl, h.m_jle}
_BIN = {h.m_add: "add", h.m_sub: "sub", h.m_mul: "mul", h.m_udiv: "udiv", h.m_sdiv: "sdiv",
        h.m_umod: "urem", h.m_smod: "srem", h.m_or: "or_", h.m_and: "and_", h.m_xor: "xor",
        h.m_shl: "shl", h.m_shr: "lshr", h.m_sar: "ashr"}
_OPNAME = {getattr(h, n): n[2:] for n in dir(h) if n.startswith("m_") and isinstance(getattr(h, n), int)}
_FP_OPS = {h.m_fadd, h.m_fsub, h.m_fmul, h.m_fdiv, h.m_fneg, h.m_f2f, h.m_i2f, h.m_u2f, h.m_f2i, h.m_f2u}
_FBIN = {h.m_fadd: "fadd", h.m_fsub: "fsub", h.m_fmul: "fmul", h.m_fdiv: "fdiv"}
_LLVM_HELPERS = {  # helper name -> (llvm intrinsic, kind)
    "__ROL1__": ("llvm.fshl.i8", "rot"), "__ROL2__": ("llvm.fshl.i16", "rot"),
    "__ROL4__": ("llvm.fshl.i32", "rot"), "__ROL8__": ("llvm.fshl.i64", "rot"),
    "__ROR1__": ("llvm.fshr.i8", "rot"), "__ROR2__": ("llvm.fshr.i16", "rot"),
    "__ROR4__": ("llvm.fshr.i32", "rot"), "__ROR8__": ("llvm.fshr.i64", "rot"),
}


class Reject(L.UnsupportedMicrocode):
    pass


def gen_mba(f):
    hf = h.hexrays_failure_t()
    mba = h.gen_microcode(h.mba_ranges_t(f), hf, h.mlist_t(),
                          h.DECOMP_NO_WAIT | h.DECOMP_NO_CACHE, h.MMAT_GENERATED)
    return mba, hf


def _iter_mops(m):
    yield m
    if m.t == h.mop_d:
        yield from _iter_insn_mops(m.d)
    elif m.t == h.mop_f:
        for a in m.f.args:
            yield from _iter_mops(a)
    elif m.t == h.mop_a and m.a is not None:
        yield from _iter_mops(m.a)
    elif m.t == h.mop_p:
        yield from _iter_mops(m.pair.lop)
        yield from _iter_mops(m.pair.hop)


def _iter_insn_mops(ins):
    for m in (ins.l, ins.r, ins.d):
        yield from _iter_mops(m)


class RegisterFile:
    """Module-level globals for microregisters; overlapping byte ranges share one integer global."""

    def __init__(self, module):
        self.module = module
        self.intervals = {}  # (mreg, size) seen
        self.cluster_of = {}  # byte -> (base, span)
        self.globals = {}

    def collect(self, mba):
        for i in range(mba.qty):
            ins = mba.get_mblock(i).head
            while ins is not None:
                for m in _iter_insn_mops(ins):
                    if m.t == h.mop_r and m.size > 0:
                        self.intervals[(int(m.r), int(m.size))] = True
                ins = ins.next

    def build(self):
        spans = sorted((r, r + s) for r, s in self.intervals)
        merged = []
        for lo, hi in spans:
            if merged and lo < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        self.wide = set()
        # clusters accessed only through one full-width view stay plain integers
        views = {}
        for r, sz in self.intervals:
            views.setdefault(r, set()).add(sz)
        self.single = {}
        for lo, hi in merged:
            self.single[lo] = views.get(lo) == {hi - lo} and not any(lo < r < hi for r in views)
        for lo, hi in merged:
            if hi - lo > MAX_CLUSTER_BYTES:
                self.wide.update(range(lo, hi))
                continue
            for b in range(lo, hi):
                self.cluster_of[b] = (lo, hi - lo)

    def _global(self, key, nbytes, as_bytes):
        g = self.globals.get(key)
        if g is None:
            base = key[0]
            name = h.get_mreg_name(base, nbytes) or f"r{base}"
            gname = f"__mc_{L._sanitize_symbol_name(name)}_{base}_{nbytes}"
            typ = ir.ArrayType(L.i8, nbytes) if as_bytes else ir.IntType(nbytes * 8)
            g = ir.GlobalVariable(self.module, typ, gname)
            g.initializer = ir.Constant(typ, None)
            g.linkage = "internal"
            self.globals[key] = g
        return g

    def _ptr(self, b, r, size):
        """Pointer (iN*) to microregister bytes [r, r+size). Clusters with overlapping views are byte arrays:
        microregister byte k is the k-th least significant byte, so on big-endian targets the in-memory
        offset is mirrored within the cluster."""
        if r in self.wide or r not in self.cluster_of:
            return self._global((r, size, "w"), size, False)
        base, span = self.cluster_of[r]
        if r + size > base + span:
            raise Reject("register_outside_cluster", f"{r}.{size}")
        if span == size and self.single.get(base, False):
            return self._global((base, span), span, False)
        g = self._global((base, span), span, True)
        off = (r - base) if not L.is_big_endian else (span - (r - base) - size)
        p = b.gep(g, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), off)], inbounds=True)
        return b.bitcast(p, ir.IntType(size * 8).as_pointer())

    def _slot(self, r, size):  # used for &register: (global, byte offset, span)
        if r in self.wide or r not in self.cluster_of:
            return self._global((r, size, "w"), size, False), 0, size
        base, span = self.cluster_of[r]
        if span == size and self.single.get(base, False):
            return self._global((base, span), span, False), 0, span
        off = (r - base) if not L.is_big_endian else (span - (r - base) - size)
        return self._global((base, span), span, True), off, span

    def read(self, b, r, size):
        return b.load(self._ptr(b, r, size))

    def write(self, b, r, size, val):
        b.store(fit(b, val, size * 8), self._ptr(b, r, size))


def fit(b, v, bits):
    """Integer value to exactly `bits` (zext/trunc); floats are bitcast first."""
    if not isinstance(v.type, ir.IntType):
        v = b.bitcast(v, ir.IntType(_fp_bits(v.type)))
    if v.type.width == bits:
        return v
    if v.type.width < bits:
        return b.zext(v, ir.IntType(bits))
    return b.trunc(v, ir.IntType(bits))


def _fp_bits(t):
    return 32 if isinstance(t, ir.FloatType) else 64


def _fnum_value(w):
    """IDA fnumber_t words (little-endian 16-bit; last = sign|exponent, explicit integer bit) -> float."""
    import math
    se = w[-1]
    sign = -1.0 if se & 0x8000 else 1.0
    exp = se & 0x7FFF
    mant = 0
    for i, x in enumerate(w[:-1]):
        mant |= int(x) << (16 * i)
    if exp == 0x7FFF:
        return sign * math.inf if mant & ((1 << (16 * (len(w) - 1) - 1)) - 1) == 0 else math.nan
    if mant == 0:
        return sign * 0.0
    try:
        return sign * math.ldexp(float(mant), exp - 16383 - (16 * (len(w) - 1) - 1))
    except OverflowError:
        return sign * math.inf


def _fp_type(size):
    return {4: ir.FloatType(), 8: ir.DoubleType()}.get(size)


class FunctionTranslator:
    def __init__(self, ctl, regs, ea, mba):
        self.ctl, self.regs, self.ea, self.mba = ctl, regs, ea, mba
        self.m = ctl.m
        self.approx = set()
        self.xref_cache = {}

    # ---- symbols -------------------------------------------------------------------------------------
    def declare(self, ea):
        """void() declaration for the function at ea (thunks resolved like production)."""
        ea = L._resolve_thunk(ea)
        name = L._get_name_cached(ea) or f"data_{ea:x}"
        g = self.m.globals.get(name)
        if g is None:
            g = ir.Function(self.m, VOID_FT, name)
            g._source_ea = ea
        return g if isinstance(g, ir.Function) else None

    def callee_ptr(self, b, fn):
        if fn.ftype == VOID_FT:
            return fn
        return b.bitcast(fn, VOID_FT.as_pointer())

    def sym_int(self, b, value, bits):
        """Address `value` as an integer of `bits`, through the symbol IDA resolves it to."""
        p = L._pointer_constant(self.m, L.i8ptr, value)
        return b.ptrtoint(p, ir.IntType(bits))

    def xrefs_from(self, ea):
        s = self.xref_cache.get(ea)
        if s is None:
            s = set()
            for x in idautils.XrefsFrom(ea, 0):
                if x.type not in (ida_xref.fl_F,):  # ordinary flow is not a reference
                    s.add(x.to)
            self.xref_cache[ea] = s
        return s

    # ---- operands ------------------------------------------------------------------------------------
    def mem_ptr(self, b, seg, addr, size):
        addr = fit(b, addr, L.ptrsize)
        segname = h.get_mreg_name(int(seg.r), 2) if seg is not None and seg.t == h.mop_r else ""
        if segname in ("fs", "gs"):
            vfs = self.m.globals.get("virtual_fs")
            if vfs is None:
                t = ir.ArrayType(L.i8, 65536)
                vfs = ir.GlobalVariable(self.m, t, "virtual_fs")
                vfs.initializer = t(None)
            self.approx.add("virtual_fs")
            base = b.ptrtoint(vfs, ir.IntType(L.ptrsize))
            addr = b.add(base, b.and_(addr, ir.Constant(addr.type, 0xFFFF)))
        return b.inttoptr(addr, ir.IntType(size * 8).as_pointer())

    def global_ptr(self, b, g_ea, size):
        p = L._pointer_constant(self.m, L.i8ptr, g_ea)
        return b.bitcast(p, ir.IntType(size * 8).as_pointer())

    def read(self, b, m, ins):
        t, size = m.t, m.size
        if t == h.mop_r:
            return self.regs.read(b, int(m.r), size)
        if t == h.mop_n:
            v = int(m.nnn.value) & ((1 << (size * 8)) - 1)
            if size * 8 == L.ptrsize and v and v in self.xrefs_from(ins.ea):
                return self.sym_int(b, v, size * 8)
            return ir.Constant(ir.IntType(size * 8), v)
        if t == h.mop_v:
            return b.load(self.global_ptr(b, int(m.g), size))
        if t == h.mop_a:
            if m.a is not None and m.a.t == h.mop_v:
                return self.sym_int(b, int(m.a.g), size * 8)
            if m.a is not None and m.a.t == h.mop_r:
                g, off, _ = self.regs._slot(int(m.a.r), m.a.size)
                self.approx.add("address_of_register")
                v = b.ptrtoint(g, ir.IntType(size * 8))
                return b.add(v, ir.Constant(v.type, off)) if off else v
            raise Reject("mop_a_target", m.dstr())
        if t == h.mop_h:
            return self.opaque_value(b, m.helper, [], size)
        if t == h.mop_d:
            return self.insn_value(b, m.d)
        if t == h.mop_p:
            lo = self.read(b, m.pair.lop, ins)
            hi = self.read(b, m.pair.hop, ins)
            lb = m.pair.lop.size * 8
            wt = ir.IntType(size * 8)
            return b.or_(b.zext(lo, wt) if lo.type != wt else lo, b.shl(fit(b, hi, size * 8), ir.Constant(wt, lb)))
        if t == h.mop_fn:
            ft = _fp_type(size)
            if ft is None:  # x87 extended (10/12/16 bytes): the 80-bit pattern as an integer
                w = list(m.fpc.fnum)
                mant = 0
                for i, x in enumerate(w[:-1]):
                    mant |= int(x) << (16 * i)
                bits = (int(w[-1]) << 64) | (mant >> (16 * (len(w) - 1) - 64))
                return ir.Constant(ir.IntType(size * 8), bits & ((1 << (size * 8)) - 1))
            return b.bitcast(ir.Constant(ft, _fnum_value(list(m.fpc.fnum))), ir.IntType(size * 8))
        raise Reject(f"read_mop_{t}", m.dstr())

    def write(self, b, m, val, ins):
        t, size = m.t, m.size
        if t == h.mop_r:
            self.regs.write(b, int(m.r), size, val)
        elif t == h.mop_v:
            b.store(fit(b, val, size * 8), self.global_ptr(b, int(m.g), size))
        elif t == h.mop_p:
            val = fit(b, val, size * 8)
            lb = m.pair.lop.size * 8
            self.write(b, m.pair.lop, b.trunc(val, ir.IntType(lb)), ins)
            hi = b.lshr(val, ir.Constant(val.type, lb))
            self.write(b, m.pair.hop, b.trunc(hi, ir.IntType(m.pair.hop.size * 8)), ins)
        elif t == h.mop_z:
            pass
        else:
            raise Reject(f"write_mop_{t}", m.dstr())

    # ---- instructions --------------------------------------------------------------------------------
    def fp(self, b, v, size):
        ft = _fp_type(size)
        if ft is None:
            raise Reject(f"fp_size_{size}", "")
        return b.bitcast(fit(b, v, size * 8), ft)

    def opaque_value(self, b, name, args, rsize):
        """Call an opaque external `__mc_helper_<name>` on integer args; returns iN (or None)."""
        self.approx.add(f"opaque_helper:{name}")
        rt = ir.IntType(rsize * 8) if rsize > 0 else ir.VoidType()
        hname = "__mc_helper_" + L._sanitize_symbol_name(name)
        ft = ir.FunctionType(rt, [a.type for a in args])
        fn = self.m.globals.get(hname)
        if fn is None:
            fn = ir.Function(self.m, ft, hname)
        elif fn.ftype != ft:
            fn = b.bitcast(fn, ft.as_pointer())
        r = b.call(fn, args)
        return r if rsize > 0 else None

    def fp_opaque(self, b, ins):
        """FP op on a width LLVM/llvmlite cannot express here (x87 10/12-byte, 16-byte): opaque integer helper."""
        ops = [m for m in (ins.l, ins.r) if m.t != h.mop_z and m.size > 0]
        args = [fit(b, self.read(b, m, ins), m.size * 8) for m in ops]
        name = f"fp_{_OPNAME.get(ins.opcode, ins.opcode)}_" + \
            "_".join(str(m.size) for m in ops) + f"_{ins.d.size}"
        self.approx.add("fp_wide_opaque")
        return self.opaque_value(b, name, args, ins.d.size)

    def call_value(self, b, ins):
        """m_call / m_icall; returns the result value (or None)."""
        if ins.opcode == h.m_icall:
            target = fit(b, self.read(b, ins.r, ins), L.ptrsize)
            b.call(b.inttoptr(target, VOID_FT.as_pointer()), [])
            return None
        callee = ins.l
        if callee.t == h.mop_h:
            name = callee.helper
            args = []
            if ins.d.t == h.mop_f:
                for a in ins.d.f.args:
                    if a.t == h.mop_z or a.size <= 0:
                        continue
                    args.append(fit(b, self.read(b, a, ins), a.size * 8))
            rsize = ins.d.size if ins.d.t == h.mop_f else 0
            if name in _LLVM_HELPERS and len(args) == 2 and rsize > 0:
                iname, _ = _LLVM_HELPERS[name]
                w = rsize * 8
                fn = self.m.declare_intrinsic(iname, fnty=ir.FunctionType(ir.IntType(w), [ir.IntType(w)] * 3))
                x = fit(b, args[0], w)
                return b.call(fn, [x, x, fit(b, args[1], w)])
            if name == "__halt":
                fn = L.lift_intrinsic_function(self.m, "__halt")
                b.call(fn, [])
                return None
            return self.opaque_value(b, name, args, rsize)
        if callee.t == h.mop_v:
            fn = self.declare(int(callee.g))
            if fn is None:
                b.call(b.inttoptr(ir.Constant(ir.IntType(L.ptrsize), int(callee.g)), VOID_FT.as_pointer()), [])
            else:
                b.call(self.callee_ptr(b, fn), [])
            return None
        target = fit(b, self.read(b, callee, ins), L.ptrsize)
        b.call(b.inttoptr(target, VOID_FT.as_pointer()), [])
        return None

    def insn_value(self, b, ins):
        """Value of an instruction used as a nested operand (or computed for its destination)."""
        op = ins.opcode
        dsize = ins.d.size
        if op == h.m_mov:
            return self.read(b, ins.l, ins)
        if op == h.m_ldx:
            return b.load(self.mem_ptr(b, ins.l, self.read(b, ins.r, ins), dsize))
        if op in _BIN:
            w = dsize * 8
            x = fit(b, self.read(b, ins.l, ins), w)
            y = fit(b, self.read(b, ins.r, ins), w)
            return getattr(b, _BIN[op])(x, y)
        if op == h.m_neg:
            return b.neg(fit(b, self.read(b, ins.l, ins), dsize * 8))
        if op == h.m_bnot:
            return b.not_(fit(b, self.read(b, ins.l, ins), dsize * 8))
        if op == h.m_lnot:
            v = self.read(b, ins.l, ins)
            return b.zext(b.icmp_unsigned("==", v, ir.Constant(v.type, 0)), ir.IntType(dsize * 8))
        if op == h.m_xds:
            v = self.read(b, ins.l, ins)
            return b.sext(v, ir.IntType(dsize * 8)) if v.type.width < dsize * 8 else fit(b, v, dsize * 8)
        if op in (h.m_xdu, h.m_low):
            return fit(b, self.read(b, ins.l, ins), dsize * 8)
        if op == h.m_high:
            v = self.read(b, ins.l, ins)
            return b.trunc(b.lshr(v, ir.Constant(v.type, v.type.width - dsize * 8)), ir.IntType(dsize * 8))
        if op in _COND:
            kind, signed = _COND[op]
            x = self.read(b, ins.l, ins)
            y = fit(b, self.read(b, ins.r, ins), x.type.width)
            c = (b.icmp_signed if signed else b.icmp_unsigned)(kind, x, y)
            return b.zext(c, ir.IntType(dsize * 8))
        if op == h.m_sets:
            x = self.read(b, ins.l, ins)
            return b.zext(b.icmp_signed("<", x, ir.Constant(x.type, 0)), ir.IntType(dsize * 8))
        if op in (h.m_seto, h.m_ofadd):
            x = self.read(b, ins.l, ins)
            y = fit(b, self.read(b, ins.r, ins), x.type.width)
            if op == h.m_seto:  # signed overflow of x - y
                r = b.sub(x, y)
                o = b.and_(b.xor(x, y), b.xor(x, r))
            else:  # signed overflow of x + y
                r = b.add(x, y)
                o = b.and_(b.xor(x, r), b.xor(y, r))
            return b.zext(b.icmp_signed("<", o, ir.Constant(x.type, 0)), ir.IntType(dsize * 8))
        if op == h.m_cfadd:
            x = self.read(b, ins.l, ins)
            y = fit(b, self.read(b, ins.r, ins), x.type.width)
            return b.zext(b.icmp_unsigned("<", b.add(x, y), x), ir.IntType(dsize * 8))
        if op in (h.m_cfshl, h.m_cfshr):
            x = self.read(b, ins.l, ins)
            n = fit(b, self.read(b, ins.r, ins), x.type.width)
            one = ir.Constant(x.type, 1)
            if op == h.m_cfshl:
                s = b.sub(ir.Constant(x.type, x.type.width), n)
            else:
                s = b.sub(n, one)
            return b.trunc(b.and_(b.lshr(x, s), one), ir.IntType(dsize * 8)) if dsize * 8 < x.type.width \
                else fit(b, b.and_(b.lshr(x, s), one), dsize * 8)
        if op == h.m_setp:
            x = self.read(b, ins.l, ins)
            y = fit(b, self.read(b, ins.r, ins), x.type.width)
            low = b.trunc(b.sub(x, y), L.i8) if x.type.width > 8 else b.sub(x, y)
            ctpop = self.m.declare_intrinsic("llvm.ctpop.i8", fnty=ir.FunctionType(L.i8, [L.i8]))
            even = b.icmp_unsigned("==", b.and_(b.call(ctpop, [low]), ir.Constant(L.i8, 1)), ir.Constant(L.i8, 0))
            return b.zext(even, ir.IntType(dsize * 8))
        if op in (h.m_call, h.m_icall):
            r = self.call_value(b, ins)
            if r is None:
                if dsize > 0:
                    self.approx.add("call_value_unknown")
                    return L._freeze_undef(b, ir.IntType(dsize * 8))
                return None
            return r
        if op in _FP_OPS and any(m.t != h.mop_z and m.size not in (4, 8) for m in (ins.l, ins.r, ins.d)):
            return self.fp_opaque(b, ins)
        if op in _FBIN:
            x = self.fp(b, self.read(b, ins.l, ins), ins.l.size)
            y = self.fp(b, self.read(b, ins.r, ins), ins.r.size)
            return getattr(b, _FBIN[op])(x, y)
        if op == h.m_fneg:
            return b.fneg(self.fp(b, self.read(b, ins.l, ins), ins.l.size))
        if op == h.m_f2f:
            x = self.fp(b, self.read(b, ins.l, ins), ins.l.size)
            ft = _fp_type(dsize)
            if ft is None:
                raise Reject(f"fp_size_{dsize}", "")
            if ins.l.size == dsize:
                return x
            return b.fpext(x, ft) if dsize > ins.l.size else b.fptrunc(x, ft)
        if op in (h.m_i2f, h.m_u2f):
            ft = _fp_type(dsize)
            if ft is None:
                raise Reject(f"fp_size_{dsize}", "")
            v = self.read(b, ins.l, ins)
            return (b.sitofp if op == h.m_i2f else b.uitofp)(v, ft)
        if op in (h.m_f2i, h.m_f2u):
            x = self.fp(b, self.read(b, ins.l, ins), ins.l.size)
            return (b.fptosi if op == h.m_f2i else b.fptoui)(x, ir.IntType(dsize * 8))
        raise Reject(f"opcode_{op}", ins.dstr())

    # ---- control flow --------------------------------------------------------------------------------
    def target_block(self, m):
        if m.t == h.mop_b:
            return self.blocks[m.b]
        if m.t == h.mop_v:
            return self.block_at.get(int(m.g))
        return None

    def jump(self, b, m, ins):
        """Unconditional transfer to operand m (goto)."""
        blk = self.target_block(m)
        if blk is not None:
            b.branch(blk)
            return
        if m.t == h.mop_v:
            fn = self.declare(int(m.g))
            f = L._get_func_cached(int(m.g))
            if fn is not None and f is not None and f.start_ea == int(m.g):
                self.approx.add("tail_jump")
                b.call(self.callee_ptr(b, fn), [])
                b.ret_void()
                return
            # into the middle of other code (shared tails, e.g. __aeabi_uidivmod -> __aeabi_uidiv body):
            # indirect tail jump to <containing function> + offset, or to the raw address
            tgt = int(m.g)
            if f is not None and f.start_ea != self.ea:
                cf = self.declare(f.start_ea)
                base = b.ptrtoint(cf, ir.IntType(L.ptrsize)) if cf is not None else None
                addr = b.add(base, ir.Constant(ir.IntType(L.ptrsize), tgt - f.start_ea)) if base is not None \
                    else ir.Constant(ir.IntType(L.ptrsize), tgt)
            else:
                addr = ir.Constant(ir.IntType(L.ptrsize), tgt)
            self.approx.add("jump_into_other_code")
            b.call(b.inttoptr(addr, VOID_FT.as_pointer()), [])
            b.ret_void()
            return
        self.indirect_jump(b, self.read(b, m, ins), ins)

    def indirect_jump(self, b, target, ins):
        target = fit(b, target, L.ptrsize)
        si = ida_nalt.switch_info_t()
        if ida_nalt.get_switch_info(si, ins.ea):
            cases = ida_xref.calc_switch_cases(ins.ea, si)
            dests = sorted({int(t) for t in cases.targets}) if cases is not None else []
            blocks = [(t, self.block_at.get(t)) for t in dests]
            if blocks and all(blk is not None for _, blk in blocks):
                unreach = self.func.append_basic_block(name=f"switch_default_{ins.ea:x}")
                with b.goto_block(unreach):
                    b.unreachable()
                sw = b.switch(target, unreach)
                for t, blk in blocks:
                    sw.add_case(self.sym_int_const(t), blk)
                return
        self.approx.add("indirect_jump_as_tailcall")
        b.call(b.inttoptr(target, VOID_FT.as_pointer()), [])
        b.ret_void()

    def sym_int_const(self, ea):
        return ir.Constant(ir.IntType(L.ptrsize), ea)

    def lift_insn(self, b, ins, next_blk):
        op = ins.opcode
        if op == h.m_nop:
            return
        if op == h.m_stx:
            v = self.read(b, ins.l, ins)
            b.store(fit(b, v, ins.l.size * 8), self.mem_ptr(b, ins.r, self.read(b, ins.d, ins), ins.l.size))
            return
        if op == h.m_und:
            if ins.d.t == h.mop_r:
                self.regs.write(b, int(ins.d.r), ins.d.size, L._freeze_undef(b, ir.IntType(ins.d.size * 8)))
            return
        if op == h.m_ext:
            self.approx.add("hexrays_ext_insn")
            b.call(self._opaque("__mc_ext_insn"), [])
            if ins.d.t == h.mop_r and ins.d.size > 0:
                self.regs.write(b, int(ins.d.r), ins.d.size, L._freeze_undef(b, ir.IntType(ins.d.size * 8)))
            return
        if op == h.m_push:
            sp, w = self.sp_reg()
            nsp = b.sub(self.regs.read(b, sp, w), ir.Constant(ir.IntType(w * 8), ins.l.size))
            self.regs.write(b, sp, w, nsp)
            v = self.read(b, ins.l, ins)
            b.store(fit(b, v, ins.l.size * 8), b.inttoptr(fit(b, nsp, L.ptrsize), ir.IntType(ins.l.size * 8).as_pointer()))
            return
        if op == h.m_pop:
            sp, w = self.sp_reg()
            cur = self.regs.read(b, sp, w)
            v = b.load(b.inttoptr(fit(b, cur, L.ptrsize), ir.IntType(ins.d.size * 8).as_pointer()))
            self.regs.write(b, sp, w, b.add(cur, ir.Constant(ir.IntType(w * 8), ins.d.size)))
            self.write(b, ins.d, v, ins)
            return
        if op == h.m_ret:
            b.ret_void()
            return
        if op == h.m_goto:
            self.jump(b, ins.l, ins)
            return
        if op == h.m_ijmp:
            if ida_idp.is_ret_insn(ins.ea):
                b.ret_void()
                return
            tgt = ins.d if ins.d.t != h.mop_z else ins.r
            self.indirect_jump(b, self.read(b, tgt, ins), ins)
            return
        if op == h.m_jcnd or op in _JCC:
            if op == h.m_jcnd:
                v = self.read(b, ins.l, ins)
                c = b.icmp_unsigned("!=", v, ir.Constant(v.type, 0))
            else:
                kind, signed = _COND[op]
                x = self.read(b, ins.l, ins)
                y = fit(b, self.read(b, ins.r, ins), x.type.width)
                c = (b.icmp_signed if signed else b.icmp_unsigned)(kind, x, y)
            if next_blk is None:
                raise Reject("conditional_without_fallthrough", ins.dstr())
            blk = self.target_block(ins.d)
            if blk is None:
                taken = self.func.append_basic_block(name=f"far_{ins.ea:x}")
                b.cbranch(c, taken, next_blk)
                with b.goto_block(taken):
                    self.jump(b, ins.d, ins)
                return
            b.cbranch(c, blk, next_blk)
            return
        if op == h.m_jtbl:
            v = self.read(b, ins.l, ins)
            cases = ins.r.c
            default = None
            pairs = []
            for vals, tgt in zip(cases.values, cases.targets):
                vals = list(vals)
                if not vals:
                    default = self.blocks[tgt]
                for x in vals:
                    pairs.append((int(x) & ((1 << v.type.width) - 1), self.blocks[tgt]))
            if default is None:
                default = self.func.append_basic_block(name=f"jtbl_default_{ins.ea:x}")
                with b.goto_block(default):
                    b.unreachable()
            sw = b.switch(v, default)
            seen = set()
            for x, blk in pairs:
                if x not in seen:
                    seen.add(x)
                    sw.add_case(ir.Constant(v.type, x), blk)
            return
        if op in (h.m_call, h.m_icall):
            r = self.call_value(b, ins)
            if ins.d.t not in (h.mop_z, h.mop_f) and r is not None:
                self.write(b, ins.d, r, ins)
            return
        val = self.insn_value(b, ins)
        if val is not None:
            self.write(b, ins.d, val, ins)

    def _opaque(self, name):
        fn = self.m.globals.get(name)
        if fn is None:
            fn = ir.Function(self.m, VOID_FT, name)
        return fn

    def sp_reg(self):
        name = "rsp" if L.ptrsize == 64 else "esp"
        reg = ida_idp.str2reg(name)
        return int(h.reg2mreg(reg)), L.ptrsize // 8

    def translate(self):
        name = L._get_name_cached(self.ea) or f"data_{self.ea:x}"
        fn = self.m.globals.get(name)
        if fn is None:
            fn = ir.Function(self.m, VOID_FT, name)
        elif not isinstance(fn, ir.Function):
            raise Reject("name_taken_by_global", name)
        elif not fn.is_declaration or getattr(fn, "_streamed_definition", False):
            raise Reject("already_defined", name)
        elif fn.ftype != VOID_FT:
            L._retype_function(fn, VOID_FT)
        self.func = fn
        mba = self.mba
        self.blocks = [fn.append_basic_block(name=f"@{i}") for i in range(mba.qty)]
        self.block_at = {}
        for i in range(mba.qty):  # first non-empty block wins (blk 0 is an empty entry stub)
            blk = mba.get_mblock(i)
            if blk.head is not None:
                self.block_at.setdefault(int(blk.start), self.blocks[i])
        for i in range(1, mba.qty):  # blocks emptied at generation (e.g. a lone nop) fall through
            self.block_at.setdefault(int(mba.get_mblock(i).start), self.blocks[i])
        b = ir.IRBuilder(self.blocks[0])
        for i, llblk in enumerate(self.blocks):
            nxt = self.blocks[i + 1] if i + 1 < len(self.blocks) else None
            b.position_at_end(llblk)
            ins = mba.get_mblock(i).head
            while ins is not None:
                self.lift_insn(b, ins, nxt)
                if llblk.is_terminated:
                    break
                ins = ins.next
            if not b.block.is_terminated:
                if nxt is not None:
                    b.branch(nxt)
                else:
                    b.ret_void()
            # cbranch helpers may have appended blocks; keep builder state consistent
        for blk in fn.blocks:
            if not blk.is_terminated:
                bb = ir.IRBuilder(blk)  # safety net
                bb.position_at_end(blk)
                bb.unreachable()
        return fn


def run_lift_generated(input_binary: str, output_llvm_ir: str, only_fvas=None) -> dict:
    """Same artifact contract as ida2llvm.run_lift (tmp + llvm-as verify + atomic publish + sidecar)."""
    L._reset_lift_state()
    started = time.time()
    out = os.path.abspath(output_llvm_ir)
    sidecar = out + L._LiftArtifactState.LIFT_SUFFIX
    L._LiftArtifactState.cleanup_stale_tmp(out)
    L._LiftArtifactState.cleanup_stale_tmp(sidecar)
    tmp = L._LiftArtifactState.tmp_path_for(out)
    result = {"schema": L._LiftArtifactState.SCHEMA, "stage": "lift", "status": "error", "mode": MODE,
              "lifter_id": LIFTER_ID, "maturity": "MMAT_GENERATED", "hooks": "none",
              "ida_install_id": L._LiftArtifactState.ida_install_id(os.environ.get("IDADIR")),
              "input": L._input_info(input_binary), "output_path": out}
    records, reasons, approx_total = [], Counter(), Counter()
    stats = Counter()
    database_open = False
    ctl = None
    try:
        idapro.open_database(input_binary, True)
        database_open = True
        ida_auto.auto_wait()
        h.init_hexrays_plugin()
        L._init_target_state()
        result["ida_version"] = L._ida_version()
        result["source"] = L._source_info()
        ctl = L.BIN2LLVMController(target_mode="source")
        result["target"] = {"triple": str(ctl.m.triple), "datalayout": str(ctl.m.data_layout)}
        regs = RegisterFile(ctl.m)
        _orig_declare = L._declare_function_at

        def _declare_void(module, ea):  # data initializers referencing functions: void() like all others
            ea = L._resolve_thunk(ea)
            name = L._get_name_cached(ea) or f"data_{ea:x}"
            g = module.globals.get(name)
            if g is None:
                g = ir.Function(module, VOID_FT, name)
            return g if isinstance(g, ir.Function) else None
        L._declare_function_at = _declare_void
        try:
            ctl.initialize()
            func_eas = list(idautils.Functions())
            t0 = time.time()
            for ea in func_eas:  # pass 1: register byte ranges over the whole binary
                f = ida_funcs.get_func(ea)
                if f is None or L._is_extern_segment(ea):
                    continue
                mba, _ = gen_mba(f)
                if mba is not None:
                    regs.collect(mba)
                    del mba
            regs.build()
            result["register_pass_s"] = round(time.time() - t0, 3)
            header = [f"ida2llvm_generated lifter={LIFTER_ID} mode={MODE} maturity=MMAT_GENERATED hooks=none "
                      f"ida={result['ida_version']}",
                      f"source proc={result['source'].get('proc')} bits={result['source']['bits']} "
                      f"endian={result['source']['endian']}"]
            if only_fvas is not None:
                # lift only the requested functions (test-pool rows); every other function stays a declaration.
                # Register clusters above are computed over ALL functions, so bodies do not depend on the subset.
                fset = set(func_eas)
                result["requested_fvas"] = len(only_fvas)
                result["requested_not_function_start"] = sorted(hex(x) for x in only_fvas if x not in fset)[:200]
                func_eas = [ea for ea in func_eas if ea in only_fvas]
                header.append(f"subset=requested functions only ({len(func_eas)} of {len(fset)})")
            ctl.begin_stream_to_file(tmp, header)
            for ea in func_eas:
                rec = {"ea": hex(ea), "name": L._get_name_cached(ea), "segment": L._segment_name(ea)}
                try:
                    f = ida_funcs.get_func(ea)
                    if L._is_extern_segment(ea):
                        rec["status"] = "extern"
                        continue
                    mba, hf = gen_mba(f)
                    if mba is None:
                        stats["no_microcode"] += 1
                        rec.update(status="no_microcode", reason=f"{hf.code}:{hf.str}"[:200])
                        continue
                    tr = FunctionTranslator(ctl, regs, ea, mba)
                    fn = tr.translate()
                    ctl._stream_function_if_enabled(fn)
                    stats["defined"] += 1
                    rec.update(status="lifted", llvm_name=fn.name)
                    if tr.approx:
                        rec["approx"] = sorted(tr.approx)
                        approx_total.update(tr.approx)
                except Exception as e:
                    stats["failed"] += 1
                    reason = e.reason if isinstance(e, L.UnsupportedMicrocode) else type(e).__name__
                    reasons[reason] += 1
                    rec.update(status="rejected", reason=reason, detail=str(e)[:200])
                    name = rec["name"] or f"data_{ea:x}"
                    g = ctl.m.globals.get(name)
                    if isinstance(g, ir.Function) and not getattr(g, "_streamed_definition", False):
                        L._release_function_body(g)
                        if g.ftype != VOID_FT:
                            L._retype_function(g, VOID_FT)
                finally:
                    records.append(rec)
            ctl.finish_stream_to_file(fsync=True)
        finally:
            L._declare_function_at = _orig_declare
        ctl = None
        idapro.close_database(False)
        database_open = False
        L._reset_lift_state()
        gc.collect()
        if stats["defined"] == 0:
            result["status"] = "no_functions_lifted"
        else:
            ok, message = L._verify_llvm_file(tmp)
            if not ok:
                result["status"] = "verify_failed"
                result["error"] = message
            else:
                result["output"] = {"path": out, "size": os.path.getsize(tmp),
                                    "sha256": L._LiftArtifactState.sha256_file(tmp)}
                result["status"] = "ok"
                os.replace(tmp, out)
                L._LiftArtifactState.fsync_dir(os.path.dirname(out))
    except Exception as e:
        import traceback
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"[:1000]
        result["traceback"] = traceback.format_exc()[-3000:]
    finally:
        if ctl is not None:
            with suppress(Exception):
                ctl.close_stream()
        with suppress(FileNotFoundError):
            os.remove(tmp)
        if database_open:
            with suppress(Exception):
                idapro.close_database(False)
        result.update(elapsed_s=round(time.time() - started, 3), counts=dict(stats),
                      unsupported_reasons=dict(sorted(reasons.items())),
                      approx_reasons=dict(sorted(approx_total.items())), functions=records)
        L._LiftArtifactState.write_json_atomic(sidecar, result)
    return result


app = typer.Typer(add_completion=False)


@app.command()
def main(file: str = typer.Option(..., "-f", "--file"), output: str = typer.Option(..., "-o", "--output"),
         functions: str = typer.Option(None, "--functions", help="file with one function start address (hex) per line")):
    logging.basicConfig(level=logging.WARNING)
    only = None
    if functions:
        only = {int(x.strip(), 16) for x in open(functions) if x.strip()}
    r = run_lift_generated(file, output, only)
    print(json.dumps({k: r.get(k) for k in ("status", "counts", "unsupported_reasons", "elapsed_s", "error")}))
    raise typer.Exit(L._exit_code(r))


if __name__ == "__main__":
    app()
