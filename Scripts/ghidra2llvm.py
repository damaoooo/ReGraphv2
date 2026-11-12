#!/usr/bin/env python3
"""
Ghidra P-Code -> LLVM IR lifter (v2), for the Ghidra-frontend replication of the lifting ablation.

Modes
- high: Ghidra decompiler High P-Code (SSA, after Ghidra's own simplification) -- counterpart of the Hex-Rays
        final microcode used by the production lifter (O0 / O3 / Oc tiers).
- raw:  per-instruction raw P-Code, no decompiler simplification -- counterpart of the MMAT_GENERATED (MG) tier.

Rules (user, 2026-11-12): never emit undef/poison and never drop part of a function. Whatever cannot be translated
exactly rejects the WHOLE function (recorded with a reason in the sidecar), like the production lifter.

Memory model (both modes)
- every initialized, non-external memory block becomes one global holding its bytes; pointer-sized slots that Ghidra
  references are replaced by the referenced symbol's address (so GOT entries, vtables, jump tables keep relocations);
  uninitialized blocks (.bss) are zero-initialized globals; data symbols in the EXTERNAL block become external globals;
- data accesses (LOAD/STORE/address-tied ram varnodes) use the block global; code references (calls, function
  pointers) use the function symbol (thunks resolved to their target, as the production lifter does);
- every defined function gets its own stack frame (alloca); address-tied stack varnodes are loads/stores into it, and
  the stack pointer's input value is the frame base.
Functions are named by the pool's IDA name (looked up by entry address) so pool rows map 1:1.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import typer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcode2llvm as P  # noqa: E402  (prototype; reused for the per-op integer/float arithmetic)
from pcode2llvm import LLVMValue, UnsupportedPcode  # noqa: E402

LOGGER = logging.getLogger("ghidra2llvm")
HERE = Path(__file__).resolve()
# covers every file that decides the output: driver + high emitter, raw emitter, dumper, reused per-op arithmetic
LIFTER_ID = hashlib.sha256(HERE.read_bytes() + (HERE.parent / "ghidra2llvm_raw.py").read_bytes()
                           + (HERE.parent / "ghidra" / "PcodeDumpV2.java").read_bytes()
                           + (HERE.parent / "pcode2llvm.py").read_bytes()).hexdigest()[:16]

# same triple / data layout as the production lifter (ida2llvm.SOURCE_TARGETS), keyed by (family, bits, endian)
SOURCE_TARGETS = {
    ("x86", 64, "little"): ("x86_64-unknown-linux-gnu",
                            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"),
    ("x86", 32, "little"): ("i386-unknown-linux-gnu",
                            "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:32-n8:16:32-S128"),
    ("arm", 32, "little"): ("arm-unknown-linux-gnueabi", "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"),
    ("arm", 64, "little"): ("aarch64-unknown-linux-gnu", "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"),
    ("mips", 32, "big"): ("mips-unknown-linux-gnu", "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"),
    ("mips", 32, "little"): ("mipsel-unknown-linux-gnu", "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"),
    ("mips", 64, "big"): ("mips64-unknown-linux-gnuabi64", "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128"),
    ("ppc", 32, "big"): ("powerpc-unknown-linux-gnu", "E-m:e-p:32:32-Fn32-i64:64-n32"),
    ("ppc", 64, "big"): ("powerpc64-unknown-linux-gnu", "E-m:e-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"),
}


def _family(language: str) -> str:
    l = language.lower()
    for key, fam in (("x86", "x86"), ("aarch64", "arm"), ("arm", "arm"), ("mips", "mips"), ("powerpc", "ppc")):
        if l.startswith(key):
            return fam
    raise UnsupportedPcode(f"unsupported Ghidra language {language}")


def _q(name: str) -> str:
    """LLVM global identifier, always quoted."""
    return '@"' + name.replace("\\", "\\5C").replace('"', "\\22") + '"'


def _bytes_literal(data: bytes) -> str:
    out = []
    for c in data:
        if 32 <= c < 127 and c not in (34, 92):
            out.append(chr(c))
        else:
            out.append("\\%02X" % c)
    return 'c"' + "".join(out) + '"'


class Program:
    """Program-level facts from the PcodeDumpV2 header record, plus symbol resolution."""

    def __init__(self, rec: Dict[str, Any], names_by_entry: Dict[int, str]):
        self.rec = rec
        self.ptr_bytes = int(rec["pointer_size"])
        self.ptr_bits = self.ptr_bytes * 8
        self.big_endian = bool(rec["big_endian"])
        fam = _family(rec["language"])
        self.family = fam
        key = (fam, self.ptr_bits, "big" if self.big_endian else "little")
        if key not in SOURCE_TARGETS:
            raise UnsupportedPcode(f"no target table entry for {key}")
        self.triple, self.datalayout = SOURCE_TARGETS[key]
        sp = rec.get("stack_pointer")
        self.sp = (int(sp["offset"]), int(sp["size"])) if sp else None
        self.blocks = []
        for b in rec["blocks"]:
            if b["space"] != "ram" and not b["external"]:
                continue
            self.blocks.append(dict(b, start=int(b["start"]), size=int(b["size"])))
        self.blocks.sort(key=lambda b: b["start"])
        # functions
        self.functions: Dict[int, Dict[str, Any]] = {}
        self.external_functions: Dict[int, Dict[str, Any]] = {}
        for f in rec["functions"]:
            e = int(f["entry_offset"])
            (self.external_functions if f["external"] else self.functions)[e] = f
        self.names_by_entry = names_by_entry  # pool names (IDA)
        self.data_symbols: Dict[int, str] = {}
        # external data symbols live in two address spaces that must never be mixed with ram constants:
        # ext_space_data is keyed by EXTERNAL-space offsets (1, 2, 3, ...; only for EXTERNAL data refs),
        # ext_data_symbols by ram addresses inside the loader's EXTERNAL block
        self.ext_space_data: Dict[int, str] = {}
        self.ext_data_symbols: Dict[int, str] = {}
        for addr, name, space in rec.get("data_symbols", []):
            if space == "EXTERNAL":
                self.ext_space_data.setdefault(int(addr), name)
            elif self._is_external_block(int(addr)):
                self.ext_data_symbols.setdefault(int(addr), name)
            else:
                self.data_symbols.setdefault(int(addr), name)
        # every reference Ghidra reports from a source address (one slot can carry several, e.g. ELF header fields)
        self.data_refs: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for frm, to, space in rec.get("data_refs", []):
            if (int(to), space) not in self.data_refs[int(frm)]:
                self.data_refs[int(frm)].append((int(to), space))
        self.used_globals: Dict[str, str] = {}  # name -> definition line (emitted at the end)
        # code = executable blocks that contain function entries; other executable blocks (ELF headers, .dynstr,
        # .rodata sharing the R+X segment) are data
        self.code_blocks = set()
        for e in self.functions:
            b = self.block_of(e)
            if b is not None:
                self.code_blocks.add(b["start"])
        self.declared_functions: Dict[str, int] = {}

    # ---------------------------------------------------------------- blocks
    def block_of(self, addr: int) -> Optional[Dict[str, Any]]:
        for b in self.blocks:
            if b["start"] <= addr < b["start"] + b["size"]:
                return b
        return None

    def _is_external_block(self, addr: int) -> bool:
        b = self.block_of(addr)
        return bool(b and b["external"])

    @staticmethod
    def block_name(b: Dict[str, Any]) -> str:
        return "__gh_%s_%x" % (re.sub(r"[^A-Za-z0-9_.]", "_", b["name"]), b["start"])

    @staticmethod
    def bytes_unavailable(b: Dict[str, Any]) -> bool:
        """Initialized block whose contents were not exported: must never be modelled as zeros."""
        return bool(b.get("bytes_unavailable")) or (bool(b["initialized"]) and b["size"] > 0 and not b["bytes_b64"])

    def block_global(self, b: Dict[str, Any]) -> str:
        if self.bytes_unavailable(b):
            raise UnsupportedPcode(f"block_bytes_unavailable: {b['name']} size={b['size']}")
        name = self.block_name(b)
        if name not in self.used_globals:
            self.used_globals[name] = None  # filled lazily in emit_globals (needs all symbol uses first)
        return name

    # ---------------------------------------------------------------- symbols
    def function_name(self, entry: int) -> str:
        if not hasattr(self, "_unique_names"):
            self._unique_names = {}
            used = set(self.names_by_entry.values())
            for e in sorted(self.functions):
                if e in self.names_by_entry:
                    continue
                n = self._base_name(e)
                if n in used:
                    n = "%s_%x" % (n, e)
                used.add(n)
                self._unique_names[e] = n
        if entry in self.names_by_entry:
            return self.names_by_entry[entry]
        if entry in self._unique_names:
            return self._unique_names[entry]
        return self._base_name(entry)

    def _base_name(self, entry: int) -> str:
        f = self.functions.get(entry) or self.external_functions.get(entry)
        if f is None:
            return "sub_%x" % entry
        name = f["name"]
        for s in f.get("symbols", []):
            if s.startswith("_Z"):
                name = s
                break
        if f.get("thunk") and not f.get("external"):
            name = "." + name  # PLT stub: distinct from the external it jumps to (IDA names it ".name" too)
        return name

    def resolve_call_target(self, entry: int) -> str:
        """Function symbol for a code address that is a function entry; thunks resolved to their target."""
        f = self.functions.get(entry)
        if f is None:
            if entry in self.external_functions:
                return self.function_name(entry)
            raise UnsupportedPcode("call_target_not_function:%x" % entry)
        if f["thunk"] and f["thunk_target"]:
            tgt = int(f["thunk_target"])
            if f["thunk_target_space"] == "EXTERNAL" or tgt in self.external_functions:
                return f["thunk_target_name"]
            return self.function_name(tgt)
        return self.function_name(entry)

    def is_code(self, addr: int) -> bool:
        b = self.block_of(addr)
        return addr in self.functions or (b is not None and b["start"] in self.code_blocks)

    def is_noreturn(self, entry: int) -> bool:
        f = self.functions.get(entry) or self.external_functions.get(entry)
        if f is None:
            return False
        if f.get("noreturn"):
            return True
        if f["thunk"] and f["thunk_target"]:
            t = int(f["thunk_target"])
            g = self.functions.get(t) or self.external_functions.get(t)
            return bool(g and g.get("noreturn"))
        return False

    def declare_function(self, name: str) -> str:
        self.declared_functions.setdefault(name, 0)
        return _q(name)

    def function_containing(self, addr: int) -> Optional[int]:
        best = None
        for e in self.functions:
            if e <= addr and (best is None or e > best):
                best = e
        return best

    def code_address_const(self, addr: int, int_ty: str) -> str:
        """Integer constant expression for a code address (function entry, or entry + offset)."""
        if addr in self.functions:
            sym = self.declare_function(self.resolve_call_target(addr))
            return f"ptrtoint (ptr {sym} to {int_ty})"
        e = self.function_containing(addr)
        if e is None:
            raise UnsupportedPcode("code_address_outside_functions:%x" % addr)
        sym = self.declare_function(self.function_name(e))
        return f"ptrtoint (ptr getelementptr (i8, ptr {sym}, i64 {addr - e}) to {int_ty})"

    def data_ptr_const(self, addr: int) -> str:
        """ptr constant expression for a data address (block global + offset, or external data symbol)."""
        if addr in self.ext_data_symbols:
            name = self.ext_data_symbols[addr]
            self.used_globals.setdefault("__ext__" + name, None)
            return _q(name)
        b = self.block_of(addr)
        if b is None:
            raise UnsupportedPcode("data_address_unmapped:%x" % addr)
        if b["external"]:
            raise UnsupportedPcode("external_data_without_symbol:%x" % addr)
        g = self.block_global(b)
        off = addr - b["start"]
        return _q(g) if off == 0 else f"getelementptr (i8, ptr {_q(g)}, i64 {off})"

    def data_int_const(self, addr: int, int_ty: str) -> str:
        return f"ptrtoint (ptr {self.data_ptr_const(addr)} to {int_ty})"

    ANCHOR_SLACK = 0x10000

    def anchored_int_const(self, addr: int, int_ty: str) -> Optional[str]:
        """Address VALUE outside every block (e.g. MIPS gp = .got + 0x7ff0): preceding block + offset."""
        prev = [b for b in self.blocks if not b["external"] and not self.is_header_block(b) and b["start"] <= addr]
        if not prev:
            return None
        b = max(prev, key=lambda x: x["start"])
        if addr - (b["start"] + b["size"]) > self.ANCHOR_SLACK:
            return None
        g = self.block_global(b)
        return f"ptrtoint (ptr getelementptr (i8, ptr {_q(g)}, i64 {addr - b['start']}) to {int_ty})"

    def address_const(self, addr: int, int_ty: str) -> str:
        """Address used as a value (pointer constant): code -> function symbol, data -> block/external symbol."""
        b = self.block_of(addr)
        if self.is_code(addr):
            return self.code_address_const(addr, int_ty)
        if b is None and addr not in self.ext_data_symbols:
            a = self.anchored_int_const(addr, int_ty)
            if a is None:
                raise UnsupportedPcode("data_address_unmapped:%x" % addr)
            return a
        return self.data_int_const(addr, int_ty)

    @staticmethod
    def is_header_block(b: Dict[str, Any]) -> bool:
        """Loader-made blocks for the ELF header / program headers (PT_LOAD bytes not covered by a section)."""
        return b["name"].startswith("segment_") or b["name"].startswith("_elf")

    def const_symbolizable(self, addr: int) -> bool:
        """May an integer CONSTANT used as a value be turned into a symbol address? A constant is ambiguous (it
        may be a plain number Ghidra typed as a pointer), so NULL and addresses inside the header blocks are kept as
        numbers unless Ghidra has a symbol exactly there; with image base 0 (PIE) small integers fall in those
        blocks. Real memory accesses and data-section pointer slots do not go through this check."""
        if addr == 0:
            return False
        if addr in self.functions:
            return True
        b = self.block_of(addr)
        if b is None:
            return False
        if b["external"]:
            return addr in self.ext_data_symbols
        return not self.is_header_block(b) or addr in self.data_symbols

    def resolvable(self, addr: int) -> bool:
        """ram address (data-section pointer slot target) that can be symbolized. external_functions and
        ext_space_data are keyed by EXTERNAL-space offsets and are deliberately not consulted here."""
        if addr in self.functions or addr in self.ext_data_symbols:
            return True
        b = self.block_of(addr)
        return b is not None and not b["external"]

    # ---------------------------------------------------------------- globals
    def emit_globals(self) -> List[str]:
        lines = []
        int_ty = f"i{self.ptr_bits}"
        done = set()
        changed = True
        while changed:  # initializers can pull in more blocks / symbols
            changed = False
            for name in list(self.used_globals):
                if name in done:
                    continue
                done.add(name)
                changed = True
                if name == "__gh_regs__":
                    size = (getattr(self, "raw_reg_end", 16) + 15) // 16 * 16
                    lines.append(f'@"__gh_regs" = internal global [{size} x i8] zeroinitializer, align 16')
                    continue
                if name == "__virtual_fs__":
                    lines.append('@"virtual_fs" = internal global [65536 x i8] zeroinitializer, align 16')
                    continue
                if name.startswith("__ext__"):
                    lines.append(f"{_q(name[7:])} = external global [0 x i8]")
                    continue
                b = next(x for x in self.blocks if self.block_name(x) == name)
                lines.append(self._block_definition(name, b, int_ty))
        return lines

    def _block_definition(self, name: str, b: Dict[str, Any], int_ty: str) -> str:
        size = b["size"]
        const = "constant" if not b["write"] else "global"
        if self.bytes_unavailable(b):  # only reachable via an initializer; fail the module, never zero-fill
            raise RuntimeError(f"block_bytes_unavailable: {b['name']} size={size}")
        if not b["initialized"]:
            return f"{_q(name)} = internal global [{size} x i8] zeroinitializer, align 16"
        data = base64.b64decode(b["bytes_b64"])
        if self.is_header_block(b):
            # ELF header / program headers: offsets, sizes, p_vaddr ... are numbers, not run-time pointers
            return f"{_q(name)} = internal {const} [{size} x i8] {_bytes_literal(data)}, align 16"
        slots = []
        pb = self.ptr_bytes
        fmt = (">" if self.big_endian else "<") + ("Q" if pb == 8 else "I")
        for frm, refs in sorted(self.data_refs.items()):
            if not (b["start"] <= frm and frm + pb <= b["start"] + size):
                continue
            off = frm - b["start"]
            word = struct.unpack_from(fmt, data, off)[0]
            ext = [to for to, space in refs if space == "EXTERNAL"]
            # the slot bytes disambiguate: only a reference whose target equals the stored value can be a pointer
            to, space = (ext[0], "EXTERNAL") if ext else (word, "ram") if any(t == word for t, _ in refs) \
                else (None, "ram")
            if to is None:
                continue  # relative / non-pointer reference: keep the raw bytes
            if space == "EXTERNAL":
                # Ghidra's loader points GOT slots at its ram EXTERNAL block (thunks there; handled below with the
                # word == to check). A reference into the EXTERNAL *space* never occurred in the pilot, and what
                # the slot bytes mean then (addend? stale?) is unknown, so it fails the module instead of guessing.
                raise RuntimeError("external_space_ref_in_initializer: %s slot %x word=%x -> EXTERNAL:%x"
                                   % (b["name"], frm, word, to))
            else:
                tb = self.block_of(to)
                if tb is not None and tb["external"] and not self.resolvable(to):
                    # the slot holds an address in the loader's EXTERNAL block that names nothing: never keep it
                    raise RuntimeError("external_data_without_symbol in initializer of %s: slot %x -> %x"
                                       % (b["name"], frm, to))
                if not self.resolvable(to):
                    continue  # unmapped target: keep the raw bytes
                try:
                    if self.is_code(to):
                        expr = f"inttoptr ({int_ty} {self.code_address_const(to, int_ty)} to ptr)"
                    else:
                        expr = self.data_ptr_const(to)
                except UnsupportedPcode as e:
                    if str(e).startswith("block_bytes_unavailable"):  # never keep raw bytes for these: fail the module
                        raise RuntimeError(f"initializer of {b['name']} points into {e}") from e
                    continue
            if slots and off < slots[-1][0] + pb:  # two pointer slots sharing bytes: ambiguous, never pick one
                raise RuntimeError("overlapping_pointer_slots in initializer of %s: %x and %x"
                                   % (b["name"], b["start"] + slots[-1][0], frm))
            slots.append((off, expr))
        if not slots:
            return f"{_q(name)} = internal {const} [{size} x i8] {_bytes_literal(data)}, align 16"
        fields, types, pos = [], [], 0
        for off, expr in slots:
            if off > pos:
                types.append(f"[{off - pos} x i8]")
                fields.append(f"[{off - pos} x i8] {_bytes_literal(data[pos:off])}")
            types.append("ptr")
            fields.append(f"ptr {expr}")
            pos = off + pb
        if pos < size:
            types.append(f"[{size - pos} x i8]")
            fields.append(f"[{size - pos} x i8] {_bytes_literal(data[pos:])}")
        return f"{_q(name)} = internal {const} <{{ {', '.join(types)} }}> <{{ {', '.join(fields)} }}>, align 16"


# ======================================================================== High P-Code emitter

MEMORY_SPACES = ("stack", "ram")


class HighEmitter(P.StrictHighPcodeLLVMEmitter):
    """Reuses the prototype's per-op arithmetic; storage, calls, phis, returns and control flow are rewritten."""

    def __init__(self, fn: Dict[str, Any], prog: Program, name: str):
        self.prog = prog
        super().__init__(fn, prog.ptr_bits, name, declarations={}, function_symbols={}, function_signatures={},
                         internal_references={}, strict=True)
        self.entry = int(fn["entry_offset"])
        self.int_ty = f"i{prog.ptr_bits}"
        self.values: Dict[str, LLVMValue] = {}
        self.deferred: List[Tuple[int, str, str, str]] = []
        self.frame_min = 0
        self.frame_size = 0
        self.param_slots: List[Tuple[int, LLVMValue, int]] = []
        self.blocks_by_index = {int(b["index"]): b for b in fn.get("blocks", [])}
        self.edge_fixups: List[Tuple[str, int, Dict[str, Any], Any, Any]] = []
        self.term_line: Dict[int, int] = {}
        self.livein_params: List[str] = []
        self.approx: set = set()
        self.block_done = False

    # ---------------------------------------------------------------- helpers
    def key(self, v: Dict[str, Any]) -> str:
        uid = v.get("uid")
        if uid:
            return "u" + uid
        return P._varnode_key(v)

    def signed_off(self, v: Dict[str, Any]) -> int:
        o = int(v["offset"])
        if o >= 1 << 63:  # Ghidra prints space offsets as unsigned 64-bit
            o -= 1 << 64
        if self.prog.ptr_bits == 32 and o >= 1 << 31:
            o -= 1 << 32
        return o

    def is_memory(self, v: Optional[Dict[str, Any]]) -> bool:
        return bool(v) and v["space"] in MEMORY_SPACES and (v["is_addr_tied"] or v["space"] == "ram")

    def is_sp_input(self, v: Dict[str, Any]) -> bool:
        return (v["space"] == "register" and v.get("is_input") and self.prog.sp is not None
                and int(v["offset"]) == self.prog.sp[0])

    def is_segment_base(self, v: Dict[str, Any]) -> bool:
        return v["space"] == "register" and v.get("is_input") and (v.get("reg") or "") in ("FS_OFFSET", "GS_OFFSET")

    def frame_ptr(self, off: int) -> str:
        return self.new_raw_temp(f"getelementptr i8, ptr %frame, i64 {off - self.frame_min}")

    def mem_ptr(self, v: Dict[str, Any]) -> str:
        if v["space"] == "stack":
            return self.frame_ptr(self.signed_off(v))
        return self.prog.data_ptr_const(int(v["offset"]))

    # ---------------------------------------------------------------- prologue
    def scan_frame(self) -> None:
        lo, hi = 0, self.prog.ptr_bytes
        ops = [op for b in self.function.get("blocks", []) for op in b["ops"]]
        for op in ops:
            for v in [op.get("output")] + op["inputs"]:
                if v and v["space"] == "stack":
                    o = self.signed_off(v)
                    lo, hi = min(lo, o), max(hi, o + int(v["size"]))
            if op["mnemonic"] == "PTRSUB" and len(op["inputs"]) == 2 and self.is_sp_input(op["inputs"][0]) \
                    and op["inputs"][1]["is_constant"]:
                o = self.signed_off(op["inputs"][1])
                lo, hi = min(lo, o), max(hi, o + 1)
        for p in self.function.get("parameters", []):
            for v in p.get("instances", []) + ([p["representative"]] if p.get("representative") else []):
                if v and v["space"] == "stack":
                    o = self.signed_off(v)
                    lo, hi = min(lo, o), max(hi, o + int(v["size"]))
        lo -= lo % 16
        self.frame_min = lo
        self.frame_size = max(16, ((hi - lo + 15) // 16) * 16)

    def emit(self) -> List[str]:
        blocks = sorted(self.function.get("blocks", []), key=lambda b: int(b["index"]))
        if not blocks:
            raise UnsupportedPcode("no_blocks")
        self.scan_frame()
        if blocks[0].get("in") and any(op["mnemonic"] == "MULTIEQUAL" for op in blocks[0]["ops"]):
            raise UnsupportedPcode("entry_block_has_predecessors")
        arg_decls = []
        for i, p in enumerate(self.function.get("parameters", [])):
            ty = self.param_type(p)
            arg = LLVMValue(P._type_width(ty), f"%arg{i}", ty=None if ty.startswith("i") else ty)
            arg_decls.append(f"{ty} %arg{i}")
            insts = list(p.get("instances", []))
            if not insts and p.get("representative"):
                insts = [p["representative"]]
            for v in insts:
                if v["space"] == "stack":
                    self.param_slots.append((self.signed_off(v), arg, int(v["size"])))
                elif v["space"] in ("register", "unique"):
                    self.values[self.key(v)] = arg
        n = len(arg_decls)
        seen = set()
        for b in blocks:
            for op in b["ops"]:
                for v in op["inputs"]:
                    if v["space"] == "stack" and v.get("is_input") and self.signed_off(v) >= 0:
                        # incoming-argument area not in Ghidra's prototype. (Negative offsets are the function's own
                        # locals, e.g. written by a callee through an escaped pointer: plain frame loads.)
                        off, sz = self.signed_off(v), int(v["size"])
                        if any(o == off and z >= sz for o, _, z in self.param_slots) or ("s", off, sz) in seen:
                            continue
                        seen.add(("s", off, sz))
                        arg = LLVMValue(sz * 8, f"%arg{n}")
                        arg_decls.append(f"i{sz * 8} %arg{n}")
                        self.param_slots.append((off, arg, sz))
                        self.livein_params.append("stack%+d" % off)
                        n += 1
                        continue
                    if v["space"] == "register" and v.get("is_input") and not self.is_sp_input(v) \
                            and not self.is_segment_base(v):
                        k = self.key(v)
                        if k in self.values or k in seen:
                            continue
                        seen.add(k)
                        w = int(v["size"]) * 8
                        arg = LLVMValue(w, f"%arg{n}")
                        arg_decls.append(f"i{w} %arg{n}")
                        self.values[k] = arg
                        self.livein_params.append(v.get("reg") or v["offset"])
                        n += 1
        self.lines.append(f"define {self.ret_type} {_q(self.name)}({', '.join(arg_decls)}) {{")
        self.lines.append("entry:")
        self.lines.append(f"  %frame = alloca [{self.frame_size} x i8], align 16")
        self.lines.append(f"  %sp0 = getelementptr i8, ptr %frame, i64 {-self.frame_min}")
        self.lines.append(f"  %sp0i = ptrtoint ptr %sp0 to {self.int_ty}")
        for off, arg, size in self.param_slots:
            ptr = self.frame_ptr(off)
            val = self.cast(arg, size * 8)
            self.lines.append(f"  store i{size * 8} {val.text}, ptr {ptr}, align 1")
        self.lines.append(f"  br label %bb{int(blocks[0]['index'])}")
        for b in blocks:
            self.emit_block(b)
        self.apply_edge_fixups()
        self.lines.append("}")
        text = "\n".join(self.lines)
        if re.search(r"\b(undef|poison)\b", text):
            raise UnsupportedPcode("undef_in_output")
        return self.lines

    def param_type(self, p: Dict[str, Any]) -> str:
        dt = p.get("type") or {}
        size = int(p.get("size") or dt.get("length") or 0)
        if dt.get("kind") == "float" and size in (4, 8):
            return "float" if size == 4 else "double"
        if size <= 0:
            rep = p.get("representative")
            size = int(rep["size"]) if rep else self.prog.ptr_bytes
        return f"i{size * 8}"

    def _return_type(self) -> str:
        dt = self.function.get("return_type") or {}
        if dt.get("kind") == "void" or str(dt.get("name", "")).lower() == "void":
            return "void"
        size = int(dt.get("length") or 0)
        if size <= 0:
            return "void"
        if dt.get("kind") == "float" and size in (4, 8):
            return "float" if size == 4 else "double"
        return f"i{size * 8}"

    # ---------------------------------------------------------------- blocks
    def emit_block(self, block: Dict[str, Any]) -> None:
        self.current_block = block
        bi = int(block["index"])
        self.lines.append(f"bb{bi}:")
        ops = list(block.get("ops", []))
        phis = [op for op in ops if op["mnemonic"] == "MULTIEQUAL"]
        rest = [op for op in ops if op["mnemonic"] != "MULTIEQUAL"]
        for op in phis:
            self.emit_multiequal(op.get("output"), op["inputs"])
        self.block_done = False
        terminated = False
        for k, op in enumerate(rest):
            self.emit_pcode_op(op)
            if self.block_done or op["mnemonic"] in self.TERMINATORS:  # terminator / call to a noreturn function
                terminated = True
                # never drop ops silently; the only ops accepted after a noreturn call are the two the decompiler
                # adds itself: the artificial halt (RETURN const 1) and memory self-copies (COPY ram X -> ram X)
                for later in rest[k + 1:]:
                    if not (self.block_done and self._is_post_noreturn_filler(later)):
                        raise UnsupportedPcode(f"ops_after_terminator: {op['mnemonic']} then {later['mnemonic']}")
                break
        if not terminated:
            outs = [int(x) for x in block.get("out", [])]
            if len(outs) == 1:
                self.lines.append(f"  br label %bb{outs[0]}")
            elif not outs:
                # the decompiler removed the fall-through edge; only accept it after a call (Ghidra decided the
                # callee does not return at this site), otherwise the exit is unknown -> reject
                if rest and rest[-1]["mnemonic"] in ("CALL", "CALLIND"):
                    self.lines.append("  unreachable")
                    self.approx.add("callsite_noreturn")
                else:
                    raise UnsupportedPcode(f"no_successor_without_terminator: last={rest[-1]['mnemonic'] if rest else None}")
            else:
                raise UnsupportedPcode("multi_successor_without_cbranch")
        # terminator = last line, except for a switch whose default block (label + unreachable) follows it
        idx = len(self.lines) - 1
        if self.lines[idx].strip() == "unreachable" and self.lines[idx - 1].startswith("swdefault"):
            idx -= 2
        self.term_line[bi] = idx
        self.current_block = None

    def emit_pcode_op(self, op: Dict[str, Any]) -> None:
        self._op = op
        m = op["mnemonic"]
        if m in ("SEGMENTOP", "CPOOLREF", "NEW", "INSERT", "EXTRACT"):
            raise UnsupportedPcode(f"op_{m}")
        super().emit_pcode_op(op)

    # ---------------------------------------------------------------- storage
    def read(self, v: Optional[Dict[str, Any]], preferred_width: Optional[int] = None,
             signed: bool = False) -> LLVMValue:
        if not v:
            raise UnsupportedPcode("missing_operand")
        width = P._normalize_width(preferred_width or int(v["size"]) * 8)
        if v["is_constant"]:
            val = int(v["offset"])
            ht = v.get("high_type") or {}
            if ht.get("kind") == "pointer" and int(v["size"]) == self.prog.ptr_bytes \
                    and self.prog.const_symbolizable(val):
                return self.cast(LLVMValue(self.prog.ptr_bits, self.prog.address_const(val, self.int_ty)), width)
            return P._const(width, val)
        if self.is_memory(v) or (v["space"] == "stack" and v.get("is_input")):
            sz = int(v["size"]) * 8
            ptr = self.mem_ptr(v)
            loaded = self.new_temp(sz, f"load i{sz}, ptr {ptr}, align 1")
            return self.cast(loaded, width, signed=signed)
        k = self.key(v)
        val = self.values.get(k)
        if val is not None:
            return self.cast(val, width, signed=signed)
        if self.is_sp_input(v):
            return self.cast(LLVMValue(self.prog.ptr_bits, "%sp0i"), width)
        if self.is_segment_base(v):  # %fs/%gs base: thread-local area, modelled like the production lifter
            self.prog.used_globals.setdefault("__virtual_fs__", None)
            return self.cast(LLVMValue(self.prog.ptr_bits, f'ptrtoint (ptr @"virtual_fs" to {self.int_ty})'), width)
        if v.get("is_input"):
            raise UnsupportedPcode("livein_%s:%s" % (v["space"], v.get("reg") or v["offset"]))
        raise UnsupportedPcode("use_before_def:%s" % v["space"])

    def write(self, out: Optional[Dict[str, Any]], value: LLVMValue, signed: bool = False) -> None:
        if not out:
            return
        width = int(out["size"]) * 8
        if self.is_memory(out):
            iv = self.cast(value, width, signed=signed)
            self.lines.append(f"  store i{width} {iv.text}, ptr {self.mem_ptr(out)}, align 1")
            return
        if value.is_integer or value.width != width:
            value = self.cast(value, width, signed=signed)
        self.values[self.key(out)] = value

    def cast(self, value: LLVMValue, width: int, signed: bool = False) -> LLVMValue:
        if value.text in ("undef", "poison"):
            raise UnsupportedPcode("undef_value")
        if value.is_integer and value.const is None and value.text.startswith("ptrtoint ("):
            # constant expression: widen/narrow through an instruction
            if value.width == width:
                return value
            t = self.new_temp(value.width, f"add i{value.width} {value.text}, 0")
            return super().cast(t, width, signed)
        return super().cast(value, width, signed)

    def address_value(self, v: Optional[Dict[str, Any]]) -> LLVMValue:
        if not v:
            raise UnsupportedPcode("missing_address")
        if v["is_constant"]:
            a = int(v["offset"])
            return LLVMValue(self.prog.ptr_bits, self.prog.data_int_const(a, self.int_ty))
        return self.read(v, self.prog.ptr_bits)

    def memory_pointer(self, address: LLVMValue) -> LLVMValue:
        address = self.cast(address, self.prog.ptr_bits)
        return self.new_temp(self.prog.ptr_bits, f"inttoptr {self.int_ty} {address.text} to ptr")

    def emit_load(self, output, inputs) -> None:
        if not output:
            raise UnsupportedPcode("load_without_output")
        width = int(output["size"]) * 8
        ptr = self.memory_pointer(self.address_value(inputs[-1]))
        self.write(output, self.new_temp(width, f"load i{width}, ptr {ptr.text}, align 1"))

    def emit_store(self, inputs) -> None:
        if len(inputs) < 3:
            raise UnsupportedPcode("store_operands")
        w = int(inputs[2]["size"]) * 8
        val = self.read(inputs[2], w)
        ptr = self.memory_pointer(self.address_value(inputs[1]))
        self.lines.append(f"  store i{w} {val.text}, ptr {ptr.text}, align 1")

    def emit_ptrsub(self, output, inputs) -> None:
        width = int(output["size"]) * 8
        a, b = inputs[0], inputs[1]
        if a["is_constant"] and int(a["offset"]) == 0 and b["is_constant"]:
            addr = int(b["offset"])
            # PTRSUB(0, A) evaluates to the integer A. Symbolize only when A lies in the image; otherwise keep the
            # constant as is (e.g. `return 1` in a function Ghidra typed as returning a pointer)
            blk = self.prog.block_of(addr)
            if self.prog.const_symbolizable(addr) or (blk is not None and blk["external"]) or \
                    (blk is None and addr != 0 and self.prog.anchored_int_const(addr, self.int_ty)):
                text = self.prog.address_const(addr, self.int_ty)  # external without symbol -> rejected there
            else:
                text = str(addr)
            self.write(output, self.cast(LLVMValue(self.prog.ptr_bits, text), width))
            return
        if self.is_sp_input(a) and b["is_constant"]:
            off = self.signed_off(b)
            p = self.frame_ptr(off)
            self.write(output, self.new_temp(self.prog.ptr_bits, f"ptrtoint ptr {p} to {self.int_ty}"))
            return
        base = self.read(a, width)
        off = self.read(b, width)
        self.write(output, self.new_temp(width, f"add i{width} {base.text}, {off.text}"))

    # ---------------------------------------------------------------- SSA joins
    def emit_multiequal(self, output, inputs) -> None:
        """Phi for register/unique outputs; for memory outputs only the non-memory inputs need a store on the edge.
        Anything an edge needs (load of a memory input, store, type cast) is inserted at the end of the predecessor."""
        if output is None or self.current_block is None:
            raise UnsupportedPcode("multiequal_shape")
        preds = [int(x) for x in self.current_block.get("in", [])]
        if len(preds) != len(inputs):
            raise UnsupportedPcode("multiequal_arity")
        if self.is_memory(output):
            for pred, v in zip(preds, inputs):
                if self.is_memory(v):
                    if v["space"] != output["space"] or v["offset"] != output["offset"] or v["size"] != output["size"]:
                        raise UnsupportedPcode("memory_phi_storage_mismatch")
                    continue
                self.edge_fixups.append(("store", pred, v, output, None))
            return
        ty = P._varnode_llvm_type(output, self.prog.ptr_bits)
        if ty not in ("float", "double"):
            ty = f"i{int(output['size']) * 8}"
        name = f"%p{self.temp_index}"
        self.temp_index += 1
        incoming = []
        for pred, v in zip(preds, inputs):
            ph = f"__PHI_{len(self.edge_fixups)}__"
            incoming.append(f"[ {ph}, %bb{pred} ]")
            self.edge_fixups.append(("value", pred, v, ty, (len(self.lines), ph)))
        self.lines.append(f"  {name} = phi {ty} {', '.join(incoming)}")
        self.values[self.key(output)] = LLVMValue(P._type_width(ty), name, ty=None if ty.startswith("i") else ty)

    def edge_value_lines(self, v: Dict[str, Any], ty: str) -> Tuple[str, List[str]]:
        """(operand text, instructions to place at the end of the predecessor) for a phi input."""
        saved, self.lines = self.lines, []
        try:
            if v["is_constant"] and ty.startswith("i"):
                val = self.read(v, P._type_width(ty))
            elif v["is_constant"]:
                val = self.cast_to_type(self.read(v, int(v["size"]) * 8), ty)
            else:
                if not self.is_memory(v) and self.key(v) not in self.values and not v.get("is_input"):
                    raise UnsupportedPcode("phi_input_undefined")
                val = self.read_as_type(v, ty) if ty in ("float", "double") else self.read(v, P._type_width(ty))
                if val.llvm_type != ty:
                    val = self.cast_to_type(val, ty)
            return val.text, self.lines
        finally:
            self.lines = saved

    def apply_edge_fixups(self) -> None:
        inserts: Dict[int, List[str]] = defaultdict(list)
        replacements: List[Tuple[int, str, str]] = []
        for kind, pred, v, dst, where in self.edge_fixups:
            if pred not in self.term_line:
                raise UnsupportedPcode("phi_pred_not_emitted")
            if kind == "value":
                text, extra = self.edge_value_lines(v, dst)
                inserts[pred].extend(extra)
                replacements.append((where[0], where[1], text))
            else:  # store a non-memory input into the memory output's storage on this edge
                w = int(dst["size"]) * 8
                saved, self.lines = self.lines, []
                try:
                    val = self.read(v, w)
                    self.lines.append(f"  store i{w} {val.text}, ptr {self.mem_ptr(dst)}, align 1")
                    inserts[pred].extend(self.lines)
                finally:
                    self.lines = saved
        for line_index, ph, text in replacements:
            self.lines[line_index] = self.lines[line_index].replace(ph, text)
        for pred in sorted(inserts, key=lambda b: self.term_line[b], reverse=True):
            at = self.term_line[pred]
            self.lines[at:at] = inserts[pred]

    def emit_indirect(self, output, inputs) -> None:
        if output is None or self.is_memory(output):
            return
        self.write(output, self.read(inputs[0]))

    # ---------------------------------------------------------------- control flow
    @staticmethod
    def _is_halt_return(inputs) -> bool:
        """Ghidra's artificial halt (FlowInfo::artificialHalt): RETURN whose only input is the constant 1."""
        return len(inputs) == 1 and inputs[0]["space"] == "const" and int(inputs[0]["offset"]) == 1

    def _is_post_noreturn_filler(self, op) -> bool:
        m = op["mnemonic"]
        if m == "RETURN":
            return self._is_halt_return(op["inputs"])
        if m == "COPY" and op.get("output") and len(op["inputs"]) == 1:
            o, i = op["output"], op["inputs"][0]
            return (o["space"] == i["space"] == "ram" and int(o["offset"]) == int(i["offset"])
                    and o["size"] == i["size"])
        return False

    def emit_return(self, inputs) -> None:
        if self._is_halt_return(inputs):  # a halt not preceded by a noreturn call: cause unknown -> reject
            raise UnsupportedPcode("halt_return")
        if self.ret_type == "void":
            self.lines.append("  ret void")
            return
        if len(inputs) < 2:
            raise UnsupportedPcode("return_without_value")
        v = self.read_as_type(inputs[1], self.ret_type) if self.ret_type in ("float", "double") else \
            self.read(inputs[1], P._type_width(self.ret_type))
        self.lines.append(f"  ret {self.ret_type} {v.text}")

    def emit_branchind(self, inputs) -> None:
        outs = [int(x) for x in self.current_block.get("out", [])]
        if not outs:
            raise UnsupportedPcode("branchind_without_targets")
        target = self.read(inputs[0], self.prog.ptr_bits)
        selfsym = self.prog.declare_function(self.name)
        rel = self.new_temp(self.prog.ptr_bits, f"sub {self.int_ty} {target.text}, ptrtoint (ptr {selfsym} to {self.int_ty})")
        cases = []
        seen = set()
        for o in outs:
            start = self.blocks_by_index[o].get("start")
            if not start:
                raise UnsupportedPcode("branchind_target_without_address")
            off = int(start) - self.entry
            if off in seen:
                continue
            seen.add(off)
            cases.append(f"{self.int_ty} {off}, label %bb{o}")
        lbl = f"swdefault{self.temp_index}"
        self.temp_index += 1
        self.lines.append(f"  switch {self.int_ty} {rel.text}, label %{lbl} [ {' '.join(cases)} ]")
        self.lines.append(f"{lbl}:")
        self.lines.append("  unreachable")

    # ---------------------------------------------------------------- calls
    def arg_value(self, v: Dict[str, Any]) -> LLVMValue:
        ht = v.get("high_type") or {}
        if ht.get("kind") == "float" and int(v["size"]) in (4, 8):
            return self.read_float(v, int(v["size"]) * 8)
        return self.read(v, int(v["size"]) * 8)

    def emit_call(self, output, inputs, indirect: bool) -> None:
        if not inputs:
            raise UnsupportedPcode("call_without_target")
        args = [self.arg_value(v) for v in inputs[1:]]
        if output is not None and not self.is_memory(output):
            ht = output.get("high_type") or {}
            rty = ("float" if int(output["size"]) == 4 else "double") if ht.get("kind") == "float" and \
                int(output["size"]) in (4, 8) else f"i{int(output['size']) * 8}"
        elif output is not None:
            rty = f"i{int(output['size']) * 8}"
        else:
            rty = "void"
        fnty = f"{rty} ({', '.join(a.llvm_type for a in args)})"
        arg_text = ", ".join(f"{a.llvm_type} {a.text}" for a in args)
        t = inputs[0]
        if indirect:
            tv = self.read(t, self.prog.ptr_bits)
            callee = self.new_raw_temp(f"inttoptr {self.int_ty} {tv.text} to ptr")
        else:
            if t["space"] not in ("ram",) and not t["is_constant"]:
                raise UnsupportedPcode("call_target_space:%s" % t["space"])
            callee = self.prog.declare_function(self.prog.resolve_call_target(int(t["offset"])))
        if rty == "void":
            self.lines.append(f"  call {fnty} {callee}({arg_text})")
        else:
            val = self.new_typed_temp(rty, f"call {fnty} {callee}({arg_text})")
            self.write(output, val)
        if not indirect and self.prog.is_noreturn(int(t["offset"])):
            self.lines.append("  unreachable")
            self.block_done = True

    def emit_callother(self, output, inputs) -> None:
        name = self._op.get("userop") or "userop%s" % (inputs[0]["offset"] if inputs else "")
        args = [self.arg_value(v) for v in inputs[1:]]
        rty = f"i{int(output['size']) * 8}" if output is not None else "void"
        callee = self.prog.declare_function("__gh_userop_" + re.sub(r"[^A-Za-z0-9_]", "_", name))
        fnty = f"{rty} ({', '.join(a.llvm_type for a in args)})"
        arg_text = ", ".join(f"{a.llvm_type} {a.text}" for a in args)
        if rty == "void":
            self.lines.append(f"  call {fnty} {callee}({arg_text})")
        else:
            self.write(output, self.new_typed_temp(rty, f"call {fnty} {callee}({arg_text})"))


# ======================================================================== driver

def run_ghidra(binary: Path, out_jsonl: Path, fva_file: Optional[Path], mode: str, ghidra_home: Path,
               jdk: Optional[Path], timeout: int, work: Path) -> None:
    env = dict(os.environ)
    if jdk:
        env["JAVA_HOME"] = str(jdk)
        env["PATH"] = f"{jdk}/bin:" + env.get("PATH", "")
    work.mkdir(parents=True, exist_ok=True)
    # PIE (ET_DYN) is rebased to 0 like IDA; ET_EXEC keeps its link addresses (IDA does the same)
    with open(binary, "rb") as fh:
        hdr = fh.read(18)
    e_type = int.from_bytes(hdr[16:18], "big" if hdr[5] == 2 else "little")
    rebase = ["-loader-imagebase", "0"] if e_type == 3 else []
    cmd = [str(ghidra_home / "support" / "analyzeHeadless"), str(work), "p", "-import", str(binary),
           *rebase, "-scriptPath", str(HERE.parent / "ghidra"),
           "-postScript", "PcodeDumpV2.java", str(out_jsonl), str(fva_file) if fva_file else "-", str(timeout), mode,
           "-deleteProject", "-max-cpu", "1"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out_jsonl.exists():
        raise RuntimeError("ghidra failed: " + (r.stdout[-2000:] + r.stderr[-2000:]))


def lift(binary: str, output: str, names_tsv: Optional[str], mode: str, ghidra_home: str, jdk: Optional[str],
         timeout: int, keep_dump: bool) -> Dict[str, Any]:
    t0 = time.time()
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    names: Dict[int, str] = {}
    fva_file = None
    tmp = Path(tempfile.mkdtemp(prefix="g2l_", dir=str(out.parent)))
    if names_tsv:
        for line in open(names_tsv):
            line = line.rstrip("\n")
            if not line:
                continue
            fva, name = line.split("\t", 1)
            names[int(fva, 16)] = name
        fva_file = tmp / "fvas.txt"
        fva_file.write_text("\n".join(hex(x) for x in sorted(names)) + "\n")
    dump = out.with_suffix(f".{mode}.jsonl") if keep_dump else tmp / "dump.jsonl"
    result: Dict[str, Any] = {"lifter": "ghidra2llvm", "lifter_id": LIFTER_ID, "mode": mode, "input": binary,
                              "output": str(out), "status": "error"}
    try:
        run_ghidra(Path(binary).resolve(), dump, fva_file, mode, Path(ghidra_home), Path(jdk) if jdk else None,
                   timeout, tmp / "proj")
        lines = open(dump, encoding="utf-8").read().splitlines()
        prog = Program(json.loads(lines[0]), names)
        result["language"] = prog.rec["language"]
        records, bodies, reasons = [], [], Counter()
        defined = set()
        for line in lines[1:]:
            fn = json.loads(line)
            entry = int(fn["entry_offset"])
            name = prog.function_name(entry)
            rec = {"ea": hex(entry), "name": name}
            if fn["kind"] == "missing":
                rec.update(status="missing_in_ghidra")
            elif fn.get("decompile_error") is not None:
                rec.update(status="no_decompile", reason=fn["decompile_error"][:200])
            else:
                try:
                    if mode == "high":
                        em = HighEmitter(fn, prog, name)
                    else:
                        from ghidra2llvm_raw import RawEmitter  # noqa
                        em = RawEmitter(fn, prog, name)
                    body = em.emit()
                    bodies.append("\n".join(body))
                    defined.add(name)
                    rec.update(status="lifted")
                    if getattr(em, "approx", None):
                        rec["approx"] = sorted(em.approx)
                    if getattr(em, "livein_params", None):
                        rec["livein_params"] = em.livein_params
                except UnsupportedPcode as e:
                    msg = str(e)
                    reasons[msg.split(":")[0]] += 1
                    rec.update(status="rejected", reason=msg[:200])
            records.append(rec)
        hdr = [f"; ghidra2llvm lifter={LIFTER_ID} mode={mode} language={prog.rec['language']}",
               f'target datalayout = "{prog.datalayout}"', f'target triple = "{prog.triple}"', ""]
        globs = prog.emit_globals()
        decls = [f"declare void {_q(n)}()" for n in sorted(prog.declared_functions) if n not in defined]
        text = "\n".join(hdr + globs + [""] + decls + [""] + bodies) + "\n"
        tmpout = out.with_suffix(".ll.tmp")
        tmpout.write_text(text, encoding="utf-8")
        r = subprocess.run(["llvm-as-18", str(tmpout), "-o", "/dev/null"], capture_output=True, text=True)
        if r.returncode != 0:
            result.update(status="verify_failed", error=r.stderr[-2000:])
        elif not defined:
            result.update(status="no_functions_lifted")
        else:
            os.replace(tmpout, out)
            result.update(status="ok")
        c = Counter(r["status"] for r in records)
        result.update(counts=dict(c), reasons=dict(reasons.most_common()), functions=records)
    except Exception as e:  # noqa
        import traceback
        result.update(status="error", error=f"{type(e).__name__}: {e}"[:1000], traceback=traceback.format_exc()[-3000:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        result["elapsed_s"] = round(time.time() - t0, 2)
        Path(str(out) + ".lift.json").write_text(json.dumps(result, indent=1))
    return result


app = typer.Typer(add_completion=False)


@app.command()
def main(file: str = typer.Option(..., "-f"), output: str = typer.Option(..., "-o"),
         names: Optional[str] = typer.Option(None, "--names", help="TSV: fva(hex)\\tpool name; only these functions"),
         mode: str = typer.Option("high", "--mode"),
         ghidra_home: str = typer.Option(os.path.expanduser("~/tools/ghidra_12.2_PUBLIC"), "--ghidra-home"),
         jdk: Optional[str] = typer.Option(os.path.expanduser("~/tools/jdk-25.0.5+8"), "--jdk"),
         timeout: int = typer.Option(60, "--timeout"), keep_dump: bool = typer.Option(False, "--keep-dump")):
    logging.basicConfig(level=logging.WARNING)
    r = lift(file, output, names, mode, ghidra_home, jdk, timeout, keep_dump)
    print(json.dumps({k: r.get(k) for k in ("status", "counts", "reasons", "elapsed_s", "error")}))
    raise typer.Exit(0 if r["status"] == "ok" else 1)


if __name__ == "__main__":
    app()
