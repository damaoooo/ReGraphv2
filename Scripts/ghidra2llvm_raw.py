"""
Raw (per-instruction) Ghidra P-Code -> LLVM IR: the Ghidra counterpart of the MMAT_GENERATED (MG) tier.
No decompiler simplification: every machine instruction's SLEIGH P-Code is translated as is.

Model (same as ida2llvm_generated.py):
- registers: one module-level byte array (@__gh_regs) indexed by Ghidra register-space offset; the module data layout
  has the target's endianness, so overlapping views (EAX/RAX, a0 low half on big-endian MIPS) alias correctly;
- unique (SLEIGH temporaries): a per-function frame; distinct unique ranges are packed compactly;
- every function is `void ()`; calls carry no arguments; LOAD/STORE go through the computed address;
- integer constants are symbolized only when the instruction carries a Ghidra reference to that address;
- one LLVM block per machine instruction (+ sub-blocks for P-Code-relative branches); MIPS delay slots: the branch's
  P-Code already contains the delay-slot instruction, so the delay-slot instruction's own P-Code is not emitted.
Whatever cannot be translated exactly rejects the whole function.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pcode2llvm as P
from pcode2llvm import LLVMValue, UnsupportedPcode

import ghidra2llvm as G

VOID = "void ()"


class RawEmitter(P.StrictHighPcodeLLVMEmitter):
    def __init__(self, fn: Dict[str, Any], prog: "G.Program", name: str):
        self.prog = prog
        super().__init__(fn, prog.ptr_bits, name, declarations={}, function_symbols={}, function_signatures={},
                         internal_references={}, strict=True)
        self.entry = int(fn["entry_offset"])
        self.int_ty = f"i{prog.ptr_bits}"
        self.insts = fn.get("instructions", [])
        self.addr_set = {int(i["address"]) for i in self.insts if not i["in_delay_slot"]}
        self.delay_addrs = {int(i["address"]) for i in self.insts if i["in_delay_slot"]}
        self.cur_ins: Optional[Dict[str, Any]] = None
        self.unique_map: Dict[Tuple[int, int], int] = {}
        self.unique_size = 0
        self.term = False

    def _return_type(self) -> str:
        return "void"

    # ---------------------------------------------------------------- layout
    def plan_uniques(self) -> None:
        spans = sorted({(int(v["offset"]), int(v["offset"]) + int(v["size"]))
                        for ins in self.insts for op in ins["pcode"]
                        for v in [op.get("output")] + op["inputs"] if v and v["space"] == "unique"})
        merged: List[List[int]] = []
        for lo, hi in spans:
            if merged and lo < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        pos = 0
        self.unique_base: List[Tuple[int, int, int]] = []
        for lo, hi in merged:
            self.unique_base.append((lo, hi, pos))
            pos += hi - lo
        self.unique_size = max(16, (pos + 15) // 16 * 16)

    def unique_off(self, v: Dict[str, Any]) -> int:
        o = int(v["offset"])
        for lo, hi, base in self.unique_base:
            if lo <= o < hi:
                return base + (o - lo)
        raise UnsupportedPcode("unique_unplanned")

    # ---------------------------------------------------------------- storage
    def loc_ptr(self, v: Dict[str, Any]) -> str:
        sp = v["space"]
        if sp == "register":
            end = int(v["offset"]) + int(v["size"])
            self.prog.raw_reg_end = max(getattr(self.prog, "raw_reg_end", 0), end)
            self.prog.used_globals.setdefault("__gh_regs__", None)
            return self.new_raw_temp(f'getelementptr i8, ptr @"__gh_regs", i64 {int(v["offset"])}')
        if sp == "unique":
            return self.new_raw_temp(f"getelementptr i8, ptr %uniq, i64 {self.unique_off(v)}")
        if sp == "ram":
            return self.prog.data_ptr_const(int(v["offset"]))
        raise UnsupportedPcode("space_%s" % sp)

    def ref_targets(self) -> set:
        return {int(t) for t, _ in (self.cur_ins or {}).get("refs", [])}

    def read(self, v: Optional[Dict[str, Any]], preferred_width: Optional[int] = None,
             signed: bool = False) -> LLVMValue:
        if not v:
            raise UnsupportedPcode("missing_operand")
        width = P._normalize_width(preferred_width or int(v["size"]) * 8)
        if v["is_constant"]:
            val = int(v["offset"])
            if int(v["size"]) == self.prog.ptr_bytes and val and val in self.ref_targets() and self.prog.const_symbolizable(val):
                return self.cast(LLVMValue(self.prog.ptr_bits, self.prog.address_const(val, self.int_ty)), width)
            return P._const(width, val)
        sz = int(v["size"]) * 8
        ptr = self.loc_ptr(v)
        loaded = self.new_temp(sz, f"load i{sz}, ptr {ptr}, align 1")
        return self.cast(loaded, width, signed=signed)

    def write(self, out: Optional[Dict[str, Any]], value: LLVMValue, signed: bool = False) -> None:
        if not out:
            return
        w = int(out["size"]) * 8
        iv = self.cast(value, w, signed=signed)
        self.lines.append(f"  store i{w} {iv.text}, ptr {self.loc_ptr(out)}, align 1")

    def cast(self, value: LLVMValue, width: int, signed: bool = False) -> LLVMValue:
        if value.text in ("undef", "poison"):
            raise UnsupportedPcode("undef_value")
        if value.is_integer and value.const is None and value.text.startswith("ptrtoint ("):
            if value.width == width:
                return value
            t = self.new_temp(value.width, f"add i{value.width} {value.text}, 0")
            return super().cast(t, width, signed)
        return super().cast(value, width, signed)

    def address_value(self, v):
        if v["is_constant"]:
            return LLVMValue(self.prog.ptr_bits, self.prog.data_int_const(int(v["offset"]), self.int_ty))
        return self.read(v, self.prog.ptr_bits)

    def memory_pointer(self, address: LLVMValue) -> LLVMValue:
        address = self.cast(address, self.prog.ptr_bits)
        return self.new_temp(self.prog.ptr_bits, f"inttoptr {self.int_ty} {address.text} to ptr")

    emit_load = G.HighEmitter.emit_load
    emit_store = G.HighEmitter.emit_store

    def emit_ptrsub(self, output, inputs):
        raise UnsupportedPcode("op_PTRSUB_in_raw")

    def emit_multiequal(self, output, inputs):
        raise UnsupportedPcode("op_MULTIEQUAL_in_raw")

    def emit_indirect(self, output, inputs):
        raise UnsupportedPcode("op_INDIRECT_in_raw")

    # ---------------------------------------------------------------- control flow
    def ins_label(self, addr: int) -> str:
        return "i_%x" % addr

    def goto_address(self, target: int) -> None:
        if target in self.addr_set:
            self.lines.append(f"  br label %{self.ins_label(target)}")
        elif target in self.delay_addrs:
            raise UnsupportedPcode("jump_into_delay_slot")
        elif target in self.prog.functions:  # ram target; external_functions is keyed by EXTERNAL offsets
            callee = self.prog.declare_function(self.prog.resolve_call_target(target))
            self.lines.append(f"  call {VOID} {callee}()")
            self.lines.append("  ret void")
        else:
            raise UnsupportedPcode("jump_target_unresolved:%x" % target)

    def emit_branch(self) -> None:
        t = self._op["inputs"][0]
        if t["is_constant"]:
            self.lines.append(f"  br label %{self.rel_label(self._opi + self.signed_rel(t))}")
        else:
            self.goto_address(int(t["offset"]))
        self.term = True

    def emit_cbranch(self, inputs) -> None:
        t = inputs[0]
        c = self.read(inputs[1], 1)
        nxt = self.rel_label(self._opi + 1)
        if t["is_constant"]:
            self.lines.append(f"  br i1 {c.text}, label %{self.rel_label(self._opi + self.signed_rel(t))}, label %{nxt}")
        else:
            target = int(t["offset"])
            if target in self.addr_set:
                self.lines.append(f"  br i1 {c.text}, label %{self.ins_label(target)}, label %{nxt}")
            else:
                far = f"far_{self.temp_index}"
                self.temp_index += 1
                self.lines.append(f"  br i1 {c.text}, label %{far}, label %{nxt}")
                self.lines.append(f"{far}:")
                self.goto_address(target)
        if nxt != self.next_label:
            self.lines.append(f"{nxt}:")
            self.declared_sub.add(nxt)
        else:
            self.term = True

    def signed_rel(self, v: Dict[str, Any]) -> int:
        o = int(v["offset"])
        bits = int(v["size"]) * 8
        if o >= 1 << (bits - 1):
            o -= 1 << bits
        return o

    def rel_label(self, opi: int) -> str:
        """Label of P-Code op index opi inside the current instruction (== len(ops) means fall through)."""
        n = len(self.cur_ins["pcode"])
        if opi < 0 or opi > n:
            raise UnsupportedPcode("relative_branch_out_of_range")
        if opi == n:
            return self.next_label
        lbl = "%s_%d" % (self.ins_label(int(self.cur_ins["address"])), opi)
        self.wanted_sub.add(lbl)
        return lbl

    def emit_branchind(self, inputs) -> None:
        target = self.read(inputs[0], self.prog.ptr_bits)
        flows = [int(x) for x in self.cur_ins.get("flows", [])]
        outside = [f for f in flows if f not in self.addr_set]
        if outside:  # a table target outside this function's instructions: never drop it silently
            raise UnsupportedPcode("branchind_target_outside_function:%x" % outside[0])
        if flows:
            selfsym = self.prog.declare_function(self.name)
            rel = self.new_temp(self.prog.ptr_bits,
                                f"sub {self.int_ty} {target.text}, ptrtoint (ptr {selfsym} to {self.int_ty})")
            lbl = f"swdefault{self.temp_index}"
            self.temp_index += 1
            cases = " ".join(f"{self.int_ty} {f - self.entry}, label %{self.ins_label(f)}" for f in sorted(set(flows)))
            self.lines.append(f"  switch {self.int_ty} {rel.text}, label %{lbl} [ {cases} ]")
            self.lines.append(f"{lbl}:")
            self.lines.append("  unreachable")
        else:  # Ghidra recovered no targets (unrecovered jump table): control flow unknown -> reject.
            # (Real register tail calls are CALLIND + RETURN in Ghidra's P-code and do not come here.)
            raise UnsupportedPcode("branchind_without_targets")
        self.term = True

    def emit_return(self, inputs) -> None:
        self.lines.append("  ret void")
        self.term = True

    def emit_call(self, output, inputs, indirect: bool) -> None:
        t = inputs[0]
        if indirect:
            tv = self.read(t, self.prog.ptr_bits)
            p = self.new_raw_temp(f"inttoptr {self.int_ty} {tv.text} to ptr")
            self.lines.append(f"  call {VOID} {p}()")
            if str(self.cur_ins.get("flow", "")).endswith("TERMINATOR"):  # Ghidra: call does not return
                self.lines.append("  unreachable")
                self.term = True
            return
        target = int(t["offset"])
        callee = self.prog.declare_function(self.prog.resolve_call_target(target))
        self.lines.append(f"  call {VOID} {callee}()")
        if self.prog.is_noreturn(target) or str(self.cur_ins.get("flow", "")).endswith("TERMINATOR"):
            self.lines.append("  unreachable")
            self.term = True

    def emit_callother(self, output, inputs) -> None:
        G.HighEmitter.emit_callother(self, output, inputs)

    def arg_value(self, v):
        return self.read(v, int(v["size"]) * 8)

    # ---------------------------------------------------------------- driver
    def emit(self) -> List[str]:
        if not self.insts:
            raise UnsupportedPcode("no_instructions")
        self.approx = set()
        self.plan_uniques()
        body = [i for i in self.insts if not i["in_delay_slot"]]
        self.lines.append(f"define void {G._q(self.name)}() {{")
        self.lines.append("entry:")
        self.lines.append(f"  %uniq = alloca [{self.unique_size} x i8], align 16")
        self.lines.append(f"  br label %{self.ins_label(int(body[0]['address']))}")
        for k, ins in enumerate(body):
            self.cur_ins = ins
            addr = int(ins["address"])
            nxt = int(body[k + 1]["address"]) if k + 1 < len(body) else None
            self.next_label = self.ins_label(nxt) if nxt is not None else "fell_off_end"
            self.wanted_sub, self.declared_sub = set(), set()
            self.lines.append(f"{self.ins_label(addr)}:")
            self.term = False
            ops = ins["pcode"]
            for i, op in enumerate(ops):
                sub = "%s_%d" % (self.ins_label(addr), i)
                if sub in self.wanted_sub_all(addr, ops) and sub not in self.declared_sub:
                    if not self.term:
                        self.lines.append(f"  br label %{sub}")
                    self.lines.append(f"{sub}:")
                    self.declared_sub.add(sub)
                    self.term = False
                elif self.term:
                    # dead P-Code after a terminator inside the instruction: give it its own (unreachable) block
                    self.lines.append(f"dead_{self.temp_index}:")
                    self.temp_index += 1
                    self.term = False
                self._op, self._opi = op, i
                self.emit_pcode_op(op)
            missing = self.wanted_sub - self.declared_sub
            if missing:
                raise UnsupportedPcode("relative_branch_target_missing")
            if not self.term:
                if nxt is None:
                    # machine semantics: execution continues at the fall-through address; if that is another
                    # function's entry (e.g. after a call that never returns), it enters that function
                    ft = int(ins["fallthrough"]) if ins.get("fallthrough") else None
                    if ft is None or ft not in self.prog.functions:
                        raise UnsupportedPcode("falls_off_function_end")
                    self.goto_address(ft)
                    self.approx.add("falls_into_next_function")
                else:
                    self.lines.append(f"  br label %{self.next_label}")
        self.lines.append("}")
        text = "\n".join(self.lines)
        if "fell_off_end" in text:
            raise UnsupportedPcode("falls_off_function_end")
        import re
        if re.search(r"\b(undef|poison)\b", text):
            raise UnsupportedPcode("undef_in_output")
        return self.lines

    def wanted_sub_all(self, addr: int, ops: List[Dict[str, Any]]) -> set:
        """All op indices of this instruction that a P-Code-relative branch targets (computed once per instruction)."""
        key = (addr, id(ops))
        if getattr(self, "_wsa_key", None) != key:
            s = set()
            for i, op in enumerate(ops):
                if op["mnemonic"] in ("BRANCH", "CBRANCH") and op["inputs"][0]["is_constant"]:
                    j = i + self.signed_rel(op["inputs"][0])
                    if 0 <= j < len(ops):
                        s.add("%s_%d" % (self.ins_label(addr), j))
            self._wsa_key, self._wsa = key, s
        return self._wsa

    def emit_pcode_op(self, op: Dict[str, Any]) -> None:
        m = op["mnemonic"]
        if m in ("SEGMENTOP", "CPOOLREF", "NEW", "INSERT", "EXTRACT"):
            raise UnsupportedPcode(f"op_{m}")
        if m == "BRANCH":
            self.comment(op.get("text", m))
            self.emit_branch()
            return
        super().emit_pcode_op(op)
