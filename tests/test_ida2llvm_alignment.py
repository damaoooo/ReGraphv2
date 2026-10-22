"""Memory-access alignment in ida2llvm (final-audit P1).

LLVM treats a load/store without `align` as ABI-aligned, which is a guarantee the source program
does not give for arbitrary addresses, packed structure members or byte offsets in a stack frame.
The lifter keeps only alignment it can prove from the base object (alloca/global) and constant
offsets and uses `align 1` otherwise; explicitly aligned SIMD accesses and atomics are untouched.

Oracles: native execution before/after the Oc pipeline with a byte-exact memory check, and the
code llc generates for a strict-alignment target (MIPS: unaligned accesses need lwl/lwr, swl/swr).
"""
import ctypes
import shutil
import subprocess
import types

import pytest
from llvmlite import binding, ir

from test_ida2llvm_microcode import L, insn, mop, new_function, stub_operands, verify  # noqa: F401
from test_ida2llvm_audit import jit, needs_opt, run_oc

LLC = shutil.which("llc-18") or shutil.which("llc")
I8P = ir.IntType(8).as_pointer()
I32 = ir.IntType(32)


def stores(module, typ="i32"):
    return [line for line in str(module).splitlines() if f"store {typ}" in line]


def test_store_to_arbitrary_address_is_unaligned(L):
    """Reviewer's reproducer: i32 stored through byte buffer + 1 must not claim align 4."""
    module, func, blocks, builder = new_function()
    raw = builder.alloca(ir.ArrayType(ir.IntType(8), 9))
    raw.align = 1
    p = builder.gep(raw, [I32(0), I32(1)])
    L._store_to_address(I32(123), p, builder)
    builder.ret_void()
    ir.IRBuilder(blocks[1]).ret_void()
    parsed = binding.parse_assembly(str(module))
    parsed.verify()
    assert all("align 1" in line for line in stores(parsed))


def build_unaligned_roundtrip(L, triple=None):
    """i32 f(i8* p, i32 v): stx v -> [p+1]; return ldx [p+1], lifted through lift_insn."""
    module = ir.Module(context=ir.Context())
    module.triple = triple or binding.get_default_triple()
    func = ir.Function(module, ir.FunctionType(I32, [I8P, I32]), "roundtrip")
    func.lvars = {}
    entry = func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    with builder.goto_entry_block():
        result = builder.alloca(I32, name="result")
    address = builder.gep(func.args[0], [I32(1)])
    selector = mop(L, "mop_r", size=2)
    selector.dstr = lambda: "ds.2"
    L.lift_insn(insn(L, "m_stx", l=mop(L, "mop_r", value=func.args[1], size=4), r=selector,
                     d=mop(L, "mop_d", value=address)), entry, builder)
    L.lift_insn(insn(L, "m_ldx", l=selector, r=mop(L, "mop_d", value=address),
                     d=mop(L, "mop_l", value=result, size=4)), entry, builder)
    builder.position_at_end(entry)
    builder.ret(builder.load(result))
    return module


def test_ldx_and_stx_use_align_1(stub_operands):
    module = build_unaligned_roundtrip(stub_operands)
    text = str(module)
    # accesses through the source address p+1; the lifter's own %result slot keeps its ABI alignment
    source = [line for line in text.splitlines()
              if ("store i32" in line or "load i32" in line) and '"result"' not in line]
    assert len(source) == 2 and all(line.endswith("align 1") for line in source)
    assert all("align" not in line for line in text.splitlines() if '"result"' in line and ("load" in line or "store" in line))
    verify(module)


@needs_opt
def test_unaligned_roundtrip_bytes_before_and_after_oc(stub_operands):
    module = build_unaligned_roundtrip(stub_operands)
    optimized = run_oc(str(module))
    assert "align 4" not in "\n".join(l for l in optimized.splitlines() if "%0" in l or "%1" in l)
    for text in (str(module), optimized):
        engine = jit(text)
        fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32)(engine.get_function_address("roundtrip"))
        buffer = (ctypes.c_uint8 * 8)(*[0xAA] * 8)
        assert fn(ctypes.addressof(buffer), 0x11223344) == 0x11223344
        assert bytes(buffer) == bytes([0xAA, 0x44, 0x33, 0x22, 0x11, 0xAA, 0xAA, 0xAA])


def build_separate_access(L, triple):
    """i32 load(i8* p) = ldx [p+1]; void store(i8* p, i32 v): stx v -> [p+1] (no store-to-load forwarding)."""
    module = ir.Module(context=ir.Context())
    module.triple = triple
    selector = mop(L, "mop_r", size=2)
    selector.dstr = lambda: "ds.2"
    load_fn = ir.Function(module, ir.FunctionType(I32, [I8P]), "load_at_1")
    load_fn.lvars = {}
    entry = load_fn.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    result = builder.alloca(I32, name="result")
    address = builder.gep(load_fn.args[0], [I32(1)])
    L.lift_insn(insn(L, "m_ldx", l=selector, r=mop(L, "mop_d", value=address),
                     d=mop(L, "mop_l", value=result, size=4)), entry, builder)
    builder.position_at_end(entry)
    builder.ret(builder.load(result))
    store_fn = ir.Function(module, ir.FunctionType(ir.VoidType(), [I8P, I32]), "store_at_1")
    store_fn.lvars = {}
    entry = store_fn.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    address = builder.gep(store_fn.args[0], [I32(1)])
    L.lift_insn(insn(L, "m_stx", l=mop(L, "mop_r", value=store_fn.args[1], size=4), r=selector,
                     d=mop(L, "mop_d", value=address)), entry, builder)
    builder.position_at_end(entry)
    builder.ret_void()
    return module


@needs_opt
@pytest.mark.skipif(LLC is None, reason="llc not installed")
@pytest.mark.parametrize("claimed_alignment", [None, 4])
def test_strict_alignment_target_uses_unaligned_instructions(stub_operands, tmp_path, claimed_alignment):
    """
    On MIPS an align-4 access compiles to lw/sw, which trap on odd addresses; the lifted align-1 access
    must compile to lwl/lwr + swl/swr. claimed_alignment=4 is the negative control (the old behaviour).
    """
    L = stub_operands
    triple, layout = L.SOURCE_TARGETS[("mips", 32, "big")]
    L.ptrsize, L.is_big_endian, L.byteorder = 32, True, "big"
    try:
        L._set_sizing_layout(layout)
        module = build_separate_access(L, triple)
        module.data_layout = layout
        if claimed_alignment:
            for fn in module.functions:
                for block in fn.blocks:
                    for inst in block.instructions:
                        if isinstance(inst, (ir.LoadInstr, ir.StoreInstr)) and inst.align == 1:
                            inst.align = claimed_alignment
        optimized = run_oc(str(module))
    finally:
        L._reset_lift_state()
        L.ptrsize, L.is_big_endian, L.byteorder = 64, False, "little"
    src = tmp_path / "m.ll"
    src.write_text(optimized)
    asm = subprocess.run([LLC, f"-mtriple={triple}", "-O2", str(src), "-o", "-"],
                         capture_output=True, text=True, check=True).stdout
    mnemonics = {line.split()[0] for line in asm.splitlines() if line.startswith("\t") and not line.startswith("\t.")}
    if claimed_alignment:
        assert {"lw", "sw"} <= mnemonics  # what an unproven align 4 turns into: trapping accesses
    else:
        assert {"lwl", "lwr", "swl", "swr"} <= mnemonics and not {"lw", "sw"} & mnemonics


@pytest.mark.parametrize("offset, expected", [(0, None), (4, None), (2, 2), (1, 1)])
def test_alignment_proven_from_base_and_offset(L, offset, expected):
    """i32 access at a constant byte offset into an 8-aligned i64 slot: only provable alignment is kept."""
    module, func, (entry, _), builder = new_function()
    slot = builder.alloca(ir.IntType(64))
    byte = builder.bitcast(slot, I8P)
    ptr = builder.bitcast(builder.gep(byte, [I32(offset)]), I32.as_pointer())
    load = builder.load(ptr)
    store = builder.store(I32(1), ptr)
    L._apply_conservative_alignment(func)
    assert load.align == expected and store.align == expected


def test_packed_struct_member_and_unknown_pointer_are_unaligned(L):
    module, func, (entry, _), builder = new_function()
    packed = ir.LiteralStructType([ir.IntType(8), I32], packed=True)
    obj = builder.alloca(packed)
    member = builder.gep(obj, [I32(0), I32(1)])
    raw = builder.inttoptr(ir.IntType(64)(0x1000), I32.as_pointer())
    loads = [builder.load(member), builder.load(raw), builder.load(builder.alloca(I32))]
    L._apply_conservative_alignment(func)
    assert [l.align for l in loads] == [1, 1, None]  # whole, typed alloca keeps its ABI alignment


def test_explicit_alignment_is_kept(L):
    module, func, (entry, _), builder = new_function()
    raw = builder.inttoptr(ir.IntType(64)(0x1000), ir.IntType(128).as_pointer())
    load = builder.load(raw, align=16)  # e.g. _mm_load_si128: alignment is part of its contract
    L._apply_conservative_alignment(func)
    assert load.align == 16


def test_virtual_fs_helper_body_is_unaligned(L):
    module = ir.Module(context=ir.Context())
    L._reset_lift_state()
    f = L.lift_intrinsic_function(module, "__readfsqword")
    L._apply_conservative_alignment(f)
    assert "align 1" in str(f)


def test_pointers_to_named_structs_and_struct_slots(L):
    """Regression: alignment of %struct* values must not need the struct body in a separate parse."""
    L._reset_lift_state()
    module = ir.Module(context=L.llvm_context)
    named = L.llvm_context.get_identified_type("utsname")
    named.packed = True
    named.set_body(ir.ArrayType(ir.IntType(8), 390))
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [named.as_pointer()]), "f")
    builder = ir.IRBuilder(func.append_basic_block())
    slot = builder.alloca(named.as_pointer())
    store = builder.store(func.args[0], slot)
    load = builder.load(builder.bitcast(builder.inttoptr(ir.IntType(64)(0x10), I8P), named.as_pointer().as_pointer()))
    builder.ret_void()
    L._apply_conservative_alignment(func)
    assert store.align is None and load.align == 1
    binding.parse_assembly(str(module)).verify()
