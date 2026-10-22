"""Shift boundary regression tests, before and after the paper's Oc pipeline."""
import ctypes
import types

import pytest
from llvmlite import ir

from test_ida2llvm_microcode import L
from test_ida2llvm_audit import jit, needs_opt, run_oc


def reference(value, count, width, family, bits, op):
    mask = (1 << width) - 1
    value &= mask
    if family == "arm" and bits == 32:
        count &= 255
    else:
        count &= 63 if width == 64 else 31
    signed = value - (1 << width) if value >> (width - 1) else value
    if op == "ashr":
        return (signed >> min(count, width)) & mask
    if count >= width:
        return 0
    return ((value << count) if op == "shl" else (value >> count)) & mask


@needs_opt
@pytest.mark.parametrize("family,bits,width", [
    ("x86", 64, 8), ("x86", 64, 16), ("x86", 64, 32), ("x86", 64, 64),
    ("x86", 32, 32), ("arm", 32, 32), ("arm", 64, 32), ("arm", 64, 64),
    ("mips", 32, 32), ("mips", 64, 64),
])
@pytest.mark.parametrize("op", ["shl", "lshr", "ashr"])
def test_shift_boundaries(L, monkeypatch, family, bits, width, op):
    monkeypatch.setattr(L, "source_family", family)
    monkeypatch.setattr(L, "ptrsize", bits)
    module = ir.Module()
    fn = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(64)]), "probe")
    b = ir.IRBuilder(fn.append_basic_block())
    x = b.trunc(fn.args[0], ir.IntType(width)) if width < 64 else fn.args[0]
    result = L._lift_shift(op, x, fn.args[1], width, b)
    b.ret(b.zext(result, ir.IntType(64)) if width < 64 else result)
    values = [0, 1, (1 << width) - 1, 1 << (width - 1), 0x8123456789ABCDEF]
    counts = [0, 1, width - 1, width, width + 1, 31, 32, 63, 64, 255, 256, 257, 2**64 - 1]
    for text in (str(module), run_oc(str(module))):
        engine = jit(text)
        call = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)(engine.get_function_address("probe"))
        for value in values:
            for count in counts:
                assert call(value, count) == reference(value, count, width, family, bits, op)


@pytest.mark.parametrize("family,bits,width", [("arm", 32, 8), ("x86", 64, 128), ("riscv", 64, 64)])
def test_unknown_shift_fails_closed(L, monkeypatch, family, bits, width):
    monkeypatch.setattr(L, "source_family", family)
    monkeypatch.setattr(L, "ptrsize", bits)
    ty = ir.IntType(width)
    fn = ir.Function(ir.Module(), ir.FunctionType(ty, [ty, ty]), "probe")
    b = ir.IRBuilder(fn.append_basic_block())
    for count in (fn.args[1], ty(width)):
        with pytest.raises(L.UnsupportedMicrocode, match="shift_count_unspecified"):
            L._lift_shift("shl", fn.args[0], count, width, b)
    assert L._lift_shift("shl", fn.args[0], ty(1), width, b).type == ty


@needs_opt
@pytest.mark.parametrize("op", ["shl", "lshr", "ashr"])
def test_arm_narrow_masked_shift(L, monkeypatch, op):
    monkeypatch.setattr(L, "source_family", "arm")
    monkeypatch.setattr(L, "ptrsize", 32)
    ty = ir.IntType(8)
    module = ir.Module()
    fn = ir.Function(module, ir.FunctionType(ty, [ty, ty]), "probe")
    b = ir.IRBuilder(fn.append_basic_block())
    count = b.and_(fn.args[1], ty(7))
    b.ret(L._lift_shift(op, fn.args[0], count, 8, b))
    for text in (str(module), run_oc(str(module))):
        engine = jit(text)
        call = ctypes.CFUNCTYPE(ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8)(engine.get_function_address("probe"))
        for x in range(256):
            for n in range(256):
                assert call(x, n) == reference(x, n & 7, 8, "arm", 32, op)


def test_nested_carry_without_previous_flag_is_rejected(L):
    with pytest.raises(L.UnsupportedMicrocode, match="shift_carry_without_previous_flag"):
        L._previous_flag(None, None)


def test_bounded_native_shift_avoids_saturation_ir(L, monkeypatch):
    monkeypatch.setattr(L, 'source_family', 'arm')
    monkeypatch.setattr(L, 'ptrsize', 32)
    ty = ir.IntType(32)
    fn = ir.Function(ir.Module(), ir.FunctionType(ty, [ty, ty]), 'probe')
    b = ir.IRBuilder(fn.append_basic_block())
    count = b.and_(fn.args[1], ty(31))
    b.ret(L._lift_shift('lshr', fn.args[0], count, 32, b))
    assert [i.opname for i in b.block.instructions] == ['and', 'lshr', 'ret']


@needs_opt
@pytest.mark.parametrize("family,bits,width", [("x86", 64, 8), ("x86", 64, 32), ("arm", 32, 32)])
@pytest.mark.parametrize("left", [False, True])
def test_shift_carry_zero_preserves_flag(L, monkeypatch, family, bits, width, left):
    monkeypatch.setattr(L, "source_family", family)
    monkeypatch.setattr(L, "ptrsize", bits)
    module = ir.Module()
    ty = ir.IntType(width)
    fn = ir.Function(module, ir.FunctionType(ir.IntType(8), [ty, ty, ir.IntType(8)]), "probe")
    b = ir.IRBuilder(fn.append_basic_block())
    instruction = types.SimpleNamespace(l=types.SimpleNamespace(size=width//8), r=types.SimpleNamespace(size=width//8))
    previous = b.trunc(fn.args[2], ir.IntType(1))
    if family == "arm":
        with pytest.raises(L.UnsupportedMicrocode, match="arm_shift_carry_source_unresolved"):
            L._shift_carry_bit(fn.args[0], fn.args[1], b, instruction, left, previous)
        return  # ARM is tested through the early hook against actual machine code.
    result = L._shift_carry_bit(fn.args[0], fn.args[1], b, instruction, left, previous)
    b.ret(b.zext(result, ir.IntType(8)))
    ctype = ctypes.c_uint8 if width == 8 else ctypes.c_uint32
    for text in (str(module), run_oc(str(module))):
        engine = jit(text)
        call = ctypes.CFUNCTYPE(ctypes.c_uint8, ctype, ctype, ctypes.c_uint8)(engine.get_function_address("probe"))
        for previous in (0, 1):
            assert call(1, 0, previous) == previous
            # A nonzero raw count can mask to zero on x86 too.
            if family == "x86":
                assert call(1, 32, previous) == previous
        assert call(1 if not left else 1 << (width-1), 1, 0) == 1


def test_nested_constant_carry_needs_no_previous_flag(L, monkeypatch):
    monkeypatch.setattr(L, "source_family", "x86")
    monkeypatch.setattr(L, "ptrsize", 32)
    ty = ir.IntType(32)
    fn = ir.Function(ir.Module(), ir.FunctionType(ir.IntType(1), [ty]), "probe")
    b = ir.IRBuilder(fn.append_basic_block())
    instruction = types.SimpleNamespace(l=types.SimpleNamespace(size=4), r=types.SimpleNamespace(size=1))
    result = L._shift_carry_bit(fn.args[0], ir.IntType(8)(1), b, instruction, False, None)
    b.ret(result)
