"""Numeric exceptional inputs must not turn into LLVM poison/UB."""
import ctypes
import types

import pytest
from llvmlite import ir

from test_ida2llvm_microcode import L
from test_ida2llvm_audit import jit, needs_opt, run_oc


@needs_opt
@pytest.mark.parametrize('signed', [False, True])
def test_a64_division_boundaries(L, monkeypatch, signed):
    monkeypatch.setattr(L, 'source_family', 'arm')
    monkeypatch.setattr(L, 'ptrsize', 64)
    op = 'sdiv' if signed else 'udiv'
    monkeypatch.setattr(L.idc, 'print_insn_mnem', lambda ea: op)
    monkeypatch.setattr(L.idc, 'print_operand', lambda ea, i: f'x{i}')
    ty = ir.IntType(64)
    module = ir.Module()
    fn = ir.Function(module, ir.FunctionType(ty, [ty, ty]), 'probe')
    b = ir.IRBuilder(fn.append_basic_block())
    result = L._lift_division(op, *fn.args, b, 0)
    # A guard AFTER division must survive Oc, including division-by-zero.
    b.ret(b.select(b.icmp_unsigned('==', fn.args[1], ty(0)), ty(99), result))
    values = [0, 1, -1, -(1 << 63), (1 << 63) - 1]
    mask = (1 << 64) - 1
    for text in (str(module), run_oc(str(module))):
        engine = jit(text)
        call = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)(engine.get_function_address('probe'))
        for x in values:
            for y in values:
                if y == 0:
                    expected = 99
                elif signed:
                    expected = (abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1)
                else:
                    expected = (x & mask) // (y & mask)
                assert call(x, y) == expected & mask


@pytest.mark.parametrize('family,bits,mnemonic', [('x86', 64, 'idiv'), ('mips', 64, 'div'), ('arm', 32, 'sdiv'), ('arm', 64, 'bl')])
def test_unresolved_division_rejected(L, monkeypatch, family, bits, mnemonic):
    monkeypatch.setattr(L, 'source_family', family)
    monkeypatch.setattr(L, 'ptrsize', bits)
    monkeypatch.setattr(L.idc, 'print_insn_mnem', lambda ea: mnemonic)
    ty = ir.IntType(64)
    fn = ir.Function(ir.Module(), ir.FunctionType(ty, [ty, ty]), 'probe')
    with pytest.raises(L.UnsupportedMicrocode, match='division_machine_semantics_unresolved'):
        L._lift_division('sdiv', *fn.args, ir.IRBuilder(fn.append_basic_block()), 0)


@needs_opt
@pytest.mark.parametrize('signed', [False, True])
def test_a64_saturating_conversion(L, monkeypatch, signed):
    monkeypatch.setattr(L, 'source_family', 'arm')
    monkeypatch.setattr(L, 'ptrsize', 64)
    monkeypatch.setattr(L.idc, 'print_insn_mnem', lambda ea: 'fcvtzs' if signed else 'fcvtzu')
    monkeypatch.setattr(L.idc, 'print_operand', lambda ea, i: ('x0', 'd0', '')[i])
    module = ir.Module()
    ty = ir.IntType(64)
    fn = ir.Function(module, ir.FunctionType(ty, [ir.DoubleType()]), 'probe')
    b = ir.IRBuilder(fn.append_basic_block())
    b.ret(L._lift_float_to_int(fn.args[0], ty, signed, b, 0))
    lower, upper = (-(1 << 63), (1 << 63) - 1) if signed else (0, (1 << 64) - 1)
    pairs = [(float('nan'), 0), (float('inf'), upper), (-float('inf'), lower),
             (-1.75, -1 if signed else 0), (1.75, 1), (0.0, 0), (float(1 << 64), upper)]
    for text in (str(module), run_oc(str(module))):
        engine = jit(text)
        call = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_double)(engine.get_function_address('probe'))
        for value, expected in pairs:
            assert call(value) == expected & ((1 << 64) - 1)


def test_helper_conversion_and_unmatched_arm_carry_rejected(L, monkeypatch):
    monkeypatch.setattr(L, 'source_family', 'arm')
    monkeypatch.setattr(L, 'ptrsize', 64)
    monkeypatch.setattr(L.idc, 'print_insn_mnem', lambda ea: 'bl')
    with pytest.raises(L.UnsupportedMicrocode, match='float_conversion_machine_semantics_unresolved'):
        L._lift_float_to_int(ir.DoubleType()(1.0), ir.IntType(64), False, None, 0)
    with pytest.raises(L.UnsupportedMicrocode, match='arm_shift_carry_source_unresolved'):
        L._shift_carry_bit(None, None, None, types.SimpleNamespace(), False, None)


@pytest.mark.parametrize('operands', [('x0', 'd0', '#3'), ('v0.2d', 'v0.2d', ''), ('w0', 'd0', '')])
def test_nonmatching_float_conversion_rejected(L, monkeypatch, operands):
    monkeypatch.setattr(L, 'source_family', 'arm')
    monkeypatch.setattr(L, 'ptrsize', 64)
    monkeypatch.setattr(L.idc, 'print_insn_mnem', lambda ea: 'fcvtzs')
    monkeypatch.setattr(L.idc, 'print_operand', lambda ea, i: operands[i])
    with pytest.raises(L.UnsupportedMicrocode, match='float_conversion_machine_semantics_unresolved'):
        L._lift_float_to_int(ir.DoubleType()(1.0), ir.IntType(64), True, None, 0)


@needs_opt
@pytest.mark.parametrize('family', ['x86', 'arm', 'mips'])
@pytest.mark.parametrize('op', ['udiv', 'sdiv', 'urem', 'srem'])
def test_reconstructed_constant_division(L, monkeypatch, family, op):
    monkeypatch.setattr(L, 'source_family', family)
    monkeypatch.setattr(L, 'ptrsize', 32)
    def no_decode(ea):
        raise AssertionError('safe constant arithmetic must not require a native DIV')
    monkeypatch.setattr(L.idc, 'print_insn_mnem', no_decode)
    ty = ir.IntType(32)
    mask = (1 << 32) - 1
    divisors = [1, 3, 8, 0x80000000, -7]
    if op.startswith('u'):
        divisors += [-1]
    for divisor in divisors:
        L.current_approx.clear()
        module = ir.Module()
        fn = ir.Function(module, ir.FunctionType(ty, [ty]), 'probe')
        b = ir.IRBuilder(fn.append_basic_block())
        b.ret(L._lift_division(op, fn.args[0], ty(divisor), b, 0))
        assert not L.current_approx
        d = divisor & mask
        if op.startswith('s') and d >= 1 << 31:
            d -= 1 << 32
        for text in (str(module), run_oc(str(module))):
            engine = jit(text)
            call = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32)(engine.get_function_address('probe'))
            for raw in [0, 1, 7, 8, 0x7fffffff, 0x80000000, 0xffffffff]:
                x = raw - (1 << 32) if op.startswith('s') and raw >= 1 << 31 else raw
                q = (abs(x) // abs(d)) * (-1 if (x < 0) != (d < 0) else 1)
                expected = x - q * d if op.endswith('rem') else q
                assert call(raw) == expected & mask


@pytest.mark.parametrize('op,divisor', [('udiv', 0), ('urem', 0), ('sdiv', -1), ('srem', -1)])
def test_exceptional_constant_not_admitted(L, monkeypatch, op, divisor):
    monkeypatch.setattr(L, 'source_family', 'x86')
    monkeypatch.setattr(L.idc, 'print_insn_mnem', lambda ea: 'sar')
    ty = ir.IntType(32)
    fn = ir.Function(ir.Module(), ir.FunctionType(ty, [ty]), 'probe')
    b = ir.IRBuilder(fn.append_basic_block())
    with pytest.raises(L.UnsupportedMicrocode, match='division_machine_semantics_unresolved'):
        L._lift_division(op, fn.args[0], ty(divisor), b, 0)


def test_constant_cast_proof_is_bounded_and_preserves_sign(L):
    ty = ir.IntType(32)
    fn = ir.Function(ir.Module(), ir.FunctionType(ty, [ty]), 'probe')
    b = ir.IRBuilder(fn.append_basic_block())
    assert L._constant_int_bits(b.zext(ir.IntType(8)(-1), ty)) == 255
    assert L._constant_int_bits(b.sext(ir.IntType(8)(-1), ty)) == 0xffffffff
    assert L._constant_int_bits(b.trunc(ir.IntType(64)(0x100000000), ty)) == 0
    assert L._constant_int_bits(fn.args[0]) is None
    assert L._constant_int_bits(b.zext(ir.IntType(8)(1), ty), depth=0) is None
