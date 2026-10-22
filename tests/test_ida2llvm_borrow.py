"""Subtraction-with-carry helpers must include both the incoming flag and wrap."""
import ctypes

import pytest
from llvmlite import ir
from test_ida2llvm_microcode import L
from test_ida2llvm_audit import jit, run_oc, needs_opt


@needs_opt
@pytest.mark.parametrize('family', ['arm', 'ppc', 'x86'])
@pytest.mark.parametrize('helper', ['__CFSUB__', '__OFSUB__'])
def test_borrow_helpers_exhaustive(L, monkeypatch, family, helper):
    monkeypatch.setattr(L, 'source_family', family)
    module = ir.Module()
    ty = ir.IntType(8)
    fn = ir.Function(module, ir.FunctionType(ty, [ty, ty, ty]), 'probe')
    b = ir.IRBuilder(fn.append_basic_block())
    b.ret(L._lift_semantic_helper(helper, ty, list(fn.args), b))
    for text in (str(module), run_oc(str(module))):
        engine = jit(text)
        call = ctypes.CFUNCTYPE(ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8)(engine.get_function_address('probe'))
        for x in range(256):
            for y in range(256):
                for cf in (0, 1):
                    borrow = 1 - cf if family in ('arm', 'ppc') else cf
                    signed = (x if x < 128 else x - 256) - (y if y < 128 else y - 256) - borrow
                    if helper == '__OFSUB__':
                        expected = not -128 <= signed <= 127
                    else:
                        expected = x < y + borrow
                        if family in ('arm', 'ppc'):
                            expected = not expected
                    assert call(x, y, cf) == expected, (x, y, cf)
