"""Exact Hex-Rays helper semantics in ida2llvm (audit C09).

The LLVM implementation of each helper is optimized with the paper's Oc pipeline and executed
natively; these are pure integer/memory operations, so host execution is a valid oracle. The
reference is IDA's own include/defs.h compiled with clang (`-D__ARM__` for the ARM carry
convention). Where defs.h itself is undefined behaviour (rotate by a multiple of 32/64 bits)
the expected value is the machine semantics (identity).
"""
import ctypes
import os
import random
import shutil
import subprocess
import types

import pytest
from llvmlite import binding, ir

from test_ida2llvm_microcode import L, callinfo, insn, mop, new_function, stub_operands, verify  # noqa: F401
from test_ida2llvm_audit import jit, needs_opt, run_oc

IDA_INCLUDE = os.path.join(os.environ.get("IDADIR", os.path.expanduser("~/ida-pro-9.4")), "include")
CLANGXX = shutil.which("clang++-18") or shutil.which("clang++")
needs_defs = pytest.mark.skipif(CLANGXX is None or not os.path.exists(os.path.join(IDA_INCLUDE, "defs.h")),
                                reason="needs clang++ and IDA include/defs.h")
WIDTHS = {8: ("uint8", ctypes.c_uint8), 16: ("uint16", ctypes.c_uint16), 32: ("uint32", ctypes.c_uint32),
          64: ("uint64", ctypes.c_uint64)}
SUFFIX = {8: "1", 16: "2", 32: "4", 64: "8"}


def compile_reference(tmp_dir, defines=()):
    lines = ["#include <defs.h>", 'extern "C" {']
    for width, (ctype, _) in WIDTHS.items():
        n = SUFFIX[width]
        lines += [f"{ctype} rol{width}({ctype} v, int c) {{ return __ROL{n}__(v, c); }}",
                  f"{ctype} ror{width}({ctype} v, int c) {{ return __ROR{n}__(v, c); }}"]
        for flag in ("OFSUB", "OFADD", "CFSUB", "CFADD"):
            lines.append(f"int8 {flag.lower()}{width}({ctype} x, {ctype} y) {{ return __{flag}__(x, y); }}")
    for width, half in ((16, "uint8"), (32, "uint16"), (64, "uint32")):
        lines.append(f"uint{width} pair{width}({half} h, {half} l) {{ return __PAIR{width}__(h, l); }}")
    lines.append("}")
    src, lib = tmp_dir / "ref.cpp", tmp_dir / f"ref{'_'.join(defines)}.so"
    src.write_text("\n".join(lines))
    subprocess.run([CLANGXX, "-std=c++17", "-O1", "-shared", "-fPIC", *[f"-D{d}" for d in defines],
                    "-I", IDA_INCLUDE, str(src), "-o", str(lib)], check=True)
    return ctypes.CDLL(str(lib))


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    return compile_reference(tmp_path_factory.mktemp("defs"))


@pytest.fixture(scope="module")
def ref_arm(tmp_path_factory):
    return compile_reference(tmp_path_factory.mktemp("defs_arm"), ("__ARM__",))


def build_helper(L, name, ret, params, family="x86"):
    """A module with one function computing helper `name` on its parameters, run through Oc and JIT."""
    L.source_family = family
    module = ir.Module(context=ir.Context())
    module.triple = binding.get_default_triple()
    fn = ir.Function(module, ir.FunctionType(ret, params), "probe")
    builder = ir.IRBuilder(fn.append_basic_block())
    value = L._lift_semantic_helper(name, ret, list(fn.args), builder)
    assert value is not None and value is not True, name
    builder.ret(value)
    text = str(module)
    assert f'@"{name}"' not in text  # no opaque declaration of the helper
    return jit(run_oc(text))


def counts_for(width):
    return sorted({0, 1, 2, width - 1, width, width + 1, 2 * width, -1, -2, -width, -(width + 1), 37, -37})


def values_for(width):
    rng = random.Random(width)
    mask = (1 << width) - 1
    return [0, 1, mask, 1 << (width - 1), 0x5A5A5A5A5A5A5A5A & mask] + [rng.getrandbits(width) for _ in range(6)]


@needs_opt
@needs_defs
@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("direction", ["L", "R"])
def test_rotate_matches_defs_h(L, ref, width, direction):
    ty = ir.IntType(width)
    engine = build_helper(L, f"__RO{direction}{SUFFIX[width]}__", ty, [ty, ir.IntType(32)])
    ctype = WIDTHS[width][1]
    lifted = ctypes.CFUNCTYPE(ctype, ctype, ctypes.c_int32)(engine.get_function_address("probe"))
    reference = getattr(ref, f"ro{direction.lower()}{width}")
    reference.restype, reference.argtypes = ctype, [ctype, ctypes.c_int32]
    for value in values_for(width):
        for count in counts_for(width):
            got = lifted(value, count)
            if width >= 32 and count % width == 0:
                assert got == value, (value, count)  # defs.h shifts by W here (UB); the machine rotate is identity
            else:
                assert got == reference(value, count), (value, count)


@needs_opt
@needs_defs
@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("flag", ["__OFSUB__", "__OFADD__", "__CFSUB__", "__CFADD__"])
@pytest.mark.parametrize("family", ["x86", "arm"])
def test_flag_helpers_match_defs_h(L, ref, ref_arm, width, flag, family):
    ty = ir.IntType(width)
    engine = build_helper(L, flag, ir.IntType(8), [ty, ty], family)
    ctype = WIDTHS[width][1]
    lifted = ctypes.CFUNCTYPE(ctypes.c_int8, ctype, ctype)(engine.get_function_address("probe"))
    reference = getattr(ref_arm if family == "arm" else ref, f"{flag.strip('_').lower()}{width}")
    reference.restype, reference.argtypes = ctypes.c_int8, [ctype, ctype]
    for x in values_for(width):
        for y in values_for(width):
            assert lifted(x, y) == reference(x, y), (x, y)


@needs_opt
@needs_defs
@pytest.mark.parametrize("width", [16, 32, 64])
def test_pair_matches_defs_h(L, ref, width):
    half = ir.IntType(width // 2)
    engine = build_helper(L, f"__PAIR{width}__", ir.IntType(width), [half, half])
    ctype, htype = WIDTHS[width][1], WIDTHS[width // 2][1]
    lifted = ctypes.CFUNCTYPE(ctype, htype, htype)(engine.get_function_address("probe"))
    reference = getattr(ref, f"pair{width}")
    reference.restype, reference.argtypes = ctype, [htype, htype]
    for h in values_for(width // 2):
        for low in values_for(width // 2):
            assert lifted(h, low) == reference(h, low)


@needs_opt
@pytest.mark.parametrize("name, width", [("_byteswap_ushort", 16), ("_byteswap_ulong", 32), ("_byteswap_uint64", 64)])
def test_byteswap(L, name, width):
    ty = ir.IntType(width)
    engine = build_helper(L, name, ty, [ty])
    ctype = WIDTHS[width][1]
    lifted = ctypes.CFUNCTYPE(ctype, ctype)(engine.get_function_address("probe"))
    for value in values_for(width):
        assert lifted(value) == int.from_bytes(value.to_bytes(width // 8, "little"), "big")


@needs_opt
def test_clz_of_zero_is_width(L):
    ty = ir.IntType(32)
    engine = build_helper(L, "__clz", ty, [ty])
    lifted = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32)(engine.get_function_address("probe"))
    for value in values_for(32):
        assert lifted(value) == (32 - value.bit_length())


@needs_opt
@pytest.mark.parametrize("name, args, memory, expected_ret, expected_mem", [
    ("_InterlockedCompareExchange", (7, 5), 5, 5, 7),     # (Destination, Exchange, Comparand) -> initial
    ("_InterlockedCompareExchange", (7, 4), 5, 5, 5),
    ("_InterlockedExchange", (9,), 5, 5, 9),
    ("_InterlockedExchangeAdd", (3,), 5, 5, 8),
    ("_InterlockedIncrement", (), 5, 6, 6),                 # returns the new value
    ("_InterlockedDecrement", (), 5, 4, 4),
    ("_InterlockedOr", (0x10,), 5, 5, 0x15),
])
def test_interlocked(L, name, args, memory, expected_ret, expected_mem):
    ty = ir.IntType(32)
    engine = build_helper(L, name, ty, [ty.as_pointer()] + [ty] * len(args))
    lifted = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), *[ctypes.c_int32] * len(args))(
        engine.get_function_address("probe"))
    cell = ctypes.c_int32(memory)
    assert lifted(ctypes.byref(cell), *args) == expected_ret
    assert cell.value == expected_mem


@needs_opt
@pytest.mark.parametrize("name", ["_mm_load_si128", "_mm_loadu_si128"])
def test_sse_loads_read_16_bytes(L, name):
    i128 = ir.IntType(128)
    engine = build_helper(L, name, i128, [ir.IntType(8).as_pointer()])
    buffer = (ctypes.c_uint8 * 48)(*range(48))
    offset = 16 if name == "_mm_load_si128" else 3  # aligned vs unaligned source
    lifted = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)(engine.get_function_address("probe"))
    low = lifted(ctypes.addressof(buffer) + offset)  # i128 returns its low 64 bits in rax on x86_64
    assert low == int.from_bytes(bytes(range(offset, offset + 8)), "little")


def test_helpers_reject_width_mismatch(L):
    module, func, (entry, _), builder = new_function()
    value = ir.Constant(ir.IntType(64), 1)
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L._lift_semantic_helper("__ROL4__", ir.IntType(64), [value, ir.Constant(ir.IntType(32), 1)], builder)
    assert err.value.reason == "helper_width_mismatch"


def test_mixed_width_flag_helper_stays_opaque(L):
    module, func, (entry, _), builder = new_function()
    args = [ir.Constant(ir.IntType(32), 1), ir.Constant(ir.IntType(64), 2)]
    assert L._lift_semantic_helper("__CFSUB__", ir.IntType(8), args, builder) is None


def test_unknown_or_machine_specific_helpers_stay_opaque(L):
    module, func, (entry, _), builder = new_function()
    for name in ("__readeflags", "_bittest64", "va_arg", "qmemcpy", "__udf", "cpuid"):
        assert L._lift_semantic_helper(name, ir.IntType(64), [], builder) is None


def test_fence_and_dynamic_alloca(L):
    module, func, (entry, _), builder = new_function()
    assert L._lift_semantic_helper("_mm_mfence", ir.VoidType(), [], builder) is True
    ptr = L._lift_semantic_helper("alloca", ir.IntType(8).as_pointer(), [ir.Constant(ir.IntType(64), 24)], builder)
    text = str(module)
    assert "fence seq_cst" in text and "alloca i8, i64 24" in text and ptr.type == ir.IntType(8).as_pointer()
    verify(module)


def test_helper_call_in_microcode_is_expanded(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    with builder.goto_entry_block():
        slot = builder.alloca(ir.IntType(32), name="eax")
    target = types.SimpleNamespace(t=L.ida_hexrays.mop_h, helper="__ROL4__", size=0, value=None, dstr=lambda: "!__ROL4__")
    args = [ir.Constant(ir.IntType(32), 0x80000001), ir.Constant(ir.IntType(32), 1)]
    d = mop(L, "mop_f", value=([slot], args), size=4, f=callinfo(L, ir.IntType(32), retregs=1))
    L._reset_lift_state()
    value = L.lift_insn(insn(L, "m_call", l=target, d=d), entry, builder)
    assert value.type == ir.IntType(32) and "llvm.fshl.i32" in str(module)
    assert "__ROL4__" not in str(module) and not L.current_approx
    verify(module)
