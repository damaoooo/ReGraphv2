"""Regression tests for ida2llvm microcode dispatch (calls, control flow, m_und/m_ext).

Scripts/ida2llvm.py is imported for real with stub IDA modules, so these run without
IDA. Only operand decoding (lift_mop) and type conversion (lift_tif) are replaced;
instruction dispatch, call-signature derivation and control-flow handling are the
actual implementation, and every module that should be valid is checked by LLVM.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from llvmlite import binding, ir

SCRIPT = Path(__file__).resolve().parents[1] / "Scripts" / "ida2llvm.py"
IDA_MODULES = (
    "idapro idc ida_idaapi ida_kernwin idautils ida_name ida_bytes ida_ida ida_funcs "
    "ida_typeinf ida_segment ida_nalt ida_hexrays ida_auto idaapi"
).split()


class _StubIdaModule(types.ModuleType):
    """Every missing attribute becomes a distinct int, enough for opcode/mop-type constants."""

    _counter = [10_000]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        _StubIdaModule._counter[0] += 1
        setattr(self, name, _StubIdaModule._counter[0])
        return getattr(self, name)


@pytest.fixture(scope="module")
def L():
    saved = {name: sys.modules.get(name) for name in IDA_MODULES}
    for name in IDA_MODULES:
        sys.modules[name] = _StubIdaModule(name)
    try:
        spec = importlib.util.spec_from_file_location("ida2llvm_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture
def stub_operands(L, monkeypatch):
    """lift_mop returns the operand's prepared value; lift_tif unwraps FakeTif."""
    monkeypatch.setattr(L, "lift_mop", lambda mop, blk, builder, dest=False, knowntyp=None: getattr(mop, "value", None))
    monkeypatch.setattr(L, "lift_tif", lambda tif, width=-1: tif.ty)
    L.idc.print_insn_mnem = lambda ea: "tzcnt"
    return L


class FakeTif:
    def __init__(self, ty):
        self.ty = ty

    def empty(self):
        return self.ty is None


class FakeList:
    def __init__(self, n):
        self.n = n

    def size(self):
        return self.n


def mop(L, kind, value=None, size=8, **extra):
    return types.SimpleNamespace(t=getattr(L.ida_hexrays, kind), value=value, size=size, dstr=lambda: kind, **extra)


def insn(L, opcode, l=None, r=None, d=None, fp=False):
    zero = mop(L, "mop_z")
    return types.SimpleNamespace(
        opcode=getattr(L.ida_hexrays, opcode), l=l or zero, r=r or zero, d=d or zero, ea=0x1000,
        is_fpinsn=lambda: fp, dstr=lambda: opcode,
    )


def callinfo(L, ret_ty, retregs=0, vararg=False, solid_args=0):
    return types.SimpleNamespace(
        return_type=FakeTif(ret_ty), retregs=FakeList(retregs), is_vararg=lambda: vararg, solid_args=solid_args,
    )


def new_function(nblocks=2, ret=ir.VoidType()):
    module = ir.Module(context=ir.Context())
    func = ir.Function(module, ir.FunctionType(ret, []), "probe")
    blocks = [func.append_basic_block(f"b{i}") for i in range(nblocks)]
    func.lvars = {}
    return module, func, blocks, ir.IRBuilder(blocks[0])


def verify(module):
    """Terminate any block the test left open, then require LLVM to parse and verify."""
    for func in module.functions:
        for blk in func.blocks:
            if blk.terminator is None:
                ir.IRBuilder(blk).ret_void()
    binding.parse_assembly(str(module)).verify()


# --------------------------------------------------------------------------- calls

CALL_TYPES = {
    "double": ir.DoubleType(),
    "float": ir.FloatType(),
    "i32": ir.IntType(32),
    "pointer": ir.IntType(8).as_pointer(),
    "void": ir.VoidType(),
}


@pytest.mark.parametrize("name", list(CALL_TYPES))
def test_toplevel_icall_takes_return_type_from_callinfo(stub_operands, name):
    L = stub_operands
    ret_ty = CALL_TYPES[name]
    module, func, (entry, _), builder = new_function()
    rets = []
    if not isinstance(ret_ty, ir.VoidType):
        with builder.goto_entry_block():
            rets.append(builder.alloca(ret_ty, name="retreg"))
    target = mop(L, "mop_d", value=ir.Constant(ir.IntType(64), 0x401000))
    d = mop(L, "mop_f", value=(rets, [ir.Constant(ir.IntType(32), 7)]), f=callinfo(L, ret_ty, retregs=len(rets)))

    call = L.lift_insn(insn(L, "m_icall", r=target, d=d), entry, builder)  # top level: no result_type

    assert call.type == ret_ty, f"indirect call must return {ret_ty}, not a guessed pointer"
    stores = [i for i in entry.instructions if isinstance(i, ir.instructions.StoreInstr)]
    assert len(stores) == len(rets)  # the single retreg receives the call result; void stores nothing
    verify(module)


def test_nested_icall_uses_demanded_result_type(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    target = mop(L, "mop_d", value=ir.Constant(ir.IntType(64), 0x401000))
    d = mop(L, "mop_f", value=([], []), f=callinfo(L, ir.DoubleType()))
    call = L.lift_insn(insn(L, "m_icall", r=target, d=d), entry, builder, result_type=ir.DoubleType())
    assert call.type == ir.DoubleType()
    verify(module)


def test_untyped_call_target_takes_signature_from_callinfo(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    target = mop(L, "mop_d", value=ir.Constant(ir.IntType(64), 0x401000))  # not a function pointer
    d = mop(L, "mop_f", value=([], [ir.Constant(ir.DoubleType(), 1.5)]), f=callinfo(L, ir.DoubleType()))
    call = L.lift_insn(insn(L, "m_call", l=target, d=d), entry, builder)
    assert call.type == ir.DoubleType()
    verify(module)


def test_variadic_callinfo_keeps_only_solid_args_fixed(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    fmt = ir.Constant(ir.IntType(8).as_pointer(), None)
    args = [fmt, ir.Constant(ir.IntType(32), 1), ir.Constant(ir.DoubleType(), 2.0)]
    target = mop(L, "mop_d", value=ir.Constant(ir.IntType(64), 0x401000))
    d = mop(L, "mop_f", value=([], args), f=callinfo(L, ir.IntType(32), vararg=True, solid_args=1))
    call = L.lift_insn(insn(L, "m_icall", r=target, d=d), entry, builder)
    ftype = call.callee.type.pointee
    assert ftype.var_arg and list(ftype.args) == [fmt.type]
    assert len(call.args) == 3
    verify(module)


@pytest.mark.parametrize("ret_ty, retregs, reason", [
    (ir.LiteralStructType([ir.IntType(64), ir.IntType(64)]), 1, "call_aggregate_return"),
    (ir.IntType(64), 2, "call_multi_retregs"),
    (None, 0, "call_unknown_return_type"),
])
def test_unsupported_call_returns_are_rejected(stub_operands, ret_ty, retregs, reason):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    target = mop(L, "mop_d", value=ir.Constant(ir.IntType(64), 0x401000))
    d = mop(L, "mop_f", value=([], []), f=callinfo(L, ret_ty, retregs=retregs))
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(insn(L, "m_icall", r=target, d=d), entry, builder)
    assert err.value.reason == reason


# --------------------------------------------------------------------- control flow

@pytest.mark.parametrize("opcode", ["m_ijmp", "m_push", "m_pop"])
def test_unmodelled_opcodes_reject_instead_of_falling_through(stub_operands, opcode):
    L = stub_operands
    module, func, (entry, nxt), builder = new_function()
    operand = mop(L, "mop_n", value=ir.Constant(ir.IntType(64), 1234))
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(insn(L, opcode, l=operand, r=operand, d=operand), entry, builder, nxt)
    assert err.value.reason == opcode
    assert entry.terminator is None  # no fabricated edge


def test_unresolved_goto_rejects(stub_operands):
    L = stub_operands
    module, func, (entry, nxt), builder = new_function()
    target = mop(L, "mop_r", value=ir.Constant(ir.IntType(64), 0))
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(insn(L, "m_goto", l=target), entry, builder, nxt)
    assert err.value.reason == "goto_unresolved_target"
    assert entry.terminator is None


def test_resolved_goto_branches_to_block(stub_operands):
    L = stub_operands
    module, func, (entry, nxt), builder = new_function()
    target = mop(L, "mop_b", b=1)
    L.lift_insn(insn(L, "m_goto", l=target), entry, builder)
    assert entry.terminator.operands[0] is nxt
    builder.position_at_end(nxt)
    verify(module)


def test_unresolved_conditional_jump_rejects(stub_operands):
    L = stub_operands
    module, func, (entry, nxt), builder = new_function()
    l = mop(L, "mop_n", value=ir.Constant(ir.IntType(32), 1), size=4)
    r = mop(L, "mop_n", value=ir.Constant(ir.IntType(32), 0), size=4)
    d = mop(L, "mop_r", value=ir.Constant(ir.IntType(64), 0))
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(insn(L, "m_jnz", l=l, r=r, d=d), entry, builder, nxt)
    assert err.value.reason == "jcc_unresolved_target"


def test_jtbl_without_default_targets_unreachable_after_body(stub_operands):
    L = stub_operands
    module, func, blocks, builder = new_function(nblocks=4)
    entry, case0, case1, exit_blk = blocks
    index = mop(L, "mop_n", value=ir.Constant(ir.IntType(32), 1), size=4)
    cases = mop(L, "mop_c", value={0: 1, 1: 2})
    switch = L.lift_insn(insn(L, "m_jtbl", l=index, r=cases), entry, builder)
    for blk in (case0, case1):
        builder.position_at_end(blk)
        builder.branch(exit_blk)
    builder.position_at_end(exit_blk)
    builder.ret_void()

    assert func.blocks[-1] is exit_blk  # mba block numbering untouched while lifting
    L._finalize_nodefault_switches(func, builder)
    default = switch.default
    assert default is func.blocks[-1] and default is not exit_blk
    assert isinstance(default.terminator, ir.instructions.Unreachable)
    verify(module)


def test_nop_is_still_a_noop(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    assert L.lift_insn(insn(L, "m_nop"), entry, builder) is None
    assert not entry.instructions


# ----------------------------------------------------------------------- m_und / m_ext

def test_und_writes_frozen_undef(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    with builder.goto_entry_block():
        slot = builder.alloca(ir.IntType(32), name="v1")
    L.lift_insn(insn(L, "m_und", d=mop(L, "mop_l", value=slot, size=4)), entry, builder)
    freeze = [i for i in entry.instructions if i.opname == "freeze"]
    store = [i for i in entry.instructions if isinstance(i, ir.instructions.StoreInstr)]
    assert len(freeze) == 1 and store[-1].operands[0] is freeze[0]  # an unknown value, not zero
    verify(module)


def test_ext_becomes_opaque_external_call(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    with builder.goto_entry_block():
        out = builder.alloca(ir.IntType(32), name="ecx")
    l = mop(L, "mop_r", value=ir.Constant(ir.IntType(32), 8), size=4)
    d = mop(L, "mop_r", value=out, size=4)
    call = L.lift_insn(insn(L, "m_ext", l=l, d=d), entry, builder)
    assert call.callee.name == "__ida_ext_tzcnt" and call.callee.is_declaration
    store = [i for i in entry.instructions if isinstance(i, ir.instructions.StoreInstr)]
    assert store[-1].operands[0] is call and store[-1].operands[1] is out
    verify(module)


def test_rejection_reason_is_a_runtime_error(L):
    err = L.UnsupportedMicrocode("m_ijmp", "ijmp cs.2, a8.4")
    assert isinstance(err, RuntimeError) and not isinstance(err, NotImplementedError)
    assert err.reason == "m_ijmp" and "ijmp" in str(err)
