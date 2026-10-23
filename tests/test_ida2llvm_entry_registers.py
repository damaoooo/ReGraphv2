"""Entry values of register lvars that Hex-Rays splits from an argument on the same register.

`cbz x1,1f; ret; 1: mov x0,#-1; ret` decompiles to separate lvars a1 (x0) and result (x0); the
path that does not write result returns the incoming x0. The dataflow finds such reads and the
entry block initializes them from the argument -- but only while the register still holds its
entry value (no intervening write by another lvar / raw register, no call that spoils the register). Machine-level
checks are in test_ida2llvm_shift_e2e.py::test_return_of_entry_register.
"""
import types

import pytest
from llvmlite import binding, ir

from test_ida2llvm_microcode import L  # noqa: F401

I64 = ir.IntType(64)


def lvar(name, reg, width=8, arg=False, result=False):
    loc = types.SimpleNamespace(is_reg1=lambda: reg is not None, reg1=lambda: reg)
    return types.SimpleNamespace(name=name, location=loc, width=width, is_arg_var=arg, is_result_var=result)


class Vars(list):
    def size(self):
        return len(self)


def fake_mba(L, monkeypatch, variables, blocks):
    """blocks: list of (events, successors, is_stop); events are _collect_lvar_accesses tuples."""
    accesses = {}
    mblocks = []
    for index, (events, succs, stop) in enumerate(blocks):
        head = None
        for event in reversed(events):
            head = types.SimpleNamespace(next=head)
            accesses[id(head)] = event
        mblocks.append(types.SimpleNamespace(
            head=head, type=L.ida_hexrays.BLT_STOP if stop else -1,
            nsucc=lambda s=succs: len(s), succ=lambda k, s=succs: s[k]))
    monkeypatch.setattr(L, "_collect_lvar_accesses", lambda ins, lvars: accesses[id(ins)])
    return types.SimpleNamespace(vars=Vars(variables), qty=len(mblocks), get_mblock=lambda i: mblocks[i])


def use(*idx, size=8):
    return ([(i, 0, size) for i in idx], False, [], [])


def define(idx, full=True):
    return ([], False, [(idx, 0, 8 if full else 4)], [])


CALL = ([], True, [], [])
A1, A2, RESULT = lvar("a1", 8, arg=True), lvar("a2", 16, arg=True), lvar("result", 8, result=True)


def test_result_returned_unwritten_on_one_path(L, monkeypatch):
    mba = fake_mba(L, monkeypatch, [A1, A2, RESULT], [
        ([], [1], False), ([use(1)], [2, 3], False), ([define(2)], [3], False), ([], [], True)])
    assert L._entry_register_reads(mba) == {2: False}


def test_result_written_on_every_path_needs_nothing(L, monkeypatch):
    mba = fake_mba(L, monkeypatch, [A1, A2, RESULT], [
        ([], [1, 2], False), ([define(2)], [3], False), ([define(2)], [3], False), ([], [], True)])
    assert L._entry_register_reads(mba) == {}


@pytest.mark.parametrize("clobber", [define(0), CALL, ([], False, [], [(8, 12)]), define(2, full=False)],
                         ids=["arg-redefined", "call", "raw-register-write", "partial-write"])
def test_entry_value_destroyed_before_read_is_dirty(L, monkeypatch, clobber):
    mba = fake_mba(L, monkeypatch, [A1, A2, RESULT], [
        ([clobber], [1, 2], False), ([define(2)], [2], False), ([], [], True)])
    assert L._entry_register_reads(mba) == {2: True}


def test_unrelated_register_write_keeps_entry_value(L, monkeypatch):
    mba = fake_mba(L, monkeypatch, [A1, A2, RESULT], [
        ([define(1), ([], False, [], [(16, 24)])], [1], False), ([], [], True)])
    assert L._entry_register_reads(mba) == {2: False}


def test_partial_write_then_same_bytes_read_is_defined(L, monkeypatch):
    """`mov #-1.4, i.4` ... `jz i.4, ...`: only the written low half is read (unpack_revinfo in openssl)."""
    i = lvar("i", 160)
    mba = fake_mba(L, monkeypatch, [A1, i], [
        ([define(1, full=False), use(1, size=4)], [1], False), ([use(1)], [], True)])
    assert L._entry_register_reads(mba) == {1: True}  # the full-width read in the stop block sees the unwritten half
    mba = fake_mba(L, monkeypatch, [A1, i], [([define(1, full=False), use(1, size=4)], [1], False), ([], [], True)])
    assert L._entry_register_reads(mba) == {}


def test_loop_back_edge_reaches_fixpoint(L, monkeypatch):
    # entry -> header -> {body (writes a1 on x0) -> header, exit (reads result)}
    mba = fake_mba(L, monkeypatch, [A1, A2, RESULT], [
        ([], [1], False), ([use(1)], [2, 3], False), ([define(0)], [1], False), ([], [], True)])
    # the direct path is clean; the state that comes back over the loop edge must still make it dirty
    assert L._entry_register_reads(mba) == {2: True}


def entry_function(L, variables, ret=I64):
    module = ir.Module(context=ir.Context())
    func = ir.Function(module, ir.FunctionType(ret, [I64, I64]), "f")
    func.lvars = {}
    builder = ir.IRBuilder(func.append_basic_block("entry"))
    func.lvars["funcresult"] = builder.alloca(ret, name="funcresult")
    builder.store(ir.Constant(ret, 0), func.lvars["funcresult"])
    for var, arg in zip([v for v in variables if v.is_arg_var], func.args):
        func.lvars[var.name] = builder.alloca(I64, name=var.name)
        builder.store(arg, func.lvars[var.name])
    for var in variables:
        if not var.is_arg_var and not var.is_result_var:
            func.lvars[var.name] = builder.alloca(ir.IntType(var.width * 8), name=var.name)
    return module, func, builder


def run_init(L, variables, reads, ret=I64, big_endian=False):
    L._reset_lift_state()
    L.current_approx.clear()
    saved = L.is_big_endian
    L.is_big_endian = big_endian
    try:
        module, func, builder = entry_function(L, variables, ret)
        L._init_entry_register_reads(func, Vars(variables), reads, builder)
    finally:
        L.is_big_endian = saved
    builder.ret(builder.load(func.lvars["funcresult"]))
    binding.parse_assembly(str(module)).verify()
    return str(func), set(L.current_approx)


def test_clean_read_takes_argument_entry_value(L):
    text, approx = run_init(L, [A1, A2, RESULT], {2: False})
    assert 'load i64, i64* %"a1"' in text and "freeze" not in text and not approx


def test_narrow_read_takes_low_bits(L):
    text, approx = run_init(L, [A1, A2, lvar("result", 8, width=4, result=True)], {2: False}, ir.IntType(32))
    assert "trunc i64" in text and not approx


def test_narrow_read_on_big_endian_is_not_guessed(L):
    text, approx = run_init(L, [A1, A2, lvar("result", 8, width=4, result=True)], {2: False}, ir.IntType(32),
                            big_endian=True)
    assert "freeze" in text and approx == {"register_livein"}


def test_dirty_read_rejects_function(L):
    with pytest.raises(L.UnsupportedMicrocode, match="register_reuse_unrecoverable"):
        run_init(L, [A1, A2, RESULT], {2: True})


def test_read_without_argument_on_register_is_unknown(L):
    text, approx = run_init(L, [A1, A2, lvar("v3", 152)], {2: False})
    assert "freeze" in text and approx == {"register_livein"}


def test_record_lvar_access_classifies_operands(L):
    acc = types.SimpleNamespace(uses=[], defs=[], reg_defs=[], call=False)
    h = L.ida_hexrays
    variables = Vars([A1, A2, RESULT])
    for op, target in [(types.SimpleNamespace(t=h.mop_l, l=types.SimpleNamespace(idx=1, off=0), size=8), False),
                       (types.SimpleNamespace(t=h.mop_l, l=types.SimpleNamespace(idx=2, off=0), size=8), True),
                       (types.SimpleNamespace(t=h.mop_l, l=types.SimpleNamespace(idx=0, off=4), size=4), True),
                       (types.SimpleNamespace(t=h.mop_r, r=8, size=4), True),
                       (types.SimpleNamespace(t=h.mop_r, r=16, size=8), False),
                       (types.SimpleNamespace(t=h.mop_f), False)]:
        L._record_lvar_access(acc, variables, op, target)
    assert acc.uses == [(1, 0, 8)] and acc.defs == [(2, 0, 8), (0, 4, 4)] and acc.reg_defs == [(8, 12)] and acc.call
