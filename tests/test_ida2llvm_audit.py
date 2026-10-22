"""Regression tests for the ida2llvm fidelity audit (lifter at 5a4bfd6, task1/task2 pipeline).

Same approach as test_ida2llvm_microcode.py: the real Scripts/ida2llvm.py is imported with
stub IDA modules. Semantic checks build IR, run the paper's Oc pass list with `opt`, then
execute the optimized code with MCJIT, so they test behavior rather than only `verify`.
The source memory-layout contract (C01/C02) is covered by test_ida2llvm_target_layout.py, output
publication / resume / downstream state by test_pipeline_state.py, helpers by test_ida2llvm_helpers.py.
"""
import ctypes
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest
from llvmlite import binding, ir

from test_ida2llvm_microcode import L, callinfo, insn, mop, new_function, stub_operands, verify  # noqa: F401

SCRIPTS = Path(__file__).resolve().parents[1] / "Scripts"
sys.path.insert(0, str(SCRIPTS))
import task2_reoptimize  # noqa: E402

OC_PASSES = ",".join(task2_reoptimize.TASK2_CANONICALIZE_PASSES_BY_OPT_LEVEL["-Oc"])
OPT = shutil.which("opt-18") or shutil.which("opt")
needs_opt = pytest.mark.skipif(OPT is None, reason="LLVM opt not installed")


def run_oc(text: str) -> str:
    """Optimize textual IR with the Oc pipeline used for the paper's datasets."""
    result = subprocess.run([OPT, "-S", f"-passes={OC_PASSES}"], input=text, capture_output=True, text=True, check=True)
    return result.stdout


def jit(text: str):
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()
    module = binding.parse_assembly(text)
    module.verify()
    target_machine = binding.Target.from_default_triple().create_target_machine()
    engine = binding.create_mcjit_compiler(module, target_machine)
    engine.finalize_object()
    return engine


def host_module():
    module = ir.Module(context=ir.Context())
    module.triple = binding.get_default_triple()
    return module


@pytest.fixture
def fresh(L):
    L._reset_lift_state()
    yield L
    L._reset_lift_state()


_REAL_LIFT_MOP = [None]


@pytest.fixture(autouse=True)
def _remember_real_lift_mop(L):
    if _REAL_LIFT_MOP[0] is None:
        _REAL_LIFT_MOP[0] = L.lift_mop


@pytest.fixture
def real_mop_for(stub_operands, monkeypatch):
    """Stubbed operands, except the operand kinds listed here go through the real lift_mop."""
    L = stub_operands
    real, stub = _REAL_LIFT_MOP[0], L.lift_mop
    monkeypatch.setattr(L.ida_hexrays, "mcallarg_t", type("mcallarg_t", (), {}), raising=False)
    monkeypatch.setattr(L.ida_hexrays, "mop_addr_t", type("mop_addr_t", (), {}), raising=False)

    def make(*kinds):
        wanted = {getattr(L.ida_hexrays, k) for k in kinds}

        def dispatch(mop_, blk, builder, dest=False, knowntyp=None):
            if mop_.t in wanted:
                return real(mop_, blk, builder, dest, knowntyp)
            return stub(mop_, blk, builder, dest, knowntyp)
        monkeypatch.setattr(L, "lift_mop", dispatch)
        return L
    return make


# ------------------------------------------------------------- C05 registers / C06 C07 floats

def test_register_entry_value_is_frozen_undef_not_zero(fresh):
    L = fresh
    module, func, (entry, _), builder = new_function()
    reg = types.SimpleNamespace(size=8, r=0, dstr=lambda: "rax.8")
    L._get_register_storage(reg, entry, builder, is_read=True)
    stores = [i for i in entry.instructions if isinstance(i, ir.instructions.StoreInstr)]
    assert stores and stores[0].operands[0].opname == "freeze"
    assert L.current_approx == {"register_livein"}


def test_register_first_written_is_not_flagged(fresh):
    L = fresh
    module, func, (entry, _), builder = new_function()
    L._get_register_storage(types.SimpleNamespace(size=4, r=0, dstr=lambda: "eax.4"), entry, builder, is_read=False)
    assert not L.current_approx


@needs_opt
def test_register_livein_survives_oc_as_unknown(fresh):
    L = fresh
    module = host_module()
    func = ir.Function(module, ir.FunctionType(ir.IntType(64), []), "livein")
    func.lvars = {}
    entry = func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    slot = L._get_register_storage(types.SimpleNamespace(size=8, r=0, dstr=lambda: "rbx.8"), entry, builder)
    builder.position_at_end(entry)
    builder.ret(builder.load(slot))
    optimized = run_oc(str(module))
    assert "ret i64 0" not in optimized  # the old zero entry value folded to a constant here


def test_unreadable_float_constant_rejects(stub_operands, real_mop_for):
    L = real_mop_for("mop_fn")
    module, func, (entry, _), builder = new_function()
    operand = types.SimpleNamespace(t=L.ida_hexrays.mop_fn, size=8, fpc=object(), dstr=lambda: "#?.8")
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_mop(operand, entry, builder)
    assert err.value.reason == "fp_constant_unreadable"


@pytest.mark.parametrize("size", [2, 10, 16])
def test_unsupported_float_widths_reject_instead_of_narrowing(L, size):
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.float_type(size)
    assert err.value.reason == f"fp_width_{size}"


def test_wide_fp_register_load_moves_all_bytes(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    with builder.goto_entry_block():
        dst = builder.alloca(ir.IntType(128), name="result")
    sel = mop(L, "mop_r", size=2)
    sel.dstr = lambda: "cs.2"
    addr = mop(L, "mop_l", value=ir.Constant(ir.IntType(64), 0x1000))
    L.lift_insn(insn(L, "m_ldx", l=sel, r=addr, d=mop(L, "mop_l", value=dst, size=16), fp=True), entry, builder)
    loads = [i for i in entry.instructions if isinstance(i, ir.instructions.LoadInstr)]
    assert loads and loads[-1].type == ir.IntType(128)  # 16-byte q-register copy, not a double
    verify(module)


# ----------------------------------------------------------------------- C08 / C11 / P01 data

@pytest.fixture
def memory(fresh, monkeypatch):
    L = fresh
    bss = L.ida_segment.SEG_BSS
    state = {"segtype": 0, "data": None, "loaded": False}
    monkeypatch.setattr(L.ida_bytes, "get_bytes", lambda ea, size: state["data"])
    monkeypatch.setattr(L.ida_bytes, "is_loaded", lambda ea: state["loaded"])
    monkeypatch.setattr(L, "_get_segtype_cached", lambda ea: state["segtype"])
    monkeypatch.setattr(L.ida_segment, "getseg", lambda ea: None)
    return L, state, bss


def test_unreadable_data_is_unknown_not_zero(memory):
    L, state, _ = memory
    assert L._read_bytes(0x1000, 8) is None
    assert L.lift_stats["unresolved_data"] == 1 and L.unresolved_data_items[0]["ea"] == "0x1000"


def test_bss_data_is_zero(memory):
    L, state, bss = memory
    state["segtype"] = bss
    assert L._read_bytes(0x1000, 4) == b"\0" * 4 and L.lift_stats["unresolved_data"] == 0


def test_partially_loaded_data_is_unknown(memory):
    L, state, _ = memory
    state.update(data=b"\1\2\3\4", loaded=False)
    assert L._read_bytes(0x1000, 4) is None


def test_large_unknown_object_preserves_storage_size(L):
    class Unknown:
        def __getattr__(self, name):
            return lambda *args: False
    size = 1024 * 1024 + 1
    assert L._type_size(L._lift_tif_uncached(Unknown(), width=size)) == size


@needs_opt
@pytest.mark.parametrize("width", [16, 32, 64])
def test_integer_array_initializer_round_trips_through_oc(L, width):
    count = 64
    data = bytes((i * 37 + 11) & 0xFF for i in range(count * width // 8))
    module = host_module()
    elem = ir.IntType(width)
    g = ir.GlobalVariable(module, ir.ArrayType(elem, count), "table")
    g.initializer = L._constant_from_bytes(module, g.value_type, data, 0)
    g.global_constant = True
    func = ir.Function(module, ir.FunctionType(elem, [ir.IntType(32)]), "get")
    builder = ir.IRBuilder(func.append_basic_block())
    builder.ret(builder.load(builder.gep(g, [ir.Constant(ir.IntType(32), 0), func.args[0]])))
    engine = jit(run_oc(str(module)))
    ctype = {16: ctypes.c_uint16, 32: ctypes.c_uint32, 64: ctypes.c_uint64}[width]
    get = ctypes.CFUNCTYPE(ctype, ctypes.c_int32)(engine.get_function_address("get"))
    expected = struct.unpack(f"<{count}{ {16: 'H', 32: 'I', 64: 'Q'}[width] }", data)
    assert [get(i) for i in range(count)] == list(expected)


# --------------------------------------------------------------------------- C09 helpers

def test_helper_signature_comes_from_callinfo(L):
    module = ir.Module(context=ir.Context())
    info = types.SimpleNamespace(
        args=[types.SimpleNamespace(type=ir.IntType(32)), types.SimpleNamespace(type=ir.IntType(8))],
        return_type=ir.IntType(32), is_vararg=lambda: False, solid_args=2,
    )
    original = L.lift_tif
    try:
        L.lift_tif = lambda tif, width=-1: tif
        f = L.create_intrinsic_function(module, "__ROL4__", info)
    finally:
        L.lift_tif = original
    assert f.function_type == ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(8)])


def test_void_helper_stays_void(L):
    module = ir.Module(context=ir.Context())
    info = types.SimpleNamespace(args=[], return_type=ir.VoidType(), is_vararg=lambda: False, solid_args=0)
    original = L.lift_tif
    try:
        L.lift_tif = lambda tif, width=-1: tif
        f = L.create_intrinsic_function(module, "__writeeflags", info)
    finally:
        L.lift_tif = original
    assert isinstance(f.function_type.return_type, ir.VoidType)


def test_no_hardcoded_helper_signatures_remain(L):
    module = ir.Module(context=ir.Context())
    with pytest.raises(NotImplementedError):
        L.lift_intrinsic_function(module, "__ROL4__")
    pair = L.lift_intrinsic_function(module, "__PAIR64__")
    assert pair.function_type == ir.FunctionType(ir.IntType(64), [ir.IntType(32), ir.IntType(32)])


@pytest.mark.parametrize("name, tag", [
    ("__ROR2__", "opaque_helper:__ROR2__"),
    ("__readfsqword", "virtual_fs"),
    ("memcpy", None),
    ("alloca", "opaque_helper:alloca"),
])
def test_helper_approximation_tags(fresh, name, tag):
    fresh._note_helper(name)
    assert fresh.current_approx == ({tag} if tag else set())


def test_pair_destination_rejects(stub_operands, real_mop_for):
    L = real_mop_for("mop_p")
    module, func, (entry, _), builder = new_function()
    pair = types.SimpleNamespace(t=L.ida_hexrays.mop_p, size=8, dstr=lambda: "edx.4:eax.4")
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_mop(pair, entry, builder, dest=True)
    assert err.value.reason == "pair_destination"


# --------------------------------------------------------------------- C03 / C04 calls

def typed_call(L, ret, cinfo_ret, *, result_size=8, retregs=None, rets=()):
    module, func, (entry, _), builder = new_function()
    callee = ir.Function(module, ir.FunctionType(ret, [ir.IntType(32)]), "callee")
    target = mop(L, "mop_v", value=callee)
    nregs = len(rets) if retregs is None else retregs
    d = mop(L, "mop_f", value=(list(rets), [ir.Constant(ir.IntType(32), 1)]), size=result_size,
            f=callinfo(L, cinfo_ret, retregs=nregs))
    return module, entry, builder, insn(L, "m_call", l=target, d=d)


def test_void_declaration_takes_used_callinfo_return(stub_operands):
    L = stub_operands
    module, entry, builder, call_insn = typed_call(L, ir.VoidType(), ir.IntType(32))
    call = L.lift_insn(call_insn, entry, builder)
    assert call.type == ir.IntType(32)
    verify(module)


def test_return_abi_conflict_rejects_when_result_used(stub_operands):
    L = stub_operands
    module, entry, builder, call_insn = typed_call(L, ir.DoubleType(), ir.IntType(32))
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(call_insn, entry, builder)
    assert err.value.reason == "call_return_abi_conflict"


def test_return_abi_conflict_ignored_when_result_unused(stub_operands):
    L = stub_operands
    module, entry, builder, call_insn = typed_call(L, ir.DoubleType(), ir.IntType(32), result_size=0)
    assert L.lift_insn(call_insn, entry, builder).type == ir.DoubleType()
    verify(module)


def test_demanded_type_does_not_change_call_abi(stub_operands):
    L = stub_operands
    module, entry, builder, call_insn = typed_call(L, ir.IntType(32), ir.IntType(32))
    call = L.lift_insn(call_insn, entry, builder, result_type=ir.DoubleType())
    assert call.type == ir.IntType(32)  # the consumer converts afterwards; the call keeps its ABI


def test_direct_call_multi_retregs_rejects(stub_operands):
    L = stub_operands
    module, entry, builder, call_insn = typed_call(L, ir.IntType(64), ir.IntType(64), retregs=2)
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(call_insn, entry, builder)
    assert err.value.reason == "call_multi_retregs"


def test_aggregate_into_smaller_slot_rejects(stub_operands):
    L = stub_operands
    pair = ir.LiteralStructType([ir.IntType(64), ir.IntType(64)])
    module, func, (entry, _), builder = new_function()
    with builder.goto_entry_block():
        slot = builder.alloca(ir.IntType(64), name="rax")
    module, entry, builder, call_insn = typed_call(L, pair, pair, rets=[slot])
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(call_insn, entry, builder)
    assert err.value.reason == "call_aggregate_into_smaller_slot"


def test_aggregate_return_is_tagged(stub_operands, fresh):
    L = stub_operands
    pair = ir.LiteralStructType([ir.IntType(64), ir.IntType(64)])
    module, entry, builder, call_insn = typed_call(L, pair, pair)
    L.lift_insn(call_insn, entry, builder)
    assert "aggregate_return_call" in L.current_approx


def test_used_void_call_result_rejects(stub_operands, real_mop_for):
    L = real_mop_for("mop_d")
    module, func, (entry, _), builder = new_function()
    callee = ir.Function(module, ir.FunctionType(ir.VoidType(), []), "noret")
    d = mop(L, "mop_f", value=([], []), f=callinfo(L, ir.VoidType()))
    sub = insn(L, "m_call", l=mop(L, "mop_v", value=callee), d=d)
    operand = types.SimpleNamespace(t=L.ida_hexrays.mop_d, d=sub, size=8, dstr=lambda: "call $noret")
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_mop(operand, entry, builder)
    assert err.value.reason == "void_call_result_used"


# ------------------------------------------------------------- R03 / R07 / R10 / C10 control

def test_fs_selector_rejects(stub_operands):
    L = stub_operands
    module, func, (entry, _), builder = new_function()
    sel = mop(L, "mop_r", size=2)
    sel.dstr = lambda: "fs.2"
    ldx = insn(L, "m_ldx", l=sel, r=mop(L, "mop_n", value=ir.Constant(ir.IntType(64), 0x28)),
               d=mop(L, "mop_l", value=None))
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(ldx, entry, builder)
    assert err.value.reason == "segment_selector_fs"


def test_rejected_opcode_reported_before_operand_errors(stub_operands, monkeypatch):
    L = stub_operands

    def broken(*args, **kwargs):
        raise ValueError("operand decoding failed")
    monkeypatch.setattr(L, "lift_mop", broken)
    module, func, (entry, _), builder = new_function()
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L.lift_insn(insn(L, "m_ijmp"), entry, builder)
    assert err.value.reason == "m_ijmp"


def test_jtbl_without_default_is_tagged(stub_operands, fresh):
    L = stub_operands
    module, func, blocks, builder = new_function(nblocks=3)
    index = mop(L, "mop_n", value=ir.Constant(ir.IntType(32), 1), size=4)
    L.lift_insn(insn(L, "m_jtbl", l=index, r=mop(L, "mop_c", value={0: 1, 1: 2})), blocks[0], builder)
    assert "jtbl_assumed_exhaustive" in L.current_approx


def test_unterminated_block_rejects_instead_of_inventing_return(L):
    module, func, (entry, _), builder = new_function(ret=ir.IntType(32))
    builder.ret(ir.Constant(ir.IntType(32), 1))
    with pytest.raises(L.UnsupportedMicrocode) as err:
        L._require_terminated_blocks(func)
    assert err.value.reason == "unterminated_block"


def test_text_writer_does_not_repair_ir(L):
    out = []
    L._write_llvm_text("define i32 @missing() {\nentry:\n}", types.SimpleNamespace(write=out.append))
    assert "".join(out) == "define i32 @missing() {\nentry:\n}\n"


# ------------------------------------------------------------------------- task2 unsafe fallback

@needs_opt
def test_task2_sanitizer_is_function_scoped_and_store_survives_oc(tmp_path):
    src = tmp_path / "input.ll"
    src.write_text(
        'define void @broken() {\nentry:\n  %same = bitcast i64 1 to void ()\n  store void ()* %same, void ()** null\n  ret void\n}\n'
        'define void @valid(i64* %p) {\nentry:\n  %same = add i64 1, 2\n  store i64 %same, i64* %p\n  ret void\n}\n'
    )
    path, removed = task2_reoptimize.sanitize_illegal_function_bitcasts(str(src), str(tmp_path / "out.bc"))
    assert removed == 2  # the bitcast and its store in @broken only
    engine = jit(run_oc(Path(path).read_text()))
    cell = ctypes.c_int64(0)
    ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int64))(engine.get_function_address("valid"))(ctypes.byref(cell))
    assert cell.value == 3
    assert not list(tmp_path.glob("*.tmp.*"))
