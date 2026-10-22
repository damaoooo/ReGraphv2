"""Source memory-layout contract for ida2llvm (audit C01/C02).

In the default `source` mode the lifted module carries the source binary's triple and data
layout, and the lifter's own size/offset computation uses that same layout. Cross-target
checks do not run foreign code on the host: `llc` compiles the module for the source triple
and the emitted section bytes / constant-folded offsets are compared with the source bytes,
i.e. a target-layout byte oracle.
"""
import shutil
import subprocess
from pathlib import Path

import pytest
from llvmlite import binding, ir

from test_ida2llvm_microcode import L  # noqa: F401

CLANG = shutil.which("clang-18") or shutil.which("clang")
LLC = shutil.which("llc-18") or shutil.which("llc")
OBJCOPY = shutil.which("llvm-objcopy-18") or shutil.which("llvm-objcopy")
needs_clang = pytest.mark.skipif(CLANG is None, reason="clang not installed")
needs_llc = pytest.mark.skipif(LLC is None or OBJCOPY is None, reason="llc/llvm-objcopy not installed")

TARGETS = [  # (IDA processor name, bits, big endian)
    ("metapc", 64, False), ("metapc", 32, False), ("ARM", 32, False), ("ARM", 64, False),
    ("mipsb", 32, True), ("mipsl", 32, False), ("mipsb", 64, True), ("mipsl", 64, False),
]


@pytest.fixture
def target(L, monkeypatch):
    """Put the lifter in the state it has after opening a database for the given source target."""
    def use(proc, bits, big_endian, mode="source"):
        L._reset_lift_state()  # fresh LLVM context per module
        monkeypatch.setattr(L, "ptrsize", bits)
        monkeypatch.setattr(L, "is_big_endian", big_endian)
        monkeypatch.setattr(L, "byteorder", "big" if big_endian else "little")
        monkeypatch.setattr(L.ida_ida, "inf_get_procname", lambda: proc, raising=False)
        monkeypatch.setattr(L.ida_ida, "inf_get_filetype", lambda: L.ida_ida.f_ELF, raising=False)
        monkeypatch.setattr(L, "_source_elf_header", lambda: {
            "bits": bits, "endian": "big" if big_endian else "little", "type": 3,
            "machine": 40 if proc == "ARM" else 8, "flags": 0x05000200 if proc == "ARM" else 0,
        })
        # pointer initializers that resolve to no symbol stay raw source addresses (inttoptr)
        monkeypatch.setattr(L.ida_funcs, "get_func", lambda ea: None, raising=False)
        monkeypatch.setattr(L.ida_bytes, "get_item_head", lambda ea: ea, raising=False)
        return L.BIN2LLVMController(mode)
    yield use
    L._reset_lift_state()


def emitted_section(module: ir.Module, tmp_path: Path) -> bytes:
    """Compile for the module's own triple and return the bytes of the `.oracle` section."""
    src, obj, out = tmp_path / "m.ll", tmp_path / "m.o", tmp_path / "m.bin"
    src.write_text(str(module))
    subprocess.run([LLC, f"-mtriple={module.triple}", "-filetype=obj", str(src), "-o", str(obj)], check=True)
    subprocess.run([OBJCOPY, "-O", "binary", "--only-section=.oracle", str(obj), str(out)], check=True)
    return out.read_bytes()


@needs_clang
@pytest.mark.parametrize("key", [
    ("x86", 64, "little"), ("x86", 32, "little"), ("arm", 32, "little"), ("arm", 64, "little"),
    ("mips", 32, "big"), ("mips", 32, "little"), ("mips", 64, "big"), ("mips", 64, "little"),
    ("ppc", 32, "big"), ("ppc", 64, "big"), ("ppc", 64, "little"), ("riscv", 32, "little"), ("riscv", 64, "little"),
])
def test_source_layout_table_matches_clang(L, key):
    triple, layout = L.SOURCE_TARGETS[key]
    out = subprocess.run([CLANG, f"--target={triple}", "-S", "-emit-llvm", "-x", "c", "/dev/null", "-o", "-"],
                         capture_output=True, text=True, check=True).stdout
    assert f'target datalayout = "{layout}"' in out
    binding.create_target_data(layout)  # llvmlite (bundled LLVM) parses the same string


@pytest.mark.parametrize("proc, bits, big_endian", TARGETS)
def test_source_mode_uses_source_triple_and_layout(target, proc, bits, big_endian):
    ctrl = target(proc, bits, big_endian)
    assert ctrl.memory_contract == "source-layout"
    assert ctrl.m.data_layout.startswith("E" if big_endian else "e")
    assert ("p:32:32" in ctrl.m.data_layout) == (bits == 32)


def test_source_target_rejects_unknown_or_non_elf(L):
    with pytest.raises(L.UnsupportedTarget):
        L.source_target_spec("z80", 16, False)
    with pytest.raises(L.UnsupportedTarget):
        L.source_target_spec("metapc", 64, False, is_elf=False)
    with pytest.raises(L.UnsupportedTarget):
        L.source_target_spec("ARM", 64, True)  # no big-endian AArch64 entry: fail instead of guessing


def test_source32_packed_layout_matches_output_layout(target, L):
    ctrl = target("metapc", 32, False)
    typ = ctrl.m.context.get_identified_type("audit_pointer_struct")
    typ.packed = True
    typ.set_body(ir.IntType(8).as_pointer(), ir.IntType(32))
    ir.GlobalVariable(ctrl.m, typ, "layout_probe")
    parsed = binding.parse_assembly(str(ctrl.m))
    actual = binding.create_target_data(ctrl.m.data_layout).get_abi_size(parsed.get_struct_type("audit_pointer_struct"))
    assert actual == L._type_size(typ) == 8


def test_host_mode_is_marked_unconverted(target, L):
    ctrl = target("metapc", 32, False, mode="host")
    assert ctrl.memory_contract == "host-compat-unconverted"
    # the limitation the contract name documents: host output layout, source-width sizing
    assert L._type_size(ir.IntType(8).as_pointer()) == 4
    assert ir.IntType(8).as_pointer().get_abi_size(binding.create_target_data(ctrl.m.data_layout)) == 8


@needs_llc
@pytest.mark.parametrize("proc, bits, big_endian", TARGETS)
def test_initializer_bytes_match_source_on_target(target, L, tmp_path, proc, bits, big_endian):
    ctrl = target(proc, bits, big_endian)
    pointer_bytes = bits // 8
    struct_t = ctrl.m.context.get_identified_type("oracle_struct")
    struct_t.packed = True
    struct_t.set_body(ir.IntType(8).as_pointer(), ir.IntType(16), ir.ArrayType(ir.IntType(32), 3), ir.IntType(64))
    size = L._type_size(struct_t)
    assert size == pointer_bytes + 2 + 12 + 8
    data = bytes(range(1, size + 1))
    g = ir.GlobalVariable(ctrl.m, struct_t, "oracle")
    g.initializer = L._constant_from_bytes(ctrl.m, struct_t, data, 0)
    g.section = ".oracle"
    assert emitted_section(ctrl.m, tmp_path) == data  # same bytes, same offsets, same order as the source


@needs_llc
@pytest.mark.parametrize("proc, bits, big_endian", TARGETS)
def test_member_offsets_and_strides_match_target_layout(target, L, tmp_path, proc, bits, big_endian):
    ctrl = target(proc, bits, big_endian)
    long_double = L._sized_member_type(ir.IntType(96), 12)  # i386 long double: 12-byte slots
    struct_t = ctrl.m.context.get_identified_type("offset_struct")
    struct_t.packed = True
    members = [ir.IntType(8), ir.IntType(8).as_pointer(), ir.IntType(16), ir.ArrayType(long_double, 2), ir.DoubleType(), ir.IntType(64)]
    struct_t.set_body(*members)
    expected, offset = [], 0
    for member in members:
        expected.append(offset)
        offset += L._type_size(member)
    word = ir.IntType(64)
    null = ir.Constant(struct_t.as_pointer(), None)
    zero = ir.Constant(ir.IntType(32), 0)
    values = [null.gep([zero, ir.Constant(ir.IntType(32), k)]).ptrtoint(word) for k in range(len(members))]
    values.append(null.gep([ir.Constant(ir.IntType(32), 1)]).ptrtoint(word))  # sizeof
    g = ir.GlobalVariable(ctrl.m, ir.ArrayType(word, len(values)), "offsets")
    g.initializer = ir.Constant.literal_array(values)
    g.section = ".oracle"
    raw = emitted_section(ctrl.m, tmp_path)
    folded = [int.from_bytes(raw[i:i + 8], "big" if big_endian else "little") for i in range(0, len(raw), 8)]
    assert folded == expected + [offset]


@needs_llc
@pytest.mark.parametrize("proc, bits, big_endian", [("mipsb", 32, True), ("metapc", 32, False)])
def test_host_mode_byte_view_differs_from_source(target, L, tmp_path, proc, bits, big_endian):
    """Negative control: the same oracle detects the host-compat limitation (so the source tests are sensitive)."""
    ctrl = target(proc, bits, big_endian, mode="host")
    struct_t = ctrl.m.context.get_identified_type("host_struct")
    struct_t.packed = True
    struct_t.set_body(ir.IntType(8).as_pointer(), ir.IntType(32))
    data = bytes(range(1, L._type_size(struct_t) + 1))
    g = ir.GlobalVariable(ctrl.m, struct_t, "oracle")
    g.initializer = L._constant_from_bytes(ctrl.m, struct_t, data, 0)
    g.section = ".oracle"
    assert emitted_section(ctrl.m, tmp_path) != data
