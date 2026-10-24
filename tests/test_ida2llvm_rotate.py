"""RotateFix's matcher (_BlockValues) on hand-written preoptimized microcode.

clang -O0 leaves ROTATE(x, n) as (x << n) | (x >> (W - n)); Hex-Rays' local optimization then takes hours
on OpenSSL's unrolled SM3. The matcher must recognise the shapes seen in real clang output (x86/x64/arm32
and arm64, see the comments) and nothing that is not exactly a rotate of one value. The rewrite itself and
its semantics are covered on real binaries (lift, recompile, run against an SM3 reference).
"""
import types

import pytest

from test_ida2llvm_microcode import L  # noqa: F401


class Mop:
    def __init__(self, t, size, **fields):
        self.t, self.size = t, size
        for name, value in fields.items():
            setattr(self, name, value)


@pytest.fixture
def mc(L, monkeypatch):
    """Fake microcode builders bound to the stub ida_hexrays constants."""
    h = L.ida_hexrays
    monkeypatch.setattr(h, "mop_t", lambda mop: mop, raising=False)
    monkeypatch.setattr(L, "_mc_number", lambda value, size: Mop(h.mop_n, size, nnn=types.SimpleNamespace(value=value)))
    monkeypatch.setattr(L, "_mc_expr", lambda ea, opcode, left, right, size: Mop(
        h.mop_d, size, d=types.SimpleNamespace(opcode=opcode, l=left, r=right or Mop(h.mop_z, 0), d=Mop(h.mop_z, size))))
    L._BlockValues.PURE = None  # rebuilt from this stub module's constants

    def reg(r, size=4):
        return Mop(h.mop_r, size, r=r)

    def stk(off, size=4):
        return Mop(h.mop_S, size, s=types.SimpleNamespace(off=off))

    def num(value, size=1):
        return Mop(h.mop_n, size, nnn=types.SimpleNamespace(value=value))

    def ins(opcode, l, r=None, d=None, call=False):
        return types.SimpleNamespace(opcode=getattr(h, "m_" + opcode), l=l, r=r or Mop(h.mop_z, 0),
                                     d=d or Mop(h.mop_z, 0), ea=0x1000, contains_call=lambda helpers: call)

    def rotates(insns):
        """(n, x) for each `or` the matcher turns into __ROL__(x, n), x as the operand it would pass."""
        values, found = L._BlockValues(), []
        for one in insns:
            result, clobbered = values.value_of(one), values.clobbers(one)
            match = values.rotate(one)
            if match is not None:
                found.append((match[1], values.materialize(match[0], one.ea)))
            values.step(one, result, clobbered)
        return found

    return types.SimpleNamespace(h=h, reg=reg, stk=stk, num=num, ins=ins, rotates=rotates)


EAX, ECX, W0, W16, W17, W18 = 0, 8, 100, 164, 168, 172


def x64_rotate(mc, src, n, left_as_mul=True):
    # mov %var.4, eax; mul #2^n, eax, eax (preoptimization's form of shl); mov %var.4, ecx;
    # shr ecx, #(32-n), ecx; or ecx, eax, eax
    left = (mc.ins("mul", mc.num(1 << n, 4), mc.reg(EAX), mc.reg(EAX)) if left_as_mul
            else mc.ins("shl", mc.reg(EAX), mc.num(n), mc.reg(EAX)))
    return [mc.ins("mov", src, d=mc.reg(EAX)), left, mc.ins("mov", src, d=mc.reg(ECX)),
            mc.ins("shr", mc.reg(ECX), mc.num(32 - n), mc.reg(ECX)), mc.ins("or", mc.reg(ECX), mc.reg(EAX), mc.reg(EAX))]


@pytest.mark.parametrize("left_as_mul", [True, False])
def test_x86_shape_is_a_rotate_of_the_stack_variable(mc, left_as_mul):
    [(n, x)] = mc.rotates(x64_rotate(mc, mc.stk(0x24), 12, left_as_mul))
    assert n == 12 and x.t == mc.h.mop_S and x.s.off == 0x24


def test_arm64_shape_with_counts_in_registers(mc):
    # mov #0x14.1, w17.1; mov #0xC.1, w18.1; mov %var_34.4, w0.4; shl w0.4, w18.1, w18.4;
    # mov %var_34.4, w0.4; shr w0.4, w17.1, w17.4; or w18.4, w17.4, w17.4
    var = mc.stk(0x34)
    insns = [mc.ins("mov", mc.num(0x14), d=mc.reg(W17, 1)), mc.ins("mov", mc.num(0xC), d=mc.reg(W18, 1)),
             mc.ins("mov", var, d=mc.reg(W0)), mc.ins("shl", mc.reg(W0), mc.reg(W18, 1), mc.reg(W18)),
             mc.ins("mov", var, d=mc.reg(W0)), mc.ins("shr", mc.reg(W0), mc.reg(W17, 1), mc.reg(W17)),
             mc.ins("or", mc.reg(W18), mc.reg(W17), mc.reg(W17))]
    [(n, x)] = mc.rotates(insns)
    assert n == 12 and x.s.off == 0x34


def test_low_half_of_a_zero_extension_and_of_a_wide_constant(mc):
    # arm64: xdu %var_38.4, x0.8 then shr w0.4, ...; and a count kept as mov #7.8, x16.8 read as w16.1
    var = mc.stk(0x38)
    insns = [mc.ins("mov", mc.num(7, 8), d=mc.reg(W16, 8)), mc.ins("mov", var, d=mc.reg(W18)),
             mc.ins("shl", mc.reg(W18), mc.reg(W16, 1), mc.reg(W18)), mc.ins("xdu", var, d=mc.reg(W0, 8)),
             mc.ins("shr", mc.reg(W0), mc.num(25), mc.reg(W17)), mc.ins("or", mc.reg(W18), mc.reg(W17), mc.reg(W17))]
    [(n, x)] = mc.rotates(insns)
    assert n == 7 and x.s.off == 0x38


def test_recomputed_operand_is_rebuilt_when_no_register_holds_it(mc):
    # P1(a ^ b): clang -O0 computes a ^ b once for each shift and overwrites both registers
    a, b = mc.stk(0x40), mc.stk(0x44)
    insns = [mc.ins("mov", a, d=mc.reg(EAX)), mc.ins("xor", mc.reg(EAX), b, mc.reg(EAX)),
             mc.ins("shl", mc.reg(EAX), mc.num(15), mc.reg(EAX)),
             mc.ins("mov", a, d=mc.reg(ECX)), mc.ins("xor", mc.reg(ECX), b, mc.reg(ECX)),
             mc.ins("shr", mc.reg(ECX), mc.num(17), mc.reg(ECX)), mc.ins("or", mc.reg(EAX), mc.reg(ECX), mc.reg(EAX))]
    [(n, x)] = mc.rotates(insns)
    assert n == 15 and x.t == mc.h.mop_d and x.d.opcode == mc.h.m_xor
    assert {x.d.l.s.off, x.d.r.s.off} == {0x40, 0x44}


def test_two_different_values_are_not_a_rotate(mc):
    insns = x64_rotate(mc, mc.stk(0x24), 12)
    insns[2] = mc.ins("mov", mc.stk(0x28), d=mc.reg(ECX))
    assert mc.rotates(insns) == []


@pytest.mark.parametrize("clobber", ["store_to_variable", "store_through_pointer", "call"])
def test_a_write_between_the_reads_breaks_the_match(mc, clobber):
    insns = x64_rotate(mc, mc.stk(0x24), 12)
    write = {"store_to_variable": mc.ins("mov", mc.num(0, 4), d=mc.stk(0x24)),
             "store_through_pointer": mc.ins("stx", mc.reg(EAX), mc.reg(16), mc.reg(24)),
             "call": mc.ins("mov", mc.reg(24), d=mc.reg(16), call=True)}[clobber]
    insns.insert(2, write)
    assert mc.rotates(insns) == []


def test_arithmetic_shift_or_wrong_count_is_not_a_rotate(mc):
    sar = x64_rotate(mc, mc.stk(0x24), 12)
    sar[3] = mc.ins("sar", mc.reg(ECX), mc.num(20), mc.reg(ECX))
    wrong = x64_rotate(mc, mc.stk(0x24), 12)
    wrong[3] = mc.ins("shr", mc.reg(ECX), mc.num(19), mc.reg(ECX))
    assert mc.rotates(sar) == [] and mc.rotates(wrong) == []


def test_byte_assembly_with_shifts_is_not_a_rotate(mc):
    # LD(): (p[0] << 24) | (p[1] << 16) ... combines different bytes with left shifts only
    insns = [mc.ins("xdu", mc.stk(0x10, 1), d=mc.reg(EAX)), mc.ins("shl", mc.reg(EAX), mc.num(24), mc.reg(EAX)),
             mc.ins("xdu", mc.stk(0x11, 1), d=mc.reg(ECX)), mc.ins("shr", mc.reg(ECX), mc.num(8), mc.reg(ECX)),
             mc.ins("or", mc.reg(EAX), mc.reg(ECX), mc.reg(EAX))]
    assert mc.rotates(insns) == []


def test_stack_low_part_is_not_inferred(mc):
    # the low half of a wider stack slot depends on byte order, so it is a new unknown value
    insns = [mc.ins("mov", mc.stk(0x24), d=mc.reg(EAX)), mc.ins("xdu", mc.reg(EAX), d=mc.stk(0x30, 8)),
             mc.ins("shl", mc.reg(EAX), mc.num(12), mc.reg(EAX)), mc.ins("mov", mc.stk(0x30), d=mc.reg(ECX)),
             mc.ins("shr", mc.reg(ECX), mc.num(20), mc.reg(ECX)), mc.ins("or", mc.reg(EAX), mc.reg(ECX), mc.reg(EAX))]
    assert mc.rotates(insns) == []


def test_rotate_fix_is_off_by_default_and_names_functions(L, monkeypatch):
    monkeypatch.delenv("REGRAPH_ROTATE_FIX", raising=False)
    assert L._rotate_fix_from_env() == frozenset() and L._install_rotate_fix(frozenset()) is None
    monkeypatch.setenv("REGRAPH_ROTATE_FIX", " sm3_block_data_order,, blake2b_compress")
    assert L._rotate_fix_from_env() == {"sm3_block_data_order", "blake2b_compress"}
