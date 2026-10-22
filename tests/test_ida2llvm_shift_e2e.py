"""Opt-in real IDA -> LLVM -> Oc -> machine-code differential shift tests.

Run with REGRAPH_IDA_SMOKE=1, IDADIR set, and unicorn installed in the test
environment. No original dataset/IDB is modified; all fixtures live in tmp_path.

Example (from the repository root):
  REGRAPH_IDA_SMOKE=1 IDADIR=$HOME/ida-pro-9.4 uv run --locked --extra pipeline \
      --with unicorn python -m pytest tests/test_ida2llvm_shift_e2e.py -q
"""
import json
import os
import random
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

if os.environ.get("REGRAPH_IDA_SMOKE") != "1":
    pytest.skip("set REGRAPH_IDA_SMOKE=1 for licensed IDA integration tests", allow_module_level=True)
U = pytest.importorskip("unicorn")
from unicorn import x86_const as X, arm_const as A, arm64_const as A64, mips_const as M

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from task2_reoptimize import canonicalize_passes_for_opt_level

TARGETS = {
    "x64": ("x86_64-linux-gnu", U.UC_ARCH_X86, U.UC_MODE_64, 64),
    "x86": ("i386-linux-gnu", U.UC_ARCH_X86, U.UC_MODE_32, 32),
    "arm32": ("armv7-linux-gnueabi", U.UC_ARCH_ARM, U.UC_MODE_ARM, 32),
    "a64": ("aarch64-linux-gnu", U.UC_ARCH_ARM64, U.UC_MODE_ARM, 64),
    "mips32": ("mips-linux-gnu", U.UC_ARCH_MIPS, U.UC_MODE_MIPS32 | U.UC_MODE_BIG_ENDIAN, 32),
    "mips64": ("mips64-linux-gnuabi64", U.UC_ARCH_MIPS, U.UC_MODE_MIPS64 | U.UC_MODE_BIG_ENDIAN, 64),
}


def run(cmd, **kwargs):
    return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True, timeout=120, **kwargs).stdout


def assembly(arch):
    widths = [8, 16, 32, 64] if arch == "x64" else [32, 64] if arch in ("a64", "mips64") else [32]
    lines = [".text"]
    if arch in ("x86", "x64"):
        lines.append(".intel_syntax noprefix")
    if arch.startswith("mips"):
        lines.append(".set noreorder")
    names = []
    for width in widths:
        for op in ("shl", "shr", "sar"):
            name = f"{op}{width}"
            names.append((name, width))
            lines.extend([f".globl {name}", f".type {name},%function", f"{name}:"])
            if arch == "x64":
                reg = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]
                lines.extend(["mov rax,rdi", "mov ecx,esi", f"{op} {reg},cl"])
                if width < 32:
                    lines.append(f"movzx eax,{reg}")
                lines.append("ret")
            elif arch == "x86":
                lines.extend(["mov eax,[esp+4]", "mov ecx,[esp+8]", f"{op} eax,cl", "ret"])
            elif arch in ("arm32", "a64"):
                opcode = {"shl": "lsl", "shr": "lsr", "sar": "asr"}[op]
                reg = "r" if arch == "arm32" else "w" if width == 32 else "x"
                lines.extend([f"{opcode} {reg}0,{reg}0,{reg}1", "bx lr" if arch == "arm32" else "ret"])
            else:
                opcode = ("d" if width == 64 else "") + {"shl": "sllv", "shr": "srlv", "sar": "srav"}[op]
                lines.extend(["jr $ra", f"{opcode} $v0,$a0,$a1"])
            lines.append(f".size {name},.-{name}")
    return "\n".join(lines) + "\n", names


def object_text(obj, tmp_path):
    blob = tmp_path / (obj.name + ".bin")
    run(["llvm-objcopy", "-O", "binary", "--only-section=.text", obj, blob])
    symbols = {}
    for line in run(["llvm-nm", "--defined-only", obj]).splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] in ("T", "t"):
            symbols[fields[2]] = int(fields[0], 16)
    return blob.read_bytes(), symbols


def execute(arch, code, offset, x, n, width):
    _, machine, mode, bits = TARGETS[arch]
    uc = U.Uc(machine, mode)
    if arch == 'mips64':
        # clang's mips64-linux-gnuabi64 default includes R2 instructions (DEXT).
        uc.ctl_set_cpu_model(M.UC_CPU_MIPS64_MIPS64R2_GENERIC)
    base, stack, stop = 0x100000, 0x800000, 0x900000
    uc.mem_map(base, 0x100000)
    uc.mem_write(base, code)
    uc.mem_map(stack - 0x10000, 0x20000)
    uc.mem_map(stop, 0x1000)
    mask = (1 << bits) - 1
    if arch == "x64":
        regs = [(X.UC_X86_REG_RDI, x & mask), (X.UC_X86_REG_RSI, n), (X.UC_X86_REG_RSP, stack - 8)]
        uc.mem_write(stack - 8, stop.to_bytes(8, "little"))
        result = X.UC_X86_REG_RAX
    elif arch == "x86":
        regs = [(X.UC_X86_REG_ESP, stack - 12)]
        uc.mem_write(stack - 12, stop.to_bytes(4, "little") + (x & mask).to_bytes(4, "little") + n.to_bytes(4, "little"))
        result = X.UC_X86_REG_EAX
    elif arch == "arm32":
        regs = [(A.UC_ARM_REG_R0, x & mask), (A.UC_ARM_REG_R1, n), (A.UC_ARM_REG_SP, stack), (A.UC_ARM_REG_LR, stop)]
        if width == 64:
            regs = [(A.UC_ARM_REG_R0, x & mask), (A.UC_ARM_REG_R1, x >> 32),
                    (A.UC_ARM_REG_R2, n & mask), (A.UC_ARM_REG_R3, n >> 32),
                    (A.UC_ARM_REG_SP, stack), (A.UC_ARM_REG_LR, stop)]
        result = A.UC_ARM_REG_R0
    elif arch == "a64":
        regs = [(A64.UC_ARM64_REG_X0, x & mask), (A64.UC_ARM64_REG_X1, n), (A64.UC_ARM64_REG_SP, stack), (A64.UC_ARM64_REG_LR, stop)]
        result = A64.UC_ARM64_REG_X0
    else:
        # MIPS64 word operations require sign-extended 32-bit input registers.
        # Zero-extending 0xffffffff violates the ISA operand restriction and
        # produces a false mismatch in Unicorn's SRAV implementation.
        if bits == 64 and width == 32:
            x = (x & 0x7fffffff) - (x & 0x80000000)
        regs = [(M.UC_MIPS_REG_A0, x & mask), (M.UC_MIPS_REG_A1, n), (M.UC_MIPS_REG_SP, stack),
                (M.UC_MIPS_REG_RA, stop), (M.UC_MIPS_REG_T9, base + offset)]
        result = M.UC_MIPS_REG_V0
    for reg, value in regs:
        uc.reg_write(reg, value)
    uc.emu_start(base + offset, stop, timeout=100000, count=10000)
    pc = {"x64": X.UC_X86_REG_RIP, "x86": X.UC_X86_REG_EIP, "arm32": A.UC_ARM_REG_PC,
          "a64": A64.UC_ARM64_REG_PC}.get(arch, M.UC_MIPS_REG_PC)
    assert uc.reg_read(pc) == stop, "emulation failed to return"
    value = uc.reg_read(result)
    if arch == 'arm32' and width == 64:
        value |= uc.reg_read(A.UC_ARM_REG_R1) << 32
    return value & ((1 << width) - 1)


@pytest.mark.parametrize("arch", TARGETS)
def test_real_shifts_before_after_oc(arch, tmp_path):
    text, names = assembly(arch)
    check_machine_equivalence(arch, tmp_path, text, names)


def test_arm_carry_before_after_oc(tmp_path):
    lines = ['.syntax unified', '.text']
    names = []
    for shift in ('lsl', 'lsr', 'asr'):
        for operation in ('carry', 'adc', 'sbc', 'rsc'):
            name = operation + '_' + shift
            names.append((name, 32))
            lines.extend([f'.global {name}', f'.type {name},%function', f'{name}:', 'cmp r0,#1'])
            if operation == 'carry':
                lines.extend([f'{shift}s r2,r0,r1', 'mov r0,#0', 'adc r0,r0,#0'])
            else:
                lines.append(f'{operation}s r0,r0,r0,{shift} r1')
            lines.extend(['bx lr', f'.size {name},.-{name}'])
    check_machine_equivalence('arm32', tmp_path, '\n'.join(lines) + '\n', names)


def test_real_dataset_aeabi_dadd(tmp_path):
    """Optional real-binary regression: set REGRAPH_ARM_OPENSSL_IDB to the ARM32 IDB.

    Compares original instructions with the lifted function before/after Oc,
    including its full 64-bit argument/return PCS, not just LLVM validity.
    """
    source = os.environ.get('REGRAPH_ARM_OPENSSL_IDB')
    if not source:
        pytest.skip('set REGRAPH_ARM_OPENSSL_IDB for the real OpenSSL regression')
    ida = os.environ['IDADIR']
    db = tmp_path / 'input.i64'
    shutil.copy2(source, db)
    raw = tmp_path / 'original.bin'
    env = dict(os.environ, PYTHONPATH=os.path.join(ida, 'idalib', 'python'))
    exporter = '''import idapro
import ida_bytes, ida_funcs, ida_name, ida_idaapi, sys
from pathlib import Path
idapro.open_database(sys.argv[1], False)
ea = ida_name.get_name_ea(ida_idaapi.BADADDR, '__aeabi_dadd')
assert ea != ida_idaapi.BADADDR
f = ida_funcs.get_func(ea)
Path(sys.argv[2]).write_bytes(ida_bytes.get_bytes(f.start_ea, f.end_ea-f.start_ea))
idapro.close_database(False)
'''
    run([sys.executable, '-c', exporter, db, raw], env=env)
    lifted, function, oc = [tmp_path / name for name in ('module.ll', 'function.ll', 'oc.ll')]
    run([sys.executable, ROOT / 'Scripts/ida2llvm.py', '-f', db, '-o', lifted], env=env)
    state = json.loads(Path(str(lifted) + '.lift.json').read_text())
    record = next(r for r in state['functions'] if r['name'] == '__aeabi_dadd')
    assert record['status'] == 'lifted' and not record.get('approx')
    run(['llvm-extract', '--func=__aeabi_dadd', '-S', lifted, '-o', function])
    run(['opt', '-S', '-passes=' + ','.join(canonicalize_passes_for_opt_level('-Oc')), function, '-o', oc])
    variants = [(raw.read_bytes(), {'__aeabi_dadd': 0})]
    for ll in (function, oc):
        obj = ll.with_suffix('.o')
        run(['llc', '-mtriple=armv7-linux-gnueabi', '-filetype=obj', ll, '-o', obj])
        variants.append(object_text(obj, tmp_path))
    values = [0, 1, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000,
              0x0010000000000000, 0x7fefffffffffffff, 0xffefffffffffffff,
              0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000001, 0x7ff0000000000001]
    rng = random.Random(731)
    pairs = [(x, y) for x in values for y in values]
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(1000)]
    for x, y in pairs:
        results = [execute('arm32', code, symbols['__aeabi_dadd'], x, y, 64) for code, symbols in variants]
        assert len(set(results)) == 1, (hex(x), hex(y), results)


def check_machine_equivalence(arch, tmp_path, text, names, allowed_approx=(), pairs=None, source_thumb=False, source_c=False):
    for tool in ("clang", "ld.lld", "llc", "opt", "llvm-nm", "llvm-objcopy", "llvm-as"):
        if not shutil.which(tool):
            pytest.skip(f"{tool} unavailable")
    ida = os.environ.get("IDADIR")
    if not ida:
        pytest.skip("IDADIR is required")
    src, obj, exe = tmp_path / "probe.S", tmp_path / "ref.o", tmp_path / "probe.elf"
    if source_c:
        src = src.with_suffix('.c')
    src.write_text(text)
    run(["clang", f"--target={TARGETS[arch][0]}", "-O2", "-c", src, "-o", obj])
    run(["ld.lld", "-e", names[0][0], obj, "-o", exe])
    lifted = tmp_path / "probe.ll"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ida, "idalib", "python") + os.pathsep + env.get("PYTHONPATH", "")
    run([sys.executable, ROOT / "Scripts/ida2llvm.py", "-f", exe, "-o", lifted], env=env)
    state = json.loads(Path(str(lifted) + ".lift.json").read_text())
    assert state["status"] == "ok" and state["counts"]["failed"] == 0
    assert state["counts"]["defined"] == len(names)
    assert set(state["approx_reasons"]) <= set(allowed_approx)
    oc = tmp_path / "oc.ll"
    passes = ",".join(canonicalize_passes_for_opt_level("-Oc"))
    run(["opt", f"-passes={passes}", "-S", lifted, "-o", oc])
    variants = [object_text(obj, tmp_path)]
    if source_thumb:
        code, symbols = variants[0]
        variants[0] = (code, {name: offset | 1 for name, offset in symbols.items()})
    for source in (lifted, oc):
        output = source.with_suffix(".o")
        run(["llc", "-filetype=obj", source, "-o", output])
        variants.append(object_text(output, tmp_path))
    for name, width in names:
        mask = (1 << width) - 1
        inputs = pairs if pairs is not None else [
            (x, n) for x in (0, 1, mask, 1 << (width - 1), 0x8123456789ABCDEF & mask)
            for n in sorted({0, 1, width - 1, width, width + 1, 31, 32, 63, 64, 127, 255, 256, 257})]
        for x, n in inputs:
            outputs = [execute(arch, code, symbols[name], x, n, width) for code, symbols in variants]
            assert outputs[0] == outputs[1] == outputs[2], (arch, name, hex(x), n, outputs)


@pytest.mark.parametrize('arch', list(TARGETS))
def test_strength_reduced_constant_division(arch, tmp_path):
    text = ('unsigned int probe_uquot(unsigned int x) { return x / 8; }\n'
            'unsigned int probe_urem(unsigned int x) { return x % 8; }\n'
            'int probe_squot(int x) { return x / 8; }\n'
            'int probe_srem(int x) { return x % 8; }\n')
    names = [(name, 32) for name in ('probe_uquot', 'probe_urem', 'probe_squot', 'probe_srem')]
    check_machine_equivalence(arch, tmp_path, text, names, source_c=True)


def test_a64_native_division(tmp_path):
    lines, names = ['.text'], []
    for width, reg in ((32, 'w'), (64, 'x')):
        for op in ('sdiv', 'udiv'):
            name = op + str(width)
            names.append((name, width))
            lines += [f'.globl {name}', f'.type {name},%function', f'{name}:',
                      f'{op} {reg}0,{reg}0,{reg}1', 'ret', f'.size {name},.-{name}']
    values = [0, 1, 0xffffffff, 0x80000000, 0xffffffffffffffff, 0x8000000000000000]
    check_machine_equivalence('a64', tmp_path, '\n'.join(lines), names,
                              ['hexrays_division_control_flow_unverified'],
                              [(x, y) for x in values for y in values])


@pytest.mark.parametrize('float_bits', [32, 64])
@pytest.mark.parametrize('int_bits', [32, 64])
def test_a64_native_float_conversion(tmp_path, float_bits, int_bits):
    lines, names = ['.text'], []
    fp, source = ('d0', 'x0') if float_bits == 64 else ('s0', 'w0')
    dest = 'x0' if int_bits == 64 else 'w0'
    for op in ('fcvtzs', 'fcvtzu'):
        # Pass the raw floating bit pattern in X0, even for a 32-bit result.
        names.append((op, 64))
        lines += [f'.globl {op}', f'.type {op},%function', f'{op}:',
                  f'fmov {fp},{source}', f'{op} {dest},{fp}', 'ret', f'.size {op},.-{op}']
    values = [float('nan'), float('inf'), -float('inf'), -0.0, 1.75, -1.75,
              float(1 << 31), -float(1 << 31), float(1 << 32),
              float(1 << 63), -float(1 << 63), float(1 << 64), 1e30, -1e30]
    fmt, rawfmt = ('<d', '<Q') if float_bits == 64 else ('<f', '<I')
    pairs = [(struct.unpack(rawfmt, struct.pack(fmt, x))[0], 0) for x in values]
    check_machine_equivalence('a64', tmp_path, '\n'.join(lines), names,
                              ['fp_exception_status_unmodeled'], pairs)


@pytest.mark.parametrize('thumb', [False, True])
def test_arm_immediate_carry(tmp_path, thumb):
    lines = ['.syntax unified', '.text', '.thumb' if thumb else '.arm']
    names = []
    for op in ('lsr', 'asr'):
        name = op + '_imm32'
        names.append((name, 32))
        lines += [f'.globl {name}', f'.type {name},%function']
        if thumb:
            lines.append('.thumb_func')
        lines += [f'{name}:', 'cmp r0,#1', f'{op}s r2,r0,#32',
                  'mov.w r0,#0' if thumb else 'mov r0,#0',
                  'adc r0,r0,#0', 'bx lr', f'.size {name},.-{name}']
    check_machine_equivalence('arm32', tmp_path, '\n'.join(lines), names, source_thumb=thumb)


def test_arm_arithmetic_shifter_flags(tmp_path):
    lines, names = ['.syntax unified', '.text'], []
    for op in ('add', 'sub', 'rsb', 'cmp', 'cmn'):
        for shift in ('lsl', 'lsr', 'asr'):
            name = 'probe_' + op + '_' + shift
            names.append((name, 32))
            instruction = (f'{op} r0,r0,{shift} r1' if op in ('cmp', 'cmn')
                           else f'{op}s r2,r0,r0,{shift} r1')
            lines += [f'.globl {name}', f'.type {name},%function', f'{name}:',
                      'cmp r0,#1', instruction, 'mov r0,#0', 'adc r0,r0,#0',
                      'bx lr', f'.size {name},.-{name}']
    check_machine_equivalence('arm32', tmp_path, '\n'.join(lines), names)


def test_carry_hook_exception_prevents_publication(tmp_path):
    src, obj, exe = [tmp_path / name for name in ('probe.S', 'probe.o', 'probe.elf')]
    src.write_text('.text\n.global probe\n.type probe,%function\nprobe:\nlsls r0,r0,r1\nbx lr\n.size probe,.-probe\n')
    run(['clang', '--target=armv7-linux-gnueabi', '-c', src, '-o', obj])
    run(['ld.lld', '-e', 'probe', obj, '-o', exe])
    output = tmp_path / 'fault.ll'
    # Inject into normalize(), inside the actual Hex-Rays callback, not run_lift.
    wrapper = '''import idapro
import sys
sys.path.insert(0, sys.argv[1])
import ida2llvm as L
install = L._install_arm_carry_fix
def failing_install():
    hook = install()
    def fail(self, mba):
        raise RuntimeError("injected-normalize-failure")
    type(hook).normalize = fail
    return hook
L._install_arm_carry_fix = failing_install
L.run_lift(sys.argv[2], sys.argv[3])
'''
    env = dict(os.environ, PYTHONPATH=os.path.join(os.environ['IDADIR'], 'idalib', 'python'))
    run([sys.executable, '-c', wrapper, ROOT / 'Scripts', exe, output], env=env)
    state = json.loads(Path(str(output) + '.lift.json').read_text())
    assert state['status'] == 'error'
    assert 'injected-normalize-failure' in state['error']
    assert not output.exists()


def test_upstream_division_branch_loss_is_not_claimed_exact(tmp_path):
    """Known IDA 9.4 limitation: retain a visible approximation, not false proof."""
    src, obj, exe = [tmp_path / name for name in ('guard.S', 'guard.o', 'guard.elf')]
    src.write_text('.text\n.global guard\n.type guard,%function\nguard:\n'
                   'udiv w2,w0,w1\ncbz w1,1f\nmov w0,w2\nret\n'
                   '1: mov w0,#-1\nret\n.size guard,.-guard\n')
    run(['clang', '--target=aarch64-linux-gnu', '-c', src, '-o', obj])
    run(['ld.lld', '-e', 'guard', obj, '-o', exe])
    output = tmp_path / 'guard.ll'
    env = dict(os.environ, PYTHONPATH=os.path.join(os.environ['IDADIR'], 'idalib', 'python'))
    run([sys.executable, ROOT / 'Scripts/ida2llvm.py', '-f', exe, '-o', output], env=env)
    state = json.loads(Path(str(output) + '.lift.json').read_text())
    record = next(r for r in state['functions'] if r['name'] == 'guard')
    assert 'hexrays_division_control_flow_unverified' in record['approx']
    original, symbols = object_text(obj, tmp_path)
    assert execute('a64', original, symbols['guard'], 10, 0, 32) == 0xffffffff
    oc = tmp_path / 'oc.ll'
    run(['opt', '-S', '-passes=' + ','.join(canonicalize_passes_for_opt_level('-Oc')), output, '-o', oc])
    observed = []
    for ll in (output, oc):
        compiled = ll.with_suffix('.o')
        run(['llc', '-filetype=obj', ll, '-o', compiled])
        code, symbols = object_text(compiled, tmp_path)
        observed.append(execute('a64', code, symbols['guard'], 10, 0, 32))
    assert observed == [0, 0], 'Upstream behavior changed: reassess the limitation'


@pytest.mark.parametrize('kind', ['ror', 'aeabi'])
def test_ambiguous_native_sources_rejected(tmp_path, kind):
    src, obj, exe = [tmp_path / name for name in ('probe.S', 'probe.o', 'probe.elf')]
    objects = [obj]
    if kind == 'ror':
        entry, reason = 'carry_ror', 'arm_shift_carry_source_unresolved'
        text = ('.syntax unified\n.text\n.global carry_ror\n.type carry_ror,%function\n'
                'carry_ror:\ncmp r0,#1\nrors r2,r0,r1\nmov r0,#0\nadc r0,r0,#0\nbx lr\n'
                '.size carry_ror,.-carry_ror\n')
    else:
        entry, reason = 'conv', 'float_conversion_machine_semantics_unresolved'
        # Deliberately nonstandard implementation proves name-based substitution
        # is not a sound basis for assigning saturating conversion semantics.
        text = ('.syntax unified\n.text\n.global __aeabi_d2ulz\n.type __aeabi_d2ulz,%function\n'
                '__aeabi_d2ulz:\neor r0,r0,r1\nbx lr\n.size __aeabi_d2ulz,.-__aeabi_d2ulz\n')
        caller, caller_obj = tmp_path / 'caller.c', tmp_path / 'caller.o'
        caller.write_text('unsigned long long conv(double x) { return (unsigned long long)x; }\n')
        run(['clang', '--target=armv7-linux-gnueabi', '-O1', '-c', caller, '-o', caller_obj])
        objects.append(caller_obj)
    src.write_text(text)
    run(['clang', '--target=armv7-linux-gnueabi', '-c', src, '-o', obj])
    run(['ld.lld', '-e', entry, *objects, '-o', exe])
    output = tmp_path / 'probe.ll'
    env = dict(os.environ, PYTHONPATH=os.path.join(os.environ['IDADIR'], 'idalib', 'python'))
    process = subprocess.run([sys.executable, str(ROOT / 'Scripts/ida2llvm.py'), '-f', str(exe), '-o', str(output)],
                             env=env, capture_output=True, text=True, timeout=120)
    assert process.returncode in (0, 3), process.stderr
    state = json.loads(Path(str(output) + '.lift.json').read_text())
    record = next(r for r in state['functions'] if r['name'] == entry)
    assert record['status'] == 'rejected' and record['reason'] == reason, record


@pytest.mark.parametrize("hard", [False, True])
def test_arm_scalar_float_call_abi(hard, tmp_path):
    ida = os.environ.get("IDADIR")
    if not ida:
        pytest.skip("IDADIR is required")
    src, obj, exe = tmp_path / "float.c", tmp_path / "float.o", tmp_path / "float.elf"
    src.write_text("float pick(float x, float y) { return y; }\n")
    caller_src, caller_obj = tmp_path / "caller.c", tmp_path / "caller.o"
    caller_src.write_text("extern float pick(float, float);\nfloat caller(float x, float y) { return pick(x, y); }\n")
    triple = "armv7-linux-gnueabihf" if hard else "armv7-linux-gnueabi"
    run(["clang", f"--target={triple}", "-mfpu=vfpv3", f"-mfloat-abi={'hard' if hard else 'softfp'}",
         "-g", "-O1", "-fno-optimize-sibling-calls", "-c", src, "-o", obj])
    run(["clang", f"--target={triple}", "-mfpu=vfpv3", f"-mfloat-abi={'hard' if hard else 'softfp'}",
         "-g", "-O1", "-fno-optimize-sibling-calls", "-c", caller_src, "-o", caller_obj])
    run(["ld.lld", "-e", "caller", obj, caller_obj, "-o", exe])
    lifted = tmp_path / "lifted.ll"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ida, "idalib", "python") + os.pathsep + env.get("PYTHONPATH", "")
    run([sys.executable, ROOT / "Scripts/ida2llvm.py", "-f", exe, "-o", lifted], env=env)
    state = json.loads(Path(str(lifted) + ".lift.json").read_text())
    assert state["status"] == "ok" and state["counts"]["failed"] == 0
    assert state["target"]["abi"] == ("aapcs-vfp" if hard else "aapcs-base")
    oc = tmp_path / "oc.ll"
    run(["opt", "-passes=" + ",".join(canonicalize_passes_for_opt_level("-Oc")), "-S", lifted, "-o", oc])
    executables = [exe]
    for source in (lifted, oc):
        output = source.with_suffix(".o")
        run(["llc", "-filetype=obj", source, "-o", output])
        linked = source.with_suffix(".elf")
        run(["ld.lld", "-e", "caller", output, "-o", linked])
        executables.append(linked)
    # Link first: calls have relocations. Map .text at its linked address.
    for binary in executables:
        text, symbols = object_text(binary, tmp_path)
        headers = json.loads(run(["llvm-readobj", "--elf-output-style=JSON", "--sections", binary]))
        sections = headers[0]["Sections"]
        address = next(s["Section"]["Address"] for s in sections if s["Section"]["Name"]["Name"] == ".text")
        for name in ("pick", "caller"):
            for x, y in ((1.25, -2.5), (-0.0, 3.75)):
                uc = U.Uc(U.UC_ARCH_ARM, U.UC_MODE_ARM)
                base = address & ~0xfff
                uc.mem_map(base, 0x10000)
                uc.mem_write(address, text)
                uc.mem_map(0x800000, 0x20000)
                uc.mem_map(0x900000, 0x1000)
                uc.reg_write(A.UC_ARM_REG_SP, 0x810000)
                uc.reg_write(A.UC_ARM_REG_LR, 0x900000)
                uc.reg_write(A.UC_ARM_REG_C1_C0_2, 0xF << 20)
                uc.reg_write(A.UC_ARM_REG_FPEXC, 1 << 30)
                xb, yb = [struct.unpack('<I', struct.pack('<f', v))[0] for v in (x, y)]
                arg0, arg1 = (A.UC_ARM_REG_S0, A.UC_ARM_REG_S1) if hard else (A.UC_ARM_REG_R0, A.UC_ARM_REG_R1)
                uc.reg_write(arg0, xb)
                uc.reg_write(arg1, yb)
                uc.emu_start(symbols[name], 0x900000, timeout=100000, count=10000)
                assert uc.reg_read(A.UC_ARM_REG_PC) == 0x900000
                assert uc.reg_read(arg0) == yb, (hard, binary.name, name, hex(uc.reg_read(arg0)), hex(yb))
