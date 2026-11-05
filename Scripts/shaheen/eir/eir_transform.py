"""EIR (rev.ng isolate) module transforms, agreed with the 9950X (2026-11-04), applied before purify / normalize and
before linking:

  1. block labels: inside quotes, every character not in [A-Za-z0-9._] becomes '_' and the quotes are dropped
     ("bb.0x40b600:Code_x86_64_cloned" -> bb.0x40b600_Code_x86_64_cloned), for definitions and %"..." references;
  2. every `define` that is not @local_* (rev.ng helpers such as @get_segment_*) becomes a `declare`, so a module
     defines only the lifted function.

For part B text (rell) only, agreed with the 9950X 2026-11-04 21:36 (eir_text_b.py), after 1 + 2:
  B3. metadata removed: every `!N = ...` / named-metadata line, `, !xxx !N` attachments, `attributes` lines and `#N`
      attribute-group references (form only, as purify means: no metadata).

Main text for Qwen (text_main; the user's rule via the 9950X, 2026-11-04 21:54), after 1 + 2 + B3 and purify:
  M3. type definitions (`%x = type ...`) removed;
  M4. global variable definitions removed: every `@x = ... global|constant ...` line that is not an `external` /
      `extern_weak` declaration (the CSVs and every other variable with an initializer); B3 is applied again to drop the
      attribute groups opt prints. Kept: the lifted function's body, `declare` lines, external global declarations.
      Not valid IR (references to the removed types / globals stay as they are); used as plain text only.

No-ISA form (bodyfirst noisa fused; the user's rule via the 9950X, 2026-11-05 11:44), on the moved purified /
instrumented IR, with one label map shared by both files:
  N1. string constants: `0x<hex>:Code_<isa>` inside c"..." becomes `PC` (the [N x i8] length is recomputed);
  N2. block labels: `bb.0x<hex>_Code_<isa>` becomes `bb.pc<k>`, k = order of first occurrence in the function (from 1),
      for the definition and every reference;
  N3. every other `Code_<isa>` (e.g. local_<addr>_Code_<isa> names) becomes `PC`.

For the .re build only (eir_re.py):
  3. @local_<addr>_<arch> -> the IDA name of that function (definition and declarations), from S2 functions.json;
  4. global variable definitions with default / dso_local linkage get `weak` linkage, so llvm-link keeps one copy of
     the CPU-state CSVs and runtime constants that every module defines identically.
"""
import re

_LABEL_REF = re.compile(r'%"([^"]+)"')
_LABEL_DEF = re.compile(r'^"([^"]+)":')
_SAFE = re.compile(r"[^A-Za-z0-9._]")
_LOCAL = re.compile(r'@"?(local_(0x[0-9a-f]+)_[A-Za-z0-9_]+)"?')
_GLOBAL_DEF = re.compile(r'^(@"[^"]+"|@[-A-Za-z$._0-9]+) = (dso_local )?(global|constant) ')


def sanitize_labels(text):
    out = []
    for line in text.split("\n"):
        if line[:1] in ("@", "!") or line.startswith(("declare", "attributes", "source_filename", "target ")):
            out.append(line)
            continue
        line = _LABEL_REF.sub(lambda m: "%" + _SAFE.sub("_", m.group(1)), line)
        line = _LABEL_DEF.sub(lambda m: _SAFE.sub("_", m.group(1)) + ":", line)
        out.append(line)
    return "\n".join(out)


def only_local(text):
    out, skip = [], False
    for line in text.split("\n"):
        if skip:
            if line.startswith("}"):
                skip = False
            continue
        if line.startswith("define ") and "@local_" not in line.split("(", 1)[0]:
            head = line[len("define "):].rstrip()
            head = re.sub(r"\s*\{\s*$", "", head)
            head = re.sub(r"\s+#\d+.*$", "", head)          # drop attribute group / metadata attachments of the body
            out.append("declare " + head)
            skip = True
            continue
        out.append(line)
    return "\n".join(out)


def agreed(text):
    """Transforms 1 + 2 (EIR text for fused / rell / irsim and the first step of the .re build)."""
    return only_local(sanitize_labels(text))


_META_ATT = re.compile(r"(, )?!\S+ !\d+")
_ATTR_REF = re.compile(r"\s+#\d+")


def strip_metadata(text):
    """Transform B3: metadata lines / attachments, attributes lines and #N references removed."""
    out = []
    for line in text.split("\n"):
        if line.startswith("!") or line.startswith("attributes #"):
            continue
        out.append(_ATTR_REF.sub("", _META_ATT.sub("", line)))
    return "\n".join(out)


_TYPE_DEF = re.compile(r"^%\S+ = type ")
_EXTERN_DECL = re.compile(r"^@\S+ = (\w+ )*(external|extern_weak) ")


def main_form(text):
    """Transforms M3 + M4 (on purified IR): type definitions and global variable definitions removed, metadata /
    attribute groups removed again."""
    out = []
    for line in strip_metadata(text).split("\n"):
        if _TYPE_DEF.match(line):
            continue
        if line.startswith("@") and " = " in line and not _EXTERN_DECL.match(line):
            continue
        out.append(line)
    return "\n".join(out)


_ISA = r"Code_(?:x86_64|x86|arm_thumb|arm|aarch64|mipsel|mips|systemz)"
_STR_PC = re.compile(r"0x[0-9a-fA-F]+:" + _ISA)
_LABEL_PC = re.compile(r"(?<=bb\.)0x[0-9a-fA-F]+_" + _ISA)
_ANY_ISA = re.compile(_ISA)
_I8_STR = re.compile(r'\[(\d+) x i8\] c"((?:[^"\\]|\\[0-9A-Fa-f]{2}|\\\\)*)"')
RESIDUAL = re.compile(r"code_?(arm|thumb|aarch64|mips|x86)", re.I)


def _i8_len(body):
    return len(re.sub(r"\\[0-9A-Fa-f]{2}|\\\\", "x", body))


def noisa(text, labels=None):
    """Transforms N1-N3. labels: {'0x.._Code_..': 'pcK'} shared between the purified and the instrumented IR (new keys
    are numbered on). Returns (text, labels)."""
    labels = {} if labels is None else labels
    out = []
    for line in text.split("\n"):
        if "Code_" in line:
            if 'c"' in line and _STR_PC.search(line):
                line = _I8_STR.sub(lambda m: (f'[{_i8_len(_STR_PC.sub("PC", m.group(2)))} x i8] c"{_STR_PC.sub("PC", m.group(2))}"'
                                              if _STR_PC.search(m.group(2)) else m.group(0)), line)
                line = _STR_PC.sub("PC", line)
            line = _LABEL_PC.sub(lambda m: labels.setdefault(m.group(0), f"pc{len(labels) + 1}"), line)
            line = _ANY_ISA.sub("PC", line)
        out.append(line)
    return "\n".join(out), labels


def body_first(text):
    """The first `define ... {` ... `}` block moved to the front; everything else unchanged and in order after it."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("define "):
            for j in range(i, len(lines)):
                if lines[j].startswith("}"):
                    return "\n".join(lines[i:j + 1] + lines[:i] + lines[j + 1:])
    return text


def ir_name(name):
    return "@" + name if re.fullmatch(r"[-A-Za-z$._][-A-Za-z$._0-9]*", name) else '@"' + name.replace('"', '\\22') + '"'


def rename_locals(text, names):
    """names: {local function name (local_<addr>_<isa>): IDA name}. Returns (text, renamed, unmatched local names)."""
    seen, missing = set(), set()

    def sub(m):
        n = names.get(m.group(1))
        if n is None:
            missing.add(m.group(1))
            return m.group(0)
        seen.add(m.group(1))
        return ir_name(n)
    return _LOCAL.sub(sub, text), seen, missing


def local_name(member):
    """'/function/0x400ed4:Code_arm_thumb' -> 'local_0x400ed4_Code_arm_thumb' (rev.ng's function name)."""
    a, isa = member.rsplit("/", 1)[-1].split(":", 1)
    return f"local_{a}_{isa}"


def thumb_addresses(binary, readelf="/usr/lib/llvm-18/bin/llvm-readelf"):
    """ARM ELF: FUNC symbols with an odd value are Thumb code; returns the even addresses."""
    import subprocess
    out = subprocess.run([readelf, "-sW", binary], capture_output=True, text=True).stdout
    th = set()
    for line in out.split("\n"):
        f = line.split()
        if len(f) >= 8 and f[3] == "FUNC":
            try:
                v = int(f[1], 16)
            except ValueError:
                continue
            if v & 1:
                th.add(v - 1)
    return th


def choose_members(members, ida, offset, thumb=()):
    """Map rev.ng function members to IDA functions by address. When rev.ng lifted one address in several ISAs
    (arm32: Code_arm and Code_arm_thumb), only one gets the IDA function: Code_arm_thumb if the original ELF has a
    Thumb FUNC symbol there, else Code_arm (else the first in name order). Returns ({member: IDA ea}, duplicates)."""
    by_addr = {}
    for m in members:
        by_addr.setdefault(int(m.rsplit("/", 1)[-1].split(":")[0], 16), []).append(m)
    chosen, dups = {}, 0
    for ea in ida:
        cands = sorted(by_addr.get(ea + offset, []))
        if not cands:
            continue
        if len(cands) > 1:
            dups += 1
            want = "Code_arm_thumb" if ea in thumb else "Code_arm"
            pick = [c for c in cands if c.endswith(":" + want)] or cands
            cands = pick[:1]
        chosen[cands[0]] = ea
    return chosen, dups


def weak_globals(text):
    """Global variable definitions -> weak; returns (text, {global name: definition with the linkage removed})."""
    defs, out = {}, []
    for line in text.split("\n"):
        m = _GLOBAL_DEF.match(line)
        if m:
            name = m.group(1)
            body = line[m.end(0) - len(m.group(3)) - 1:]
            defs[name] = re.sub(r", !\S+ !\d+", "", body)    # compare without metadata attachments
            line = f"{name} = weak {m.group(3)} " + line[m.end(0):]
        out.append(line)
    return "\n".join(out), defs


def local_addresses(member_names):
    """'/function/0x40b600:Code_x86_64' -> 0x40b600 for the isolate tar members."""
    return {int(n.rsplit("/", 1)[-1].split(":")[0], 16): n for n in member_names}


OFFSETS = (0, 0x400000, 0x10000, 0x8000, 0x100000, 0x1000000, 0x10000000, 0x120000000)


def load_offset(revng_addrs, ida_eas):
    """rev.ng loads position-independent ELFs at its own base; pick the offset that matches most IDA functions
    (same rule as the 9950X eir_text.py)."""
    have = set(revng_addrs)
    return max(OFFSETS, key=lambda o: sum((e + o) in have for e in ida_eas))
