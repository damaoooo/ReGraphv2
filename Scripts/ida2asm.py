#!/usr/bin/env python3
"""Export each IDA function into a standalone .asm file via IDA 9.3 idalib."""

try:
    import idapro  # Must be imported before other IDA Python modules in idalib mode.
except ImportError as exc:
    raise SystemExit(
        "Failed to import 'idapro'. Install and activate IDA idalib first:\n"
        "  python <IDA_DIR>/idalib/python/py-activate-idalib.py -d <IDA_DIR>\n"
        "  python -m pip install <IDA_DIR>/idalib/python/idapro-*.whl\n"
        "You can also set the IDADIR environment variable to your IDA installation."
    ) from exc

import argparse
import logging
import re
import sys
from pathlib import Path

import ida_auto
import ida_funcs
import ida_lines
import ida_ua
import idautils
import idc


INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
DEFAULT_ENCODING = "utf-8"
SKIPPED_SEGMENT_TOKENS = {"plt", "got", "iplt"}
SKIPPED_SEGMENT_NAMES = {
    ".init",
    "init",
    ".fini",
    "fini",
    ".init_array",
    "init_array",
    ".fini_array",
    "fini_array",
    ".preinit_array",
    "preinit_array",
}
COMPILER_HELPER_EXACT_NAMES = {
    ".init_proc",
    "init_proc",
    ".term_proc",
    "term_proc",
    "_start",
    "start",
    "_init",
    "_fini",
    "frame_dummy",
    "__do_global_ctors_aux",
    "__do_global_dtors_aux",
    "deregister_tm_clones",
    "register_tm_clones",
    "_dl_relocate_static_pie",
    "__libc_csu_init",
    "__libc_csu_fini",
    "__libc_start_main",
    "__libc_start_call_main",
    "__x86_return_thunk",
    "__stack_chk_fail",
    "__stack_chk_fail_local",
    "__chkstk",
    "__chkstk_ms",
    "__alloca_probe",
    "__alloca_probe_16",
    "__security_init_cookie",
    "__security_check_cookie",
    "__security_check_cookie@4",
    "__static_initialization_and_destruction_0",
    "maincrtstartup",
    "wmaincrtstartup",
    "winmaincrtstartup",
    "wwinmaincrtstartup",
    "__tmaincrtstartup",
}
COMPILER_HELPER_PREFIXES = (
    "__x86.get_pc_thunk.",
    "__libc_csu_",
    "__aeabi_",
    "__gnu_",
    "__cxa_",
    "_unwind_",
    "_global__sub_i_",
    "_global__sub_d_",
    "__scrt_",
    "_guard_",
    "__guard_",
    "__gxx_personality_",
    "__clang_call_terminate",
)
LOGGER = logging.getLogger("ida2asm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export each IDA function into a separate .asm file using idalib.",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Binary, .i64, or .idb file to analyze with IDA.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory used to store exported .asm files.",
    )
    parser.add_argument(
        "--save-database",
        action="store_true",
        help="Persist IDA database changes before closing.",
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path. Defaults to <output-dir>/ida2asm.log.",
    )
    return parser.parse_args()


def configure_logging(log_file):
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding=DEFAULT_ENCODING)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def sanitize_filename(name, fallback_name):
    sanitized = INVALID_FILENAME_CHARS_RE.sub("_", name).strip().rstrip(".")
    return sanitized or fallback_name


def get_output_path(output_dir, base_name):
    return output_dir / f"{base_name}.asm"


def get_function_code_eas(func_ea):
    return [
        ea
        for ea in idautils.FuncItems(func_ea)
        if idc.is_code(idc.get_full_flags(ea))
    ]


def should_skip_segment(func):
    seg_name = idc.get_segm_name(func.start_ea) or ""
    normalized_name = seg_name.lower()
    if normalized_name in SKIPPED_SEGMENT_NAMES:
        return True, f"located in {seg_name} segment"

    tokens = {token for token in re.split(r"[._]", seg_name.lower()) if token}
    if tokens & SKIPPED_SEGMENT_TOKENS:
        return True, f"located in {seg_name} segment"
    return False, ""


def should_skip_compiler_helper(func_name, func):
    normalized_name = func_name.lower()

    if func.flags & ida_funcs.FUNC_LIB:
        return True, "marked by IDA as a library/helper function"

    if normalized_name in COMPILER_HELPER_EXACT_NAMES:
        return True, f"matched helper name {func_name}"

    for prefix in COMPILER_HELPER_PREFIXES:
        if normalized_name.startswith(prefix):
            return True, f"matched helper prefix {prefix}"

    return False, ""


def should_skip_function(func, code_eas):
    if func.flags & ida_funcs.FUNC_THUNK:
        return True, "marked by IDA as a thunk/jumper"

    skip, reason = should_skip_segment(func)
    if skip:
        return True, reason

    func_name = ida_funcs.get_func_name(func.start_ea) or f"sub_{func.start_ea:x}"
    skip, reason = should_skip_compiler_helper(func_name, func)
    if skip:
        return True, reason

    if len(code_eas) <= 1:
        return True, f"contains only {len(code_eas)} instruction(s)"

    return False, ""


def format_operand(ea, op_index):
    operand = idc.print_operand(ea, op_index) or ""
    op_type = idc.get_operand_type(ea, op_index)
    if op_type in (ida_ua.o_near, ida_ua.o_far):
        target = idc.get_operand_value(ea, op_index)
        if target != idc.BADADDR:
            return f"0x{target:x}"
    return operand


def render_instruction(ea):
    insn = ida_ua.insn_t()
    if ida_ua.decode_insn(insn, ea) <= 0:
        disasm = ida_lines.generate_disasm_line(ea, ida_lines.GENDSM_REMOVE_TAGS)
        return disasm or ""

    mnemonic = idc.print_insn_mnem(ea) or ""
    operands = []
    for op_index in range(len(insn.ops)):
        if insn.ops[op_index].type == ida_ua.o_void:
            break
        operand_text = format_operand(ea, op_index).strip()
        if operand_text:
            operands.append(operand_text)

    if operands:
        return f"{mnemonic:<8}{', '.join(operands)}"
    return mnemonic


def export_function(output_dir, func_ea):
    func = ida_funcs.get_func(func_ea)
    if func is None:
        return False, f"Skipped address {hex(func_ea)}: not a valid function"

    fallback_name = f"sub_{func.start_ea:x}"
    func_name = ida_funcs.get_func_name(func.start_ea) or fallback_name
    code_eas = get_function_code_eas(func.start_ea)
    should_skip, skip_reason = should_skip_function(func, code_eas)
    if should_skip:
        return None, f"Skipped {func_name}: {skip_reason}"

    clean_name = sanitize_filename(func_name, fallback_name)
    file_path = get_output_path(output_dir, clean_name)

    with file_path.open("w", encoding=DEFAULT_ENCODING) as handle:
        for curr in code_eas:
            handle.write(f"{render_instruction(curr)}\n")

    return True, f"Exported {func_name} -> {file_path}"


def export_all_functions(output_dir):
    exported = 0
    skipped = 0
    failed = 0

    for func_ea in idautils.Functions():
        try:
            ok, message = export_function(output_dir, func_ea)
            if ok is False:
                LOGGER.error(message)
            else:
                LOGGER.info(message)
            if ok is True:
                exported += 1
            elif ok is None:
                skipped += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            func_name = ida_funcs.get_func_name(func_ea) or f"sub_{func_ea:x}"
            LOGGER.exception("Failed to export %s: %s", func_name, exc)

    return exported, skipped, failed


def main():
    args = parse_args()

    input_path = Path(args.file).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else output_dir / "ida2asm.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file)
    LOGGER.info("Log file: %s", log_file)

    if not input_path.exists():
        LOGGER.error("Input file does not exist: %s", input_path)
        return 2

    opened = False
    try:
        LOGGER.info("Opening database with idalib: %s", input_path)
        open_status = idapro.open_database(str(input_path), True)
        if open_status != 0:
            raise RuntimeError(f"idapro.open_database() failed with status {open_status}")
        opened = True

        ida_auto.auto_wait()

        LOGGER.info("Starting function export to: %s", output_dir)
        exported, skipped, failed = export_all_functions(output_dir)
        LOGGER.info(
            "Export finished. Exported: %d, skipped: %d, failed: %d.",
            exported,
            skipped,
            failed,
        )
        return 0 if failed == 0 else 1
    except Exception as exc:
        LOGGER.exception("Export failed: %s", exc)
        return 1
    finally:
        if opened:
            LOGGER.info("Closing database (save=%s)...", args.save_database)
            idapro.close_database(save=args.save_database)
        return 0

if __name__ == "__main__":
    main()
