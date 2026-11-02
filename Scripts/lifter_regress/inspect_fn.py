"""usage: inspect_fn.py LIFTER_DIR BINARY name|0xEA ... ; prints pseudocode, lvars, microcode."""
import sys, os
sys.path.insert(0, sys.argv[1])
import ida2llvm as L
import idapro, ida_auto, ida_hexrays, ida_name, idc, ida_lines, idaapi, ida_funcs
binp, targets = sys.argv[2], sys.argv[3:]
idapro.open_database(binp, True)
ida_auto.auto_wait()
L._init_target_state()
hooks = [L._install_arm_carry_fix(), L._install_a64_division_fix(), L._install_rotate_fix(L._rotate_fix_from_env())]
L._apply_arm_runtime_types()
ida_hexrays.clear_cached_cfuncs()
for t in targets:
    ea = int(t, 16) if t.startswith("0x") else ida_name.get_name_ea(idaapi.BADADDR, t)
    print("=" * 30, t, hex(ea))
    f = ida_funcs.get_func(ea)
    for item in __import__("idautils").FuncItems(ea):
        print("  %08x  %s" % (item, idc.GetDisasm(item)))
    hf = ida_hexrays.hexrays_failure_t()
    cf = ida_hexrays.decompile(ea, hf, ida_hexrays.DECOMP_NO_CACHE)
    if cf is None:
        print("decompile failed", hf.str); continue
    print(str(cf))
    print("type:", cf.type)
    for i, v in enumerate(cf.mba.vars):
        loc = v.location
        where = ("reg %d(%s)" % (loc.reg1(), ida_hexrays.get_mreg_name(loc.reg1(), v.width))) if loc.is_reg1() else ("stk %d" % loc.stkoff() if loc.is_stkoff() else str(loc.atype()))
        print("  lvar %d %-10s w=%d %s arg=%d res=%d tif=%s" % (i, v.name, v.width, where, v.is_arg_var, v.is_result_var, v.tif))
    print("entry_reads:", {cf.mba.vars[k].name: d for k, d in L._entry_register_reads(cf.mba).items()})
    vp = ida_hexrays.vd_printer_t()
    class P(ida_hexrays.vd_printer_t):
        def __init__(s): super().__init__(); s.lines = []
        def _print(s, indent, line): s.lines.append(ida_lines.tag_remove(line)); return 1
    p = P(); cf.mba._print(p)
    print("\n".join(l.rstrip() for l in p.lines))
for h in hooks:
    if h is not None: h.unhook()
idapro.close_database(False)
