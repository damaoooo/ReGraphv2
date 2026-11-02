"""
Classify register_reuse_unrecoverable rejections by what made the register value "dirty".
usage: rr_classify.py LIFTER_DIR BINARY LIFT_JSON OUT_JSON [max_funcs]
"""
import sys, json, collections
sys.path.insert(0, sys.argv[1])
import ida2llvm as L
import idapro, ida_auto, ida_hexrays, ida_funcs, idc, ida_name, ida_segment, idautils

binp, liftjson, outp = sys.argv[2], sys.argv[3], sys.argv[4]
maxf = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9
recs = [r for r in json.load(open(liftjson))["functions"] if r.get("reason") == "register_reuse_unrecoverable"][:maxf]
idapro.open_database(binp, True); ida_auto.auto_wait()
L._reset_lift_state(); L._init_target_state()
hooks = [L._install_arm_carry_fix(), L._install_a64_division_fix(), L._install_rotate_fix(frozenset())]
L._apply_arm_runtime_types(); ida_hexrays.clear_cached_cfuncs()


def callee_info(ins):
    """for a call minsn (possibly nested), describe the callee."""
    c = ins
    if c.opcode not in (ida_hexrays.m_call, ida_hexrays.m_icall):
        # find nested call
        class F(ida_hexrays.minsn_visitor_t):
            def __init__(s): super().__init__(); s.found = None
            def visit_minsn(s):
                if s.curins.opcode in (ida_hexrays.m_call, ida_hexrays.m_icall) and s.found is None: s.found = s.curins
                return 0
        f = F(); ins.for_all_insns(f); c = f.found
        if c is None: return ("?", "nocall")
    if c.opcode == ida_hexrays.m_icall:
        return ("<indirect>", "indirect")
    if c.l.t == ida_hexrays.mop_h:
        return (c.l.helper, "helper")
    if c.l.t != ida_hexrays.mop_v:
        return ("?", "other")
    ea = c.l.g
    fn = ida_funcs.get_func(ea)
    name = ida_name.get_name(ea)
    seg = ida_segment.get_segm_name(ida_segment.getseg(ea)) if ida_segment.getseg(ea) else None
    if fn is None or L._is_extern_segment(ea):
        kind = "extern"
    elif fn.flags & ida_funcs.FUNC_THUNK:
        kind = "thunk"
    elif fn.flags & ida_funcs.FUNC_LIB:
        kind = "local_LIB"
    else:
        kind = "local"
    if seg not in (".text", None) and kind.startswith("local"):
        kind += "@" + seg
    return (name, kind)


def classify(mba):
    lvars = mba.vars
    base, ranges, total = {}, {}, 0
    for idx in range(lvars.size()):
        lv = lvars[idx]
        if lv.location.is_reg1() and lv.width > 0 and not lv.is_arg_var:
            st = int(lv.location.reg1()); base[idx], ranges[idx] = total, (st, st + lv.width); total += lv.width
    if not total: return {}
    physical = {}; whole = {}
    for idx, (s, e) in ranges.items():
        whole[idx] = ((1 << (e - s)) - 1) << base[idx]
        for k in range(e - s): physical[s + k] = physical.get(s + k, 0) | 1 << (base[idx] + k)
    def own(idx, off, size):
        w = ranges[idx][1] - ranges[idx][0]; off, end = max(off, 0), min(off + size, w)
        return ((1 << (end - off)) - 1) << (base[idx] + off) if end > off else 0
    def clob(s, e):
        m = 0
        for b in range(s, e): m |= physical.get(b, 0)
        return m
    def spoil_mask(call):
        if call is True: return (1 << total) - 1
        m = 0
        for sp in call:
            for byte, bits in physical.items():
                if sp.reg.has(byte): m |= bits
        return m
    results = [(idx, 0, lvars[idx].width) for idx in range(lvars.size()) if lvars[idx].is_result_var]
    events = []
    for index in range(mba.qty):
        blk = mba.get_mblock(index); seq = []; ins = blk.head
        while ins is not None:
            uses, call, defs, reg_defs = L._collect_lvar_accesses(ins, lvars)
            uses = [(idx, own(idx, o, s)) for idx, o, s in uses if idx in base]
            kills = []
            for idx, o, s in defs:
                loc = lvars[idx].location
                written = own(idx, o, s) if idx in base else 0
                hit = clob(int(loc.reg1()) + o, int(loc.reg1()) + o + s) if loc.is_reg1() else 0
                self_rest = (whole[idx] & ~written) if idx in base else 0
                other = hit & ~written & ~(whole[idx] if idx in base else 0)
                kills.append((written, self_rest, other, ("partial_self", lvars[idx].name), ("other_lvar", lvars[idx].name)))
            for s, e in reg_defs:
                kills.append((0, 0, clob(s, e), None, ("reg_def", s)))
            cinfo = callee_info(ins) if call else None
            seq.append((uses, spoil_mask(call) if call else 0, kills, ("call",) + cinfo if cinfo else None, ins.ea))
            ins = ins.next
        if blk.type == ida_hexrays.BLT_STOP and results:
            seq.append(([(idx, own(idx, o, s)) for idx, o, s in results if idx in base], 0, [], None, None))
        events.append(seq)
    # state: clean mask, dirty: dict bit -> frozenset(reasons)
    def bitsof(m):
        while m:
            low = m & -m; yield low.bit_length() - 1; m ^= low
    states = [None] * mba.qty
    states[0] = ((1 << total) - 1, {})
    work = [0]; exposed = collections.defaultdict(set)
    iters = 0
    while work and iters < 20000:
        iters += 1
        i = work.pop()
        clean, dirty = states[i][0], dict(states[i][1])
        for uses, spoiled, kills, cinfo, ea in events[i]:
            for idx, m in uses:
                for b in bitsof(m):
                    if b in dirty: exposed[idx] |= dirty[b]
            if spoiled:
                moved = clean & spoiled; clean &= ~moved
                for b in bitsof(moved): dirty[b] = frozenset([cinfo])
            for written, self_rest, other, r_self, r_other in kills:
                clean &= ~written
                for b in bitsof(written): dirty.pop(b, None)
                for mask, r in ((self_rest, r_self), (other, r_other)):
                    moved = clean & mask; clean &= ~moved
                    for b in bitsof(moved): dirty[b] = frozenset([r])
        blk = mba.get_mblock(i)
        for k in range(blk.nsucc()):
            s = blk.succ(k); old = states[s]
            if old is None: new = (clean, dirty)
            else:
                nd = dict(old[1])
                for b, r in dirty.items(): nd[b] = nd.get(b, frozenset()) | r
                new = (old[0] | clean, nd)
            if old is None or new[0] != old[0] or new[1] != old[1]:
                states[s] = new; work.append(s)
    out = {}
    for idx, rs in exposed.items():
        lv = lvars[idx]
        out[lv.name or f"lv{idx}"] = {"reg": ida_hexrays.get_mreg_name(int(lv.location.reg1()), lv.width), "result": bool(lv.is_result_var),
                                      "reasons": sorted(set(tuple(map(str, r)) for r in rs))}
    return out


res = []
for r in recs:
    ea = int(r["ea"], 16)
    try:
        cf = ida_hexrays.decompile(ea, ida_hexrays.hexrays_failure_t(), ida_hexrays.DECOMP_NO_CACHE)
        if cf is None: continue
        res.append({"ea": r["ea"], "name": r["name"], "type": str(cf.type), "dirty": classify(cf.mba)})
    except Exception as e:
        res.append({"ea": r["ea"], "name": r["name"], "error": repr(e)})
json.dump(res, open(outp, "w"), indent=1)
c = collections.Counter()
for x in res:
    kinds = set()
    for v in x.get("dirty", {}).values():
        for rr in v["reasons"]:
            kinds.add(rr[0] if rr[0] != "call" else "call:" + rr[2])
    c[tuple(sorted(kinds))] += 1
for k, v in c.most_common(25): print(v, k)
