#!/usr/bin/env python3
"""Per-function basic-block count and opcode hash of an IDA database, as the Dataset-1 benchmark computes them.

idalib port of Cisco-Talos binary_function_similarity IDA_scripts/IDA_flowchart/IDA_flowchart.py
(Marcelli et al., USENIX Security '22): bb_num = len(list(FlowChart(func))); hashopcodes = sha256 of the
upper-cased concatenation of the instruction mnemonics of all basic blocks, blocks sorted by start address.
Unlike the original it keeps every function (Step 1, at least five blocks, is applied by select_functions.py).

usage: ida_flowchart.py X.i64 -o X.flowchart.json
"""
import idapro  # must be the first import

import argparse
import hashlib
import json
import sys


def function_rows():
    """[{fva, name, bb_num, hashopcodes}] for every function in the open database."""
    import ida_funcs
    import ida_gdl
    import ida_ua
    import idautils
    import idc

    rows = []
    for fva in idautils.Functions():
        func = ida_funcs.get_func(fva)
        blocks = sorted((bb.start_ea, bb.end_ea) for bb in ida_gdl.FlowChart(func))
        mnemonics = []
        for start, end in blocks:
            ea = start
            while ea < end:
                mnemonics.append(ida_ua.ua_mnem(ea) or '')
                ea = idc.next_head(ea)
        rows.append({
            'fva': fva,
            'name': ida_funcs.get_func_name(fva),
            'bb_num': len(blocks),
            'hashopcodes': hashlib.sha256(''.join(mnemonics).upper().encode()).hexdigest(),
        })
    return rows


def flowchart(database: str) -> list[dict]:
    """Open an analyzed database without re-analysis and return function_rows(); the database is not saved."""
    if idapro.open_database(database, False) != 0:
        raise RuntimeError(f'cannot open {database}')
    try:
        return function_rows()
    finally:
        idapro.close_database(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('database', help='analyzed IDA database (.i64)')
    parser.add_argument('-o', '--output', required=True, help='output JSON file')
    args = parser.parse_args()
    rows = flowchart(args.database)
    with open(args.output, 'w') as f:
        json.dump({'database': args.database, 'functions': rows}, f)
    print(f'{len(rows)} functions, {sum(r["bb_num"] >= 5 for r in rows)} with at least five basic blocks',
          file=sys.stderr)


if __name__ == '__main__':
    main()
