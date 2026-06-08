#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load_ida2llvm(repo_root: Path):
    scripts_dir = repo_root / "Scripts"
    sys.path.insert(0, str(scripts_dir))
    import ida2llvm  # noqa: PLC0415

    return ida2llvm


def _count_defined_functions(ll_path: Path) -> int:
    if not ll_path.exists():
        return 0
    count = 0
    with ll_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("define "):
                count += 1
    return count


def _close_database() -> None:
    try:
        import idapro  # noqa: PLC0415

        idapro.close_database()
    except Exception:
        pass


def list_functions(repo_root: Path, binary: Path) -> int:
    ida2llvm = _load_ida2llvm(repo_root)
    import ida_auto  # noqa: PLC0415
    import ida_name  # noqa: PLC0415
    import idapro  # noqa: PLC0415
    import idautils  # noqa: PLC0415

    start = time.perf_counter()
    ida_open_seconds = 0.0
    functions: list[dict[str, object]] = []
    try:
        ida2llvm._reset_lift_state()
        open_start = time.perf_counter()
        idapro.open_database(str(binary), True)
        ida_auto.auto_wait()
        ida_open_seconds = time.perf_counter() - open_start

        for ea in idautils.Functions():
            functions.append(
                {
                    "ea": int(ea),
                    "name": ida_name.get_name(ea) or f"sub_{int(ea):x}",
                }
            )

        result = {
            "ok": True,
            "binary": str(binary),
            "elapsed_s": time.perf_counter() - start,
            "ida_open_seconds": ida_open_seconds,
            "function_count": len(functions),
            "functions": functions,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "binary": str(binary),
            "elapsed_s": time.perf_counter() - start,
            "ida_open_seconds": ida_open_seconds,
            "error": repr(exc),
            "function_count": len(functions),
            "functions": functions,
        }
    finally:
        _close_database()

    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


def lift_functions(
    repo_root: Path,
    binary: Path,
    output: Path,
    eas: list[int],
    target_mode: str,
) -> int:
    ida2llvm = _load_ida2llvm(repo_root)
    import ida_auto  # noqa: PLC0415
    import idapro  # noqa: PLC0415

    start = time.perf_counter()
    ida_open_seconds = 0.0
    controller_initialize_seconds = 0.0
    function_emit_seconds = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        ida2llvm._reset_lift_state()
        open_start = time.perf_counter()
        idapro.open_database(str(binary), True)
        ida_auto.auto_wait()
        ida_open_seconds = time.perf_counter() - open_start

        initialize_start = time.perf_counter()
        controller = ida2llvm.BIN2LLVMController(target_mode=target_mode)
        controller.initialize()
        controller_initialize_seconds = time.perf_counter() - initialize_start

        emit_start = time.perf_counter()
        controller.begin_stream_to_file(str(output))
        for ea in eas:
            controller.insertFunctionAtEa(int(ea))
        controller.finish_stream_to_file()
        function_emit_seconds = time.perf_counter() - emit_start

        defined = _count_defined_functions(output)
        result = {
            "ok": defined > 0,
            "binary": str(binary),
            "output": str(output),
            "requested_functions": len(eas),
            "defined_functions": defined,
            "elapsed_s": time.perf_counter() - start,
            "ida_open_seconds": ida_open_seconds,
            "controller_initialize_seconds": controller_initialize_seconds,
            "function_emit_seconds": function_emit_seconds,
            "steady_state_lift_seconds": controller_initialize_seconds + function_emit_seconds,
        }
        if defined <= 0:
            result["error"] = "no functions were emitted"
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "binary": str(binary),
            "output": str(output),
            "requested_functions": len(eas),
            "defined_functions": _count_defined_functions(output),
            "elapsed_s": time.perf_counter() - start,
            "ida_open_seconds": ida_open_seconds,
            "controller_initialize_seconds": controller_initialize_seconds,
            "function_emit_seconds": function_emit_seconds,
            "steady_state_lift_seconds": controller_initialize_seconds + function_emit_seconds,
            "error": repr(exc),
        }
    finally:
        _close_database()

    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="IDA subprocess worker for latency benchmarks.")
    parser.add_argument("--repo-root", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--binary", required=True, type=Path)

    lift_parser = subparsers.add_parser("lift")
    lift_parser.add_argument("--binary", required=True, type=Path)
    lift_parser.add_argument("--output", required=True, type=Path)
    lift_parser.add_argument("--eas", required=True)
    lift_parser.add_argument("--target-mode", default="host")

    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.command == "list":
        return list_functions(repo_root, args.binary.resolve())
    if args.command == "lift":
        eas = [int(value, 0) for value in args.eas.split(",") if value]
        return lift_functions(
            repo_root=repo_root,
            binary=args.binary.resolve(),
            output=args.output.resolve(),
            eas=eas,
            target_mode=args.target_mode,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
