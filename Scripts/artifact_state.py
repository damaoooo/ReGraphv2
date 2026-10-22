"""
Pipeline artifact state shared by ida2llvm, task1/2/3, GraphBuilder and DataProcess.

Every published artifact has a JSON state file next to it that records the run status, the
identity of the inputs it was made from (content hashes, upstream state), the configuration
and the identity (sha256/size) of the published output. A consumer accepts an artifact only
if its state is "ok", the file on disk still has the recorded hash, and the upstream identity
matches what it is about to use. A non-empty file alone is never trusted.

State files:
  <x>.ll.lift.json           ida2llvm (task1)
  <x>.bc.reopt.json          task2
  <x>_functions.extract.json task3 (sibling of the function directory)
"""
import glob
import hashlib
import json
import os
import shutil
import socket

SCHEMA = 1
STATUS_OK = "ok"
LIFT_SUFFIX = ".lift.json"
REOPT_SUFFIX = ".reopt.json"
EXTRACT_SUFFIX = ".extract.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(bufsize):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str) -> dict:
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "sha256": sha256_file(path)}


def fsync_dir(path: str):
    fd = os.open(path or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def tmp_path_for(path: str) -> str:
    """Temporary sibling of `path`, unique per host and process so concurrent writers never collide."""
    return f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_tmp(path: str) -> list:
    """Remove temporaries of `path` left by dead processes on this host (e.g. after SIGKILL)."""
    removed = []
    host = socket.gethostname()
    for tmp in glob.glob(glob.escape(path) + ".tmp.*"):
        parts = tmp[len(path) + len(".tmp."):].rsplit(".", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        tmp_host, pid = parts[0], int(parts[1])
        if tmp_host == host and not _pid_alive(pid):
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass
            removed.append(tmp)
    return removed


def write_json_atomic(path: str, data: dict):
    tmp = tmp_path_for(path)
    try:
        with open(tmp, "w") as handle:
            json.dump(data, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(os.path.dirname(os.path.abspath(path)))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path: str):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def lifter_id(lifter_path: str = os.path.join(SCRIPT_DIR, "ida2llvm.py")) -> str:
    """Identity of the lifter source: first 16 hex digits of its sha256 (same as ida2llvm.LIFTER_ID)."""
    with open(lifter_path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def ida_install_id(ida_dir: str | None) -> str | None:
    """
    Cheap identity of an IDA installation without opening a database: size and mtime of the
    kernel and decompiler libraries. Any IDA/Hex-Rays update changes it.
    """
    if not ida_dir or not os.path.isdir(ida_dir):
        return None
    digest = hashlib.sha256()
    names = ["libida.so"] + sorted(os.path.relpath(p, ida_dir) for p in glob.glob(os.path.join(ida_dir, "plugins", "hex*.so")))
    found = False
    for name in names:
        try:
            st = os.stat(os.path.join(ida_dir, name))
        except OSError:
            continue
        found = True
        digest.update(f"{name}:{st.st_size}:{st.st_mtime_ns};".encode())
    return digest.hexdigest()[:16] if found else None


def check_output(state: dict | None, output_path: str) -> tuple[bool, str]:
    """The artifact on disk is the one the state file describes, and that run succeeded."""
    if state is None:
        return False, "missing state file"
    if state.get("schema") != SCHEMA:
        return False, f"state schema {state.get('schema')} != {SCHEMA}"
    if state.get("status") != STATUS_OK:
        return False, f"status {state.get('status')}"
    recorded = state.get("output") or {}
    if not os.path.isfile(output_path):
        return False, "output missing"
    if os.path.getsize(output_path) != recorded.get("size"):
        return False, "output size differs from state (stale or partially published)"
    if sha256_file(output_path) != recorded.get("sha256"):
        return False, "output hash differs from state (stale or partially published)"
    return True, "ok"


def check_lift_output(ll_path: str, expected_contract: str | None = "source-layout") -> tuple[bool, str, dict | None]:
    """Validate a task1 .ll against its .lift.json; returns (ok, reason, state)."""
    state = read_json(ll_path + LIFT_SUFFIX)
    ok, reason = check_output(state, ll_path)
    if ok and expected_contract is not None and state.get("memory_contract") != expected_contract:
        ok, reason = False, f"memory_contract {state.get('memory_contract')} != {expected_contract}"
    if ok:
        recorded = state.get("input") or {}
        try:
            st = os.stat(recorded.get("path", ""))
        except OSError:
            st = None  # the .i64 may live elsewhere or be gone; the output itself is still verified
        if st is not None and (st.st_size, st.st_mtime_ns) != (recorded.get("size"), recorded.get("mtime_ns")):
            ok, reason = False, "lift input changed after this output was produced"
    return ok, reason, state


def write_failed_lift_state(ll_path: str, input_path: str, status: str, error: str, started: float):
    """
    Record a lift attempt that died before ida2llvm could write its own state (timeout kill, crash).
    The previous output stays on disk but is no longer described as this run's result.
    """
    state_path = ll_path + LIFT_SUFFIX
    previous = read_json(state_path)
    if previous is not None and os.path.getmtime(state_path) >= started:
        return  # the lifter itself recorded this attempt
    valid, reason = check_output(previous, ll_path) if os.path.exists(ll_path) else (False, "no previous output")
    st = os.stat(input_path) if os.path.exists(input_path) else None
    write_json_atomic(state_path, {
        "schema": SCHEMA,
        "stage": "lift",
        "status": status,
        "error": error[-2000:],
        "input": {"path": os.path.abspath(input_path), "size": st.st_size if st else None,
                  "mtime_ns": st.st_mtime_ns if st else None},
        "previous_output": {"path": ll_path, "valid": valid, "reason": reason,
                            "lifter_id": (previous or {}).get("lifter_id"),
                            "sha256": ((previous or {}).get("output") or {}).get("sha256")},
    })


def check_reopt_output(bc_path: str) -> tuple[bool, str, dict | None]:
    state = read_json(bc_path + REOPT_SUFFIX)
    ok, reason = check_output(state, bc_path)
    if ok:
        # the input it was made from must still be the recorded, valid lift output
        src = (state.get("input") or {}).get("path")
        if src and os.path.isfile(src) and sha256_file(src) != state["input"].get("sha256"):
            ok, reason = False, "input .ll changed after this .bc was produced"
    return ok, reason, state


def check_extract_dir(function_dir: str, verify_files: bool = True) -> tuple[bool, str, dict | None]:
    """A task3 function directory is valid if its state is ok and every listed file has the recorded hash."""
    function_dir = function_dir.rstrip(os.sep)
    state = read_json(function_dir + EXTRACT_SUFFIX)
    if state is None:
        return False, "missing extract state", None
    if state.get("schema") != SCHEMA or state.get("status") != STATUS_OK:
        return False, f"extract status {state.get('status')}", state
    if os.path.abspath(state.get("output_dir", "")) != os.path.abspath(function_dir):
        return False, "extract state describes a different directory", state
    if verify_files:
        for entry in state.get("functions", []):
            path = os.path.join(function_dir, entry["file"])
            if not os.path.isfile(path) or sha256_file(path) != entry["sha256"]:
                return False, f"function file {entry['file']} differs from extract state", state
    return True, "ok", state


def iter_function_dirs(root: str):
    """All task3 `<x>_functions` directories under root (temporaries excluded)."""
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if ".tmp." not in d]
        for name in dirnames:
            if name.endswith("_functions"):
                yield os.path.join(dirpath, name)
        dirnames[:] = [d for d in dirnames if not d.endswith("_functions")]


def _is_function_ir(name: str) -> bool:
    return name.endswith(".ll") and not name.endswith(("_purified.ll", "_instrumented.ll"))


def select_function_files(root: str, allow_legacy: bool = False, verify_files: bool = True):
    """
    Function IR files that downstream stages may use, with their provenance:
    {abs_path: {"sha256", "llvm_name", "lift", "input_state", "memory_contract", "extract_state"}}.
    Directories whose extract state is missing, failed or stale are rejected (counted by reason);
    stateless directories are only taken with allow_legacy and are marked "legacy-unverified".
    """
    from collections import Counter
    selected, rejected = {}, Counter()
    for function_dir in iter_function_dirs(root):
        ok, reason, state = check_extract_dir(function_dir, verify_files)
        if ok:
            for entry in state.get("functions", []):
                path = os.path.abspath(os.path.join(function_dir, entry["file"]))
                selected[path] = {
                    "sha256": entry["sha256"],
                    "llvm_name": entry.get("llvm_name"),
                    "lift": entry.get("lift"),
                    "input_state": state["input"].get("state"),
                    "memory_contract": state["input"].get("memory_contract"),
                    "extract_state": function_dir + EXTRACT_SUFFIX,
                }
        elif state is None and allow_legacy:
            for name in sorted(os.listdir(function_dir)):
                if _is_function_ir(name):
                    path = os.path.abspath(os.path.join(function_dir, name))
                    selected[path] = {"sha256": sha256_file(path), "llvm_name": None, "lift": None,
                                      "input_state": "legacy-unverified", "memory_contract": None,
                                      "extract_state": None}
        else:
            rejected[reason] += 1
    return selected, rejected
