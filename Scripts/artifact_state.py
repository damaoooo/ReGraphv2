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


def _upstream_missing(path: str, allow_detached: bool, what: str) -> tuple[bool, str]:
    """
    Detached artifacts: if the recorded upstream file no longer exists (e.g. only .bc/function files
    were copied elsewhere), the artifact can only be vouched for by its own state. That is refused by
    default and accepted only with allow_detached, where the reason says so explicitly.
    """
    if allow_detached:
        return True, f"detached: {what} {path} missing, upstream not re-validated"
    return False, f"upstream {what} {path} missing (detached artifact)"


def check_reopt_output(bc_path: str, allow_detached: bool = False) -> tuple[bool, str, dict | None]:
    """
    A task2 .bc is valid if its state is ok, the .bc has the recorded hash, and the upstream .ll it was
    made from is still exactly the recorded content AND still a valid task1 output whose .lift.json is
    byte-identical to the one recorded (a failed or changed upstream rerun invalidates the .bc).
    """
    state = read_json(bc_path + REOPT_SUFFIX)
    ok, reason = check_output(state, bc_path)
    if not ok:
        return ok, reason, state
    recorded = state.get("input") or {}
    src = recorded.get("path")
    if not src or not os.path.isfile(src):
        ok, reason = _upstream_missing(src, allow_detached, ".ll")
        return ok, reason, state
    if sha256_file(src) != recorded.get("sha256"):
        return False, "input .ll changed after this .bc was produced", state
    if recorded.get("state") == "legacy-unverified":
        return True, "ok (legacy-unverified upstream)", state
    lift_state_path = src + LIFT_SUFFIX
    if not os.path.isfile(lift_state_path):
        return False, "upstream .lift.json missing", state
    if "lift_state_sha256" in recorded and sha256_file(lift_state_path) != recorded["lift_state_sha256"]:
        return False, "upstream .lift.json changed after this .bc was produced (rerun, failure or new provenance)", state
    lift_ok, lift_reason, _ = check_lift_output(src, recorded.get("memory_contract"))
    if not lift_ok:
        return False, f"upstream lift no longer valid: {lift_reason}", state
    return True, "ok", state


def check_extract_dir(function_dir: str, verify_files: bool = True,
                      allow_detached: bool = False) -> tuple[bool, str, dict | None]:
    """
    A task3 function directory is valid if its state is ok, every listed file has the recorded hash,
    and the .bc it was split from is still the recorded content and a valid task2 output whose
    .reopt.json is the recorded one (recursively validating the .ll and .lift.json behind it).
    """
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
    recorded = state.get("input") or {}
    bc = recorded.get("path")
    if not bc or not os.path.isfile(bc):
        ok, reason = _upstream_missing(bc, allow_detached, ".bc")
        return ok, reason, state
    if sha256_file(bc) != recorded.get("sha256"):
        return False, "input .bc changed after these functions were extracted", state
    if recorded.get("state") == "legacy-unverified" and not os.path.exists(bc + REOPT_SUFFIX):
        return True, "ok (legacy-unverified upstream)", state
    if "reopt_state_sha256" in recorded:
        if not os.path.isfile(bc + REOPT_SUFFIX) or sha256_file(bc + REOPT_SUFFIX) != recorded["reopt_state_sha256"]:
            return False, "upstream .reopt.json changed after extraction (rerun, failure or new provenance)", state
    reopt_ok, reopt_reason, _ = check_reopt_output(bc, allow_detached)
    if not reopt_ok:
        return False, f"upstream reopt no longer valid: {reopt_reason}", state
    return True, reopt_reason if reopt_reason.startswith("detached") else "ok", state


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


def select_function_files(root: str, allow_legacy: bool = False, verify_files: bool = True,
                          allow_detached: bool = False):
    """
    Function IR files that downstream stages may use, with their provenance:
    {abs_path: {"sha256", "llvm_name", "lift", "input_state", "memory_contract", "extract_state"}}.
    Directories whose extract state is missing, failed or stale are rejected (counted by reason);
    stateless directories are only taken with allow_legacy and are marked "legacy-unverified".
    """
    from collections import Counter
    selected, rejected = {}, Counter()
    for function_dir in iter_function_dirs(root):
        ok, reason, state = check_extract_dir(function_dir, verify_files, allow_detached)
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
                    "extract_state_sha256": sha256_file(function_dir + EXTRACT_SUFFIX),
                    "detached": reason.startswith("detached"),
                }
        elif state is None and allow_legacy:
            for name in sorted(os.listdir(function_dir)):
                if _is_function_ir(name):
                    path = os.path.abspath(os.path.join(function_dir, name))
                    selected[path] = {"sha256": sha256_file(path), "llvm_name": None, "lift": None,
                                      "input_state": "legacy-unverified", "memory_contract": None,
                                      "extract_state": None, "extract_state_sha256": None, "detached": False}
        else:
            rejected[reason] += 1
    return selected, rejected
