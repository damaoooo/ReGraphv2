"""Artifact state across the pipeline: publication, resume and downstream rejection.

Covers the audit's C12/C13/C14/C15 cases and the follow-up contract:
  - ida2llvm publishes <x>.ll only after llvm-as verification, with a .lift.json carrying the
    output hash; a failed rerun keeps the previous output but marks the run failed;
  - interrupted publication (new .ll, old sidecar), no valid functions, verifier failure,
    stale temporaries, and resume identity (lifter / IDA install / target mode / input / output);
  - task2/task3/GraphBuilder/DataProcess only consume verified upstream artifacts, legacy inputs
    need an explicit flag, and the unsafe bitcast-store fallback is off by default;
  - job_runner timeouts kill the whole process group and concurrency stays bounded.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_ida2llvm_microcode import L  # noqa: F401

SCRIPTS = Path(__file__).resolve().parents[1] / "Scripts"
sys.path.insert(0, str(SCRIPTS))
import artifact_state  # noqa: E402
import job_runner  # noqa: E402
import task1_lift  # noqa: E402
import task2_reoptimize  # noqa: E402
import task3_extract  # noqa: E402
from DataProcess import provenance  # noqa: E402

LLVM_TOOLS = all(shutil.which(t) for t in ("llvm-as", "opt", "llvm-dis", "llvm-nm", "llvm-extract", "clang"))
needs_llvm = pytest.mark.skipif(not LLVM_TOOLS, reason="LLVM 18 command line tools not installed")

GOOD_FN = "define i32 @good(i32 %a) {\nentry:\n  %b = add i32 %a, 1\n  ret i32 %b\n}\n"
OTHER_FN = "define i32 @other(i32 %a) {\nentry:\n  %b = mul i32 %a, 3\n  ret i32 %b\n}\n"
BROKEN_FN = "define i32 @broken() {\nentry:\n}\n"


# ------------------------------------------------------------------------------ lift publication

@pytest.fixture
def lift_env(L, monkeypatch, tmp_path):
    """run_lift with a stubbed IDA: the test decides what insertAllFunctions streams."""
    for name in ("open_database", "close_database"):
        monkeypatch.setattr(L.idapro, name, lambda *args: 0)
    monkeypatch.setattr(L.ida_auto, "auto_wait", lambda: None)

    def init_target():
        L.ptrsize, L.is_big_endian, L.byteorder = 64, False, "little"
    monkeypatch.setattr(L, "_init_target_state", init_target)
    monkeypatch.setattr(L.ida_ida, "inf_get_procname", lambda: "metapc", raising=False)
    monkeypatch.setattr(L.ida_ida, "inf_get_filetype", lambda: L.ida_ida.f_ELF, raising=False)
    src = tmp_path / "bin.i64"
    src.write_bytes(b"database")
    out = tmp_path / "bin.ll"

    def run(body, defined=1, records=(), **kwargs):
        class Controller(L.BIN2LLVMController):
            def initialize(self):
                pass

            def insertAllFunctions(self):
                if isinstance(body, Exception):
                    self._stream_file.write(GOOD_FN)
                    raise body
                self._stream_file.write(body)
                L.lift_stats["defined"] = defined
                L.function_records.extend(records)
        monkeypatch.setattr(L, "BIN2LLVMController", Controller)
        return L.run_lift(str(src), str(out), **kwargs)
    return run, src, out


@needs_llvm
def test_success_publishes_verified_output_with_hash(lift_env, L):
    run, src, out = lift_env
    result = run(GOOD_FN, records=[{"ea": "0x10", "name": "good", "status": "lifted", "llvm_name": "good"}])
    assert result["status"] == "ok" and L._exit_code(result) == 0
    ok, reason, state = artifact_state.check_lift_output(str(out))
    assert ok, reason
    assert state["memory_contract"] == "source-layout" and state["target"]["triple"] == "x86_64-unknown-linux-gnu"
    assert out.read_text().startswith(f"; ida2llvm lifter={L.LIFTER_ID}")
    assert not list(out.parent.glob("*.tmp.*"))


@needs_llvm
def test_failed_rerun_keeps_previous_output_but_marks_run_failed(lift_env, L):
    run, src, out = lift_env
    assert run(GOOD_FN)["status"] == "ok"
    before = out.read_bytes()
    result = run(RuntimeError("simulated crash"))
    assert result["status"] == "error" and L._exit_code(result) == 1
    assert out.read_bytes() == before  # previous success is not deleted
    assert result["previous_output"]["valid"] is True
    ok, reason, _ = artifact_state.check_lift_output(str(out))
    assert not ok and "status error" in reason  # but downstream does not take it as this run's output
    usable, why, _ = task2_reoptimize.check_task2_input(str(out), False, False)
    assert not usable


@needs_llvm
def test_output_published_but_sidecar_not_written_is_rejected(lift_env, L, monkeypatch):
    run, src, out = lift_env
    assert run(GOOD_FN)["status"] == "ok"

    def crash(*args, **kwargs):
        raise OSError("killed before the sidecar was written")
    monkeypatch.setattr(L.artifact_state, "write_json_atomic", crash)
    run(GOOD_FN + OTHER_FN)
    assert "@other" in out.read_text()  # the new .ll was renamed into place ...
    ok, reason, _ = artifact_state.check_lift_output(str(out))
    assert not ok and "differs from state" in reason  # ... and the old sidecar no longer describes it
    assert not task1_lift.lift_output_reusable(str(src), str(out), L.LIFTER_ID)


@needs_llvm
def test_no_valid_functions_publishes_nothing(lift_env, L):
    run, src, out = lift_env
    result = run("", defined=0)
    assert result["status"] == "no_functions_lifted" and L._exit_code(result) == 3
    assert not out.exists()


@needs_llvm
def test_verifier_failure_publishes_nothing(lift_env, L):
    run, src, out = lift_env
    result = run(BROKEN_FN)
    assert result["status"] == "verify_failed" and "error" in result
    assert not out.exists()


def test_missing_verifier_tool_publishes_nothing(lift_env, L, monkeypatch):
    run, src, out = lift_env
    monkeypatch.setenv("REGRAPH_LLVM_AS", "/nonexistent/llvm-as")
    result = run(GOOD_FN)
    assert result["status"] == "verify_failed" and "not found" in result["error"] and not out.exists()


@needs_llvm
def test_stale_temporaries_of_dead_processes_are_removed(lift_env, L):
    run, src, out = lift_env
    import socket
    dead = subprocess.Popen(["true"])
    dead.wait()
    stale = Path(f"{out}.tmp.{socket.gethostname()}.{dead.pid}")
    stale.write_text("partial")
    run(GOOD_FN)
    assert not stale.exists()


@needs_llvm
def test_resume_identity(lift_env, L, monkeypatch):
    run, src, out = lift_env
    monkeypatch.setenv("IDADIR", str(src.parent))  # no libida.so: install id None
    run(GOOD_FN)
    lifter = task1_lift.ida2llvm_lifter_id()
    assert lifter == L.LIFTER_ID
    assert task1_lift.lift_output_reusable(str(src), str(out), lifter, "source", None)
    assert not task1_lift.lift_output_reusable(str(src), str(out), "0" * 16, "source")  # lifter changed
    assert not task1_lift.lift_output_reusable(str(src), str(out), lifter, "host")  # target option changed
    state = json.loads((out.parent / "bin.ll.lift.json").read_text())
    state["ida_install_id"] = "previous-ida"
    (out.parent / "bin.ll.lift.json").write_text(json.dumps(state))
    assert not task1_lift.lift_output_reusable(str(src), str(out), lifter, "source", "current-ida")  # IDA changed
    st = os.stat(src)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert not task1_lift.lift_output_reusable(str(src), str(out), lifter, "source")  # input changed


def test_ida_install_id_tracks_library_changes(tmp_path):
    (tmp_path / "plugins").mkdir()
    lib = tmp_path / "libida.so"
    lib.write_bytes(b"a")
    (tmp_path / "plugins" / "hexx64.so").write_bytes(b"b")
    first = artifact_state.ida_install_id(str(tmp_path))
    lib.write_bytes(b"aa")
    assert first and artifact_state.ida_install_id(str(tmp_path)) != first
    assert artifact_state.ida_install_id(str(tmp_path / "missing")) is None


# ------------------------------------------------------------------------------ task2 / task3

def make_lifted(tmp_path, name, body, triple="x86_64-unknown-linux-gnu", contract="source-layout", llvm_names=()):
    """A task1 output with a valid .lift.json, as ida2llvm would publish it."""
    ll = tmp_path / f"{name}.ll"
    ll.write_text(f'target triple = "{triple}"\n\n' + body)
    state = {"schema": artifact_state.SCHEMA, "stage": "lift", "status": "ok", "lifter_id": "test",
             "memory_contract": contract, "target": {"triple": triple},
             "output": {"path": str(ll), "size": ll.stat().st_size, "sha256": artifact_state.sha256_file(str(ll))},
             "functions": [{"name": n, "llvm_name": n, "status": "lifted", "approx": ["virtual_fs"] if n == "good" else []}
                           for n in llvm_names]}
    (tmp_path / f"{name}.ll.lift.json").write_text(json.dumps(state))
    return ll


def test_task2_input_requires_verified_source_layout(tmp_path):
    ll = make_lifted(tmp_path, "a", GOOD_FN)
    assert task2_reoptimize.check_task2_input(str(ll), False, False)[0]
    bare = tmp_path / "legacy.ll"
    bare.write_text(GOOD_FN)
    assert not task2_reoptimize.check_task2_input(str(bare), False, False)[0]
    usable, reason, info = task2_reoptimize.check_task2_input(str(bare), False, True)
    assert usable and info["state"] == "legacy-unverified"
    host = make_lifted(tmp_path, "h", GOOD_FN, contract="host-compat-unconverted")
    assert not task2_reoptimize.check_task2_input(str(host), False, False)[0]
    assert task2_reoptimize.check_task2_input(str(host), True, False)[0]
    ll.write_text(GOOD_FN + OTHER_FN)  # content no longer matches its state
    assert not task2_reoptimize.check_task2_input(str(ll), False, False)[0]


def reopt(ll, opt_level="-Oc", allow_host_retarget=False, unsafe=False, arch=None):
    _, _, info = task2_reoptimize.check_task2_input(str(ll), True, True)
    config = task2_reoptimize.task2_config(opt_level, allow_host_retarget, unsafe)
    bc = ll.with_suffix(".bc")
    ok, _, error = task2_reoptimize.reoptimize_file(str(ll), str(bc), opt_level, arch, info, config)
    return ok, error, bc, task2_reoptimize.config_token(config)


@needs_llvm
def test_task2_publishes_state_and_resume_tracks_input(tmp_path):
    ll = make_lifted(tmp_path, "a", GOOD_FN)
    ok, error, bc, token = reopt(ll)
    assert ok, error
    assert artifact_state.check_reopt_output(str(bc))[0]
    assert task2_reoptimize.reopt_output_reusable(str(ll), str(bc), token)
    ll.write_text(ll.read_text() + OTHER_FN)
    assert not task2_reoptimize.reopt_output_reusable(str(ll), str(bc), token)
    assert not artifact_state.check_reopt_output(str(bc))[0]  # input changed after this .bc was made


@needs_llvm
def test_task2_failure_keeps_old_bc_but_marks_failed(tmp_path):
    ll = make_lifted(tmp_path, "a", GOOD_FN)
    ok, _, bc, _ = reopt(ll)
    before = bc.read_bytes()
    ll.write_text('target triple = "x86_64-unknown-linux-gnu"\n' + BROKEN_FN)
    ok, error, bc, _ = reopt(ll)
    assert not ok and bc.read_bytes() == before
    assert not artifact_state.check_reopt_output(str(bc))[0]


ILLEGAL = ("define void @broken(void ()** %p) {\nentry:\n  %f = bitcast i64 1 to void ()\n"
           "  store void ()* %f, void ()** %p\n  ret void\n}\n")


@needs_llvm
def test_unsafe_bitcast_fallback_is_off_by_default(tmp_path):
    ll = make_lifted(tmp_path, "a", ILLEGAL + GOOD_FN)
    ok, error, bc, _ = reopt(ll)
    assert not ok and "@broken" not in error and "broken" in error and not bc.exists()
    ok, error, bc, _ = reopt(ll, unsafe=True)
    state = json.loads(Path(str(bc) + ".reopt.json").read_text())
    assert ok and state["approx"] == ["dropped_illegal_bitcast_stores"]


@needs_llvm
@pytest.mark.parametrize("triple", ["mips-unknown-linux-gnu", "i386-unknown-linux-gnu", "aarch64-unknown-linux-gnu"])
def test_clang_path_keeps_module_target(tmp_path, triple):
    ll = make_lifted(tmp_path, "a", GOOD_FN, triple=triple)
    ok, error, bc, _ = reopt(ll, opt_level="-O2")
    assert ok, error
    from utils import read_ir_target
    assert read_ir_target(str(bc))[0] == triple
    ok, error, bc, _ = reopt(ll, opt_level="-O2", allow_host_retarget=True, arch="m64")
    state = json.loads(Path(str(bc) + ".reopt.json").read_text())
    assert ok and state["target"]["retargeted_to_host"] is True  # legacy behaviour only on request, recorded


@needs_llvm
def test_task3_replaces_function_dir_and_joins_lift_records(tmp_path):
    ll = make_lifted(tmp_path, "a", GOOD_FN + OTHER_FN, llvm_names=("good", "other"))
    assert reopt(ll)[0]
    bc = ll.with_suffix(".bc")
    usable, _, info = task3_extract.check_task3_input(str(bc), False)
    assert usable
    fdir = tmp_path / "a_functions"
    ok, _, error = task3_extract.extract_functions(str(bc), str(fdir), info)
    assert ok, error
    ok, _, state = artifact_state.check_extract_dir(str(fdir))
    assert ok and {f["llvm_name"] for f in state["functions"]} == {"good", "other"}
    assert {f["llvm_name"]: f["lift"]["approx"] for f in state["functions"]}["good"] == ["virtual_fs"]
    old_files = {f["file"] for f in state["functions"]}

    # new upstream content: the stale function file of @other must not survive
    ll2 = make_lifted(tmp_path, "a", GOOD_FN, llvm_names=("good",))
    assert reopt(ll2)[0]
    usable, _, info = task3_extract.check_task3_input(str(bc), False)
    ok, _, error = task3_extract.extract_functions(str(bc), str(fdir), info)
    files_now = {p.name for p in fdir.glob("*.ll")}
    assert ok and len(files_now) == 1 and files_now < old_files

    # a failed extraction leaves the directory but its state makes consumers reject it
    os.remove(str(bc))
    Path(bc).write_bytes(b"not bitcode")
    ok, _, _ = task3_extract.extract_functions(str(bc), str(fdir), info)
    assert not ok and fdir.exists() and not artifact_state.check_extract_dir(str(fdir))[0]


@needs_llvm
def test_task3_rejects_unverified_bitcode(tmp_path):
    ll = make_lifted(tmp_path, "a", GOOD_FN)
    assert reopt(ll)[0]
    bc = ll.with_suffix(".bc")
    bc.write_bytes(bc.read_bytes() + b"\0")  # changed after task2 published it
    assert not task3_extract.check_task3_input(str(bc), False)[0]


# --------------------------------------------------------------------- GraphBuilder / DataProcess

def make_function_dir(root: Path, name: str, functions: dict, approx=None, status="ok"):
    fdir = root / f"{name}_functions"
    fdir.mkdir(parents=True)
    entries = []
    for llvm_name, text in functions.items():
        path = fdir / f"{llvm_name}.ll"
        path.write_text(text)
        entries.append({"llvm_name": llvm_name, "file": path.name, "sha256": artifact_state.sha256_file(str(path)),
                        "lift": {"status": "lifted", "approx": (approx or {}).get(llvm_name, [])}})
    state = {"schema": artifact_state.SCHEMA, "stage": "extract", "status": status, "output_dir": str(fdir),
             "input": {"state": "verified", "memory_contract": "source-layout"}, "functions": entries}
    (root / f"{name}_functions.extract.json").write_text(json.dumps(state))
    return fdir


def test_function_selection_requires_valid_extract_state(tmp_path):
    good = make_function_dir(tmp_path, "a", {"f": GOOD_FN})
    make_function_dir(tmp_path, "b", {"g": OTHER_FN}, status="failed")
    legacy = tmp_path / "c_functions"
    legacy.mkdir()
    (legacy / "h.ll").write_text(GOOD_FN)
    selected, rejected = artifact_state.select_function_files(str(tmp_path))
    assert set(selected) == {str(good / "f.ll")} and sum(rejected.values()) == 2
    selected, _ = artifact_state.select_function_files(str(tmp_path), allow_legacy=True)
    assert selected[str(legacy / "h.ll")]["input_state"] == "legacy-unverified"
    (good / "f.ll").write_text(OTHER_FN)  # tampered after extraction
    selected, rejected = artifact_state.select_function_files(str(tmp_path))
    assert not selected


def make_results_db(root: Path, rows, with_hash=True):
    db = root / "results.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE results (id INTEGER PRIMARY KEY, input_path TEXT UNIQUE, purify_path TEXT, "
                     "instrumented_path TEXT, cfg_dot TEXT, ddg_dot TEXT" + (", input_sha256 TEXT)" if with_hash else ")"))
        for path, digest in rows:
            if with_hash:
                conn.execute("INSERT INTO results (input_path, input_sha256) VALUES (?, ?)", (path, digest))
            else:
                conn.execute("INSERT INTO results (input_path) VALUES (?)", (path,))
    return str(db)


def test_dataprocess_gate_and_selection_policy(tmp_path):
    fdir = make_function_dir(tmp_path, "a", {"f": GOOD_FN, "g": OTHER_FN, "k": GOOD_FN.replace("good", "k")},
                             approx={"f": ["virtual_fs"], "g": ["opaque_ext:cpuid"]})
    sha = {name: artifact_state.sha256_file(str(fdir / f"{name}.ll")) for name in "fgk"}
    db = make_results_db(tmp_path, [(str(fdir / "f.ll"), sha["f"]), (str(fdir / "g.ll"), sha["g"]),
                                    (str(fdir / "k.ll"), "stale-hash"), (str(tmp_path / "gone.ll"), "x")])
    keep_all = provenance.load_policy(provenance.DEFAULT_POLICY)
    chosen, rows, summary = provenance.select_inputs(str(tmp_path), db, keep_all)
    assert sorted(os.path.basename(p) for p in chosen) == ["f.ll", "g.ll"]  # approx kept: no selection bias
    assert summary["rejected_for_integrity"] == {"row_generated_from_other_content": 1, "row_not_in_valid_function_dir": 1}
    assert summary["approx_tags_selected"] == {"virtual_fs": 1, "opaque_ext:cpuid": 1}
    strict = provenance.load_policy(str(Path(provenance.DEFAULT_POLICY).with_name("exclude_opaque_v1.json")))
    chosen, rows, summary = provenance.select_inputs(str(tmp_path), db, strict)
    assert [os.path.basename(p) for p in chosen] == ["f.ll"]
    assert summary["excluded_by_policy"] == {"approx:opaque_ext:*": 1} and summary["policy_sha256"]
    provenance.write_selection(str(tmp_path / "out"), rows, summary)
    assert len((tmp_path / "out" / "provenance.jsonl").read_text().splitlines()) == 2


def test_dataprocess_rejects_legacy_rows_without_flag(tmp_path):
    fdir = make_function_dir(tmp_path, "a", {"f": GOOD_FN})
    db = make_results_db(tmp_path, [(str(fdir / "f.ll"), None)], with_hash=False)
    policy = provenance.load_policy(provenance.DEFAULT_POLICY)
    chosen, _, summary = provenance.select_inputs(str(tmp_path), db, policy)
    assert not chosen and summary["rejected_for_integrity"] == {"row_without_input_sha256": 1}


# ------------------------------------------------------------------------------ resource control

def test_timeout_kills_whole_process_group(tmp_path):
    marker = tmp_path / "child.pid"
    cmd = ["sh", "-c", f"sleep 60 & echo $! > {marker}; wait"]
    started = time.time()
    result = job_runner.run_logged(cmd, str(tmp_path / "job.log"), timeout_s=1, grace_s=1)
    assert result.timed_out and not result.ok and time.time() - started < 10
    child = int(marker.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)  # the background grandchild was killed with the group
    assert "killed process group" in job_runner.log_tail(result.log_path)


def test_output_goes_to_log_not_memory(tmp_path):
    result = job_runner.run_logged(["sh", "-c", "head -c 20000000 /dev/zero | tr '\\0' x"], str(tmp_path / "big.log"))
    assert result.ok and os.path.getsize(tmp_path / "big.log") > 20_000_000
    assert len(job_runner.log_tail(result.log_path, 1000)) <= 1000


def test_bounded_concurrency_and_memory_gate(monkeypatch):
    import threading
    running, peak, lock = [0], [0], threading.Lock()

    def launch(item):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return item

    done = []
    job_runner.run_bounded(range(20), launch, workers=3, on_done=lambda i, r: done.append(r), poll_s=0.01)
    assert sorted(done) == list(range(20)) and peak[0] <= 3
    running[0] = peak[0] = 0
    monkeypatch.setattr(job_runner, "mem_available_mb", lambda: 100)  # below the per-job estimate
    job_runner.run_bounded(range(5), launch, workers=4, mem_per_job_mb=1000, poll_s=0.01)
    assert peak[0] == 1  # memory pressure: one job at a time, but still progressing


@needs_llvm
def test_killed_lift_is_recorded_and_previous_output_kept(lift_env, L):
    run, src, out = lift_env
    run(GOOD_FN)
    before = out.read_bytes()
    started = time.time() + 1  # the state file predates this (killed) attempt
    artifact_state.write_failed_lift_state(str(out), str(src), "timeout", "killed after 4s", started)
    state = json.loads(Path(str(out) + ".lift.json").read_text())
    assert state["status"] == "timeout" and state["previous_output"]["valid"] is True
    assert out.read_bytes() == before and not artifact_state.check_lift_output(str(out))[0]


@needs_llvm
def test_lifter_written_state_is_not_overwritten(lift_env, L):
    run, src, out = lift_env
    started = time.time() - 1
    run("", defined=0)  # the lifter wrote its own failure state during this attempt
    artifact_state.write_failed_lift_state(str(out), str(src), "crashed", "rc=3", started)
    assert json.loads(Path(str(out) + ".lift.json").read_text())["status"] == "no_functions_lifted"


@needs_llvm
def test_output_of_a_changed_lift_input_is_rejected(lift_env, L):
    run, src, out = lift_env
    run(GOOD_FN)
    assert artifact_state.check_lift_output(str(out))[0]
    src.write_bytes(b"database v2")
    ok, reason, _ = artifact_state.check_lift_output(str(out))
    assert not ok and "input changed" in reason
