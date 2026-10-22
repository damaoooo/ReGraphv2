"""
Bounded subprocess execution for the dataset pipeline.

- Each job runs in its own session/process group; on timeout the whole group gets SIGTERM and,
  after a grace period, SIGKILL (IDA and opt spawn children that must not be orphaned).
- stdout/stderr go straight to a per-job log file instead of being captured in memory; callers
  get only the tail of that log.
- run_bounded keeps at most `workers` jobs in flight, submits lazily (no queue of thousands of
  futures) and waits before launching while MemAvailable is below the per-job estimate.
"""
import os
import signal
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass


@dataclass
class JobResult:
    returncode: int | None
    timed_out: bool
    elapsed_s: float
    log_path: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.error


def mem_available_mb() -> int | None:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def default_workers(mem_per_job_mb: int, reserve_mb: int = 8192) -> int:
    """Conservative default: half the CPUs, further limited by available memory."""
    cpus = max(1, (os.cpu_count() or 2) // 2)
    available = mem_available_mb()
    if available is None or mem_per_job_mb <= 0:
        return cpus
    return max(1, min(cpus, (available - reserve_mb) // mem_per_job_mb))


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_group(proc: subprocess.Popen, grace_s: float):
    """
    SIGTERM the job's process group; after the grace period SIGKILL whatever is left of the group,
    also when the leader already exited (children that ignore SIGTERM would otherwise survive).
    Only the job's own group is signalled: it was created by start_new_session, so its id is the
    leader's pid, which the kernel does not reuse while any member of the group exists.
    """
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None and not _group_alive(pgid):
            break
        time.sleep(min(0.05, grace_s))
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def reap_group(proc: subprocess.Popen):
    """After a normal exit, kill members the job left behind in its group (none are expected)."""
    if _group_alive(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True
    return False


def run_logged(cmd, log_path: str, timeout_s: float | None = None, env=None, cwd=None,
               grace_s: float = 30.0) -> JobResult:
    """Run `cmd` with output appended to `log_path`; kill its process group on timeout."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    started = time.time()
    with open(log_path, "a") as log:
        log.write(f"$ {' '.join(map(str, cmd))}\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=env, cwd=cwd, start_new_session=True)
        except OSError as exc:
            return JobResult(None, False, 0.0, log_path, error=str(exc))
        try:
            returncode = proc.wait(timeout=timeout_s if timeout_s and timeout_s > 0 else None)
            timed_out = False
            if reap_group(proc):
                log.write("\n[job_runner] killed processes left in the job's group after it exited\n")
        except subprocess.TimeoutExpired:
            kill_group(proc, grace_s)
            returncode, timed_out = proc.returncode, True
            log.write(f"\n[job_runner] killed process group after {timeout_s}s timeout\n")
        log.write(f"[job_runner] returncode={returncode} elapsed={time.time() - started:.1f}s\n")
    return JobResult(returncode, timed_out, time.time() - started, log_path)


def log_tail(log_path: str, max_bytes: int = 4000) -> str:
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def run_bounded(items, launch, workers: int, mem_per_job_mb: int = 0, on_done=None, poll_s: float = 5.0):
    """
    Call launch(item) for every item with at most `workers` concurrently running. Before each launch,
    wait while MemAvailable < mem_per_job_mb (unless nothing is running, so progress is guaranteed).
    on_done(item, result_or_exception) is called in the main thread as jobs finish.
    """
    items = iter(items)
    inflight = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        exhausted = False
        while True:
            while not exhausted and len(inflight) < workers:
                if mem_per_job_mb > 0 and inflight:
                    available = mem_available_mb()
                    if available is not None and available < mem_per_job_mb:
                        break
                try:
                    item = next(items)
                except StopIteration:
                    exhausted = True
                    break
                inflight[pool.submit(launch, item)] = item
            if not inflight:
                if exhausted:
                    return
                time.sleep(poll_s)
                continue
            done, _ = wait(list(inflight), timeout=poll_s, return_when=FIRST_COMPLETED)
            for future in done:
                item = inflight.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:  # report, keep the pipeline going
                    outcome = exc
                if on_done is not None:
                    on_done(item, outcome)
