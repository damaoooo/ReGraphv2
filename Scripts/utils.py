"""
Common utilities for the lift dataset pipeline
"""
import subprocess
import os
import re
import resource
import typer
from rich.console import Console

console = Console()


def normalize_clang_opt_level(opt_level: str) -> str:
    """Normalize a clang optimization level flag to the form ``-O*``."""
    value = opt_level.strip()
    if not value:
        raise typer.BadParameter("opt level cannot be empty")

    normalized = value if value.startswith("-") else f"-{value}"

    if " " in normalized or "\t" in normalized:
        raise typer.BadParameter("opt level cannot contain whitespace")
    if not normalized.startswith("-O") or len(normalized) <= 2:
        raise typer.BadParameter(
            "opt level must look like O0/O1/O2/O3/Os/Oz/Ofast or start with -O"
        )

    return normalized

def run_command(command, description="", timeout=3600, memory_limit_gb=10, tail_bytes=8000, kill_grace_s=30):
    """Run a command and return (success, stdout_tail, stderr_tail).

    Output is spooled to temporary files rather than held in memory, and only the last
    `tail_bytes` of each stream are returned. The command runs in its own process group,
    which is killed (SIGTERM, then SIGKILL) on timeout.

    Args:
        command: Command to execute
        description: Description of the command
        timeout: Timeout in seconds (default: 3600; 0 or None disables it)
        memory_limit_gb: Address-space limit in GB (default: 10)
    """
    import tempfile
    from job_runner import kill_group, reap_group

    def set_limits():
        memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

    def tail(handle):
        handle.flush()
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - tail_bytes))
        return handle.read().decode("utf-8", "replace")

    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            p = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                 preexec_fn=set_limits, start_new_session=True)
        except OSError as e:
            return False, "", str(e)
        try:
            p.wait(timeout=timeout or None)
            reap_group(p)
        except subprocess.TimeoutExpired:
            kill_group(p, grace_s=kill_grace_s)
            return False, tail(out), f"Command timed out after {timeout} seconds\n{tail(err)}"
        return p.returncode == 0, tail(out), tail(err)

def ensure_directory(path):
    """Ensure a directory exists"""
    os.makedirs(path, exist_ok=True)

def file_exists_and_not_empty(path):
    """Check if file exists and is not empty"""
    return os.path.exists(path) and os.path.getsize(path) > 0

def directory_exists_and_not_empty(path):
    """Check if directory exists and has files"""
    return os.path.exists(path) and len(os.listdir(path)) > 0


_TRIPLE_RE = re.compile(r'^target\s+triple\s*=\s*"([^"]*)"', re.M)
_LAYOUT_RE = re.compile(r'^target\s+datalayout\s*=\s*"([^"]*)"', re.M)


def read_ir_target(path: str) -> tuple[str | None, str | None]:
    """
    读取 .ll/.bc 模块头部的 target triple 和 datalayout。
    .bc 用 llvm-dis 流式输出，读到 triple 后立即结束，不反汇编整个模块。
    """
    header = ""
    if path.endswith(".bc"):
        proc = subprocess.Popen(["llvm-dis", "-o", "-", path], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        try:
            for _ in range(200):
                line = proc.stdout.readline()
                if not line:
                    break
                header += line
                if line.startswith("target triple"):
                    break
        finally:
            proc.kill()
            proc.wait()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            header = handle.read(65536)
    triple = _TRIPLE_RE.search(header)
    layout = _LAYOUT_RE.search(header)
    return (triple.group(1) if triple else None, layout.group(1) if layout else None)


def clang_target_args(path: str, arch_flag: str | None, allow_host_retarget: bool) -> tuple[list[str], dict]:
    """
    clang 编译 IR 时的目标参数。默认使用模块自己的 triple（不改写源目标的 data layout）；
    只有显式 allow_host_retarget 时才沿用旧的 -m32/-m64（会把模块改写成宿主目标），并记录下来。
    """
    triple, layout = read_ir_target(path)
    if allow_host_retarget:
        if arch_flag is None:
            raise ValueError(f"{path}: host retarget requested but 32/64-bit mode is unknown")
        return [arch_flag], {"target_args": arch_flag, "retargeted_to_host": True, "module_triple": triple}
    if not triple:
        raise ValueError(f"{path}: module has no target triple; refusing to let clang pick the host target")
    return [f"--target={triple}"], {"target_args": f"--target={triple}", "retargeted_to_host": False,
                                    "module_triple": triple, "module_datalayout": layout}
