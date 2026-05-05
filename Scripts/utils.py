"""
Common utilities for the lift dataset pipeline
"""
import subprocess
import os
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
            "opt level must look like O0/O1/O2/O3/Os/Og/Oz/Ofast or start with -O"
        )

    return normalized

def run_command(command, description="", timeout=3600, memory_limit_gb=10):
    """Run a shell command and return success status, stdout, stderr
    
    Args:
        command: Command to execute
        description: Description of the command
        timeout: Timeout in seconds (default: 3600)
        memory_limit_gb: Memory limit in GB (default: 10)
    """
    def set_limits():
        # Set memory limit (in bytes)
        memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    
    try:
        p = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=set_limits
        )
        stdout, stderr = p.communicate(timeout=timeout)
        success = p.returncode == 0
        return success, stdout.decode(), stderr.decode()
    except subprocess.TimeoutExpired:
        p.kill()
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)

def ensure_directory(path):
    """Ensure a directory exists"""
    os.makedirs(path, exist_ok=True)

def file_exists_and_not_empty(path):
    """Check if file exists and is not empty"""
    return os.path.exists(path) and os.path.getsize(path) > 0

def directory_exists_and_not_empty(path):
    """Check if directory exists and has files"""
    return os.path.exists(path) and len(os.listdir(path)) > 0
