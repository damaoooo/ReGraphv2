"""
Common utilities for the lift dataset pipeline
"""
import subprocess
import os
from rich.console import Console

console = Console()

def run_command(command, description=""):
    """Run a shell command and return success status, stdout, stderr"""
    try:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        success = p.returncode == 0
        return success, stdout.decode(), stderr.decode()
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
