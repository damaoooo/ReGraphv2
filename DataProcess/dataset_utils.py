"""
Utility functions for ASM dataset building
"""
import glob
import os
from typing import List


def find_asm_files(directory: str, pattern: str = "**/*.asm") -> List[str]:
    """Find all ASM files in a directory recursively."""
    return sorted(glob.glob(os.path.join(directory, pattern), recursive=True))
