"""
LLVM IR Dataset Builder - Legacy compatibility module
This module provides backward compatibility for the refactored dataset builder.
New code should use the modular components directly.
"""

# Import the refactored components
from .dataset_features import get_dataset_features
from .processing_result import ProcessingResult
from .file_processor import FileProcessor
from .parallel_processor import ParallelProcessor, create_hf_dataset_from_files
from .dataset_utils import find_llvm_files, cleanup_residual_temp_files, FileFilter
from .dataset_builder_new import DatasetBuilder

from Utils.utils import DEFAULT_DDG_SO_PATH, DEFAULT_PURIFY_SO_PATH, DEFAULT_CFG_SO_PATH

# Import necessary types for compatibility
from typing import List
from transformers import PreTrainedTokenizerFast

# Re-export the CLI app for compatibility
import typer
from .cli import app

# Legacy functions for backward compatibility
def generate_ddg(tokenizer: PreTrainedTokenizerFast, llvm_ir_path: str):
    """Legacy function for generating DDG - use FileProcessor directly in new code"""
    processor = FileProcessor(
        tokenizer=tokenizer,
        ddg_so_path=DEFAULT_DDG_SO_PATH,
        purify_so_path=DEFAULT_PURIFY_SO_PATH,
        cfg_so_path=DEFAULT_CFG_SO_PATH
    )
    purified_file = processor._opt_initial_purify(llvm_ir_path)
    if purified_file is None:
        return None
    try:
        return processor._generate_ddg(llvm_ir_path, purified_file)
    finally:
        processor._cleanup_temp_files([purified_file])

def generate_cfg(tokenizer: PreTrainedTokenizerFast, llvm_ir_path: str):
    """Legacy function for generating CFG - use FileProcessor directly in new code"""
    processor = FileProcessor(
        tokenizer=tokenizer,
        ddg_so_path=DEFAULT_DDG_SO_PATH,
        purify_so_path=DEFAULT_PURIFY_SO_PATH,
        cfg_so_path=DEFAULT_CFG_SO_PATH
    )
    purified_file = processor._opt_initial_purify(llvm_ir_path)
    if purified_file is None:
        return None
    try:
        return processor._generate_cfg(llvm_ir_path, purified_file)
    finally:
        processor._cleanup_temp_files([purified_file])

def generate_token_ids(tokenizer: PreTrainedTokenizerFast, llvm_ir_path: str):
    """Legacy function for generating token IDs - use FileProcessor directly in new code"""
    processor = FileProcessor(
        tokenizer=tokenizer,
        ddg_so_path=DEFAULT_DDG_SO_PATH,
        purify_so_path=DEFAULT_PURIFY_SO_PATH,
        cfg_so_path=DEFAULT_CFG_SO_PATH
    )
    purified_file = processor._opt_initial_purify(llvm_ir_path)
    if purified_file is None:
        return None
    try:
        return processor._generate_token_ids(llvm_ir_path, purified_file)
    finally:
        processor._cleanup_temp_files([purified_file])


# Legacy CLI entry point
if __name__ == "__main__":
    app()