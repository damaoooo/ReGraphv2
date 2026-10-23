"""
Single file processing logic for LLVM IR files
"""
import subprocess
import os
import logging
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerFast

from GraphBuilder.ddg_graph_builder import DataDependencyGraphBuilder
from GraphBuilder.cfg_graph_builder import CFGGraphBuilder
from Tokenizer.normalizer import normalize_file
from .processing_result import ProcessingResult


class FileProcessor:
    """Handles processing of individual LLVM IR files"""
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerFast,
                 cleanup_temp_files: bool = True):
        self.tokenizer = tokenizer
        self.cleanup_temp_files = cleanup_temp_files
        self.logger = logging.getLogger(__name__)
        
            
    def _generate_ddg(self, input_file, purified_file, instrumented_file: str, dot_file: str,
                      normalized_ir: str, encoding) -> Union[None, List[Tuple[int, int, int, int]]]:
        """Generate DDG graph structure using purified IR"""
        try:
            builder = DataDependencyGraphBuilder(self.tokenizer)
            ddg_graph = builder.generate_ddg_matrix(dot_file, instrumented_file, purified_file, normalized_ir, encoding)
            return ddg_graph
        except Exception as e:
            self.logger.error(f"Error in DDG building for {input_file}: {e}")
            return None
            
    def _generate_cfg(self, input_file: str, purified_file: str, dot_file: str,
                      normalized_ir: str, encoding) -> Union[None, List[Tuple[int, int, int, int, float]]]:
        """Generate CFG graph structure using purified IR"""
        try:
            builder = CFGGraphBuilder(self.tokenizer, purified_file, dot_file, normalized_ir, encoding)
            cfg_graph = builder.build_cfg_edges()
            return cfg_graph
        except Exception as e:
            self.logger.error(f"Error in CFG building for {input_file}: {e}")
            return None

    def _encode(self, input_file: str, purified_file: str) -> Optional[Tuple[str, object]]:
        """Normalize and tokenize the purified IR once; tokens and both graphs index the same encoding."""
        try:
            normalized_ir = normalize_file(purified_file)
            # normalize_file returns "" when the file cannot be read; that is a failure, not an empty function
            if not normalized_ir.strip():
                self.logger.error(f"Empty normalized IR for {input_file} ({purified_file})")
                return None
            return normalized_ir, self.tokenizer(normalized_ir, return_offsets_mapping=True)
        except Exception as e:
            self.logger.error(f"Error in tokenization for {input_file}: {e}")
            return None
            
    def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files"""
        if not self.cleanup_temp_files:
            return
            
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up {temp_file}: {e}")

    def process_single_file(self, input_file, purified_file, instrumented_file, cfg_dot, ddg_dot) -> ProcessingResult:
        """Process a single LLVM IR file"""
        temp_files = []
        
        try:
            encoded = self._encode(input_file, purified_file)
            if encoded is None:
                return ProcessingResult(file_path=input_file, success=False)
            normalized_ir, encoding = encoded
            input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
            
            # Generate DDG and CFG as token spans of the same encoding
            ddg_graph = self._generate_ddg(input_file, purified_file, instrumented_file, ddg_dot, normalized_ir, encoding)
            cfg_graph = self._generate_cfg(input_file, purified_file, cfg_dot, normalized_ir, encoding)
            
            # A sample needs its tokens and both graphs; an empty edge list is a valid graph, None is a failure
            success = ddg_graph is not None and cfg_graph is not None
            
            return ProcessingResult(
                file_path=input_file,
                success=success,
                ddg_graph=ddg_graph,
                cfg_graph=cfg_graph,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        except Exception as e:
            return ProcessingResult(
                file_path=input_file,
                success=False,
                error_message=str(e)
            )
        finally:
            self._cleanup_temp_files(temp_files)
