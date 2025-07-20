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
        
            
    def _generate_ddg(self, input_file, purified_file, instrumented_file: str, dot_file: str) -> Union[None, List[Tuple[int, int, int, int]]]:
        """Generate DDG graph structure using purified IR"""
        try:
            builder = DataDependencyGraphBuilder(self.tokenizer)
            ddg_graph = builder.generate_ddg_matrix(dot_file, instrumented_file, purified_file)
            return ddg_graph
        except Exception as e:
            self.logger.error(f"Error in DDG building for {input_file}: {e}")
            return None
            
    def _generate_cfg(self, input_file: str, purified_file: str, dot_file: str) -> Union[None, List[Tuple[int, int, int, int, float]]]:
        """Generate CFG graph structure using purified IR"""
        try:
            builder = CFGGraphBuilder(self.tokenizer, purified_file, dot_file)
            cfg_graph = builder.build_cfg_edges()
            return cfg_graph
        except Exception as e:
            self.logger.error(f"Error in CFG building for {input_file}: {e}")
            return None

    def _generate_token_ids(self, input_file: str, purified_file: str) -> Optional[Tuple[List[int], List[int]]]:
        """Generate token IDs and attention mask using purified IR"""
        try:
            normalized_ir = normalize_file(purified_file)
            tokens = self.tokenizer(normalized_ir)
            return tokens['input_ids'], tokens['attention_mask']
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
            # Generate DDG using purified IR
            ddg_graph = self._generate_ddg(input_file=input_file, purified_file=purified_file, instrumented_file=instrumented_file, dot_file=ddg_dot)
            
            # Generate CFG using purified IR
            cfg_graph = self._generate_cfg(input_file=input_file, purified_file=purified_file, dot_file=cfg_dot)
            
            # Generate tokens using purified IR
            token_result = self._generate_token_ids(input_file=input_file, purified_file=purified_file)
            input_ids, attention_mask = token_result if token_result else (None, None)
            
            success = ddg_graph is not None or cfg_graph is not None or input_ids is not None
            
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
