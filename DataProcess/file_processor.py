"""
Single file processing logic for ASM files
"""
import logging
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerFast

from .processing_result import ProcessingResult


class FileProcessor:
    """Handles processing of individual ASM files."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        cleanup_temp_files: bool = True,
    ):
        self.tokenizer = tokenizer
        self.cleanup_temp_files = cleanup_temp_files
        self.logger = logging.getLogger(__name__)

    def _load_asm_text(self, input_file: str) -> Optional[str]:
        """Load raw ASM text from disk."""
        try:
            with open(input_file, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read().replace("\r\n", "\n")
        except Exception as exc:
            self.logger.error(f"Error reading ASM file {input_file}: {exc}")
            return None

    def _generate_token_ids(self, input_file: str) -> Optional[Tuple[List[int], List[int]]]:
        """Generate token ids directly from ASM text."""
        try:
            asm_text = self._load_asm_text(input_file)
            if asm_text is None:
                return None
            if not asm_text.strip():
                self.logger.error(f"ASM file is empty: {input_file}")
                return None

            tokens = self.tokenizer(asm_text)
            return tokens["input_ids"], tokens.get("attention_mask")
        except Exception as exc:
            self.logger.error(f"Error in tokenization for {input_file}: {exc}")
            return None

    def process_single_file(self, input_file: str) -> ProcessingResult:
        """Process a single ASM file."""
        try:
            token_result = self._generate_token_ids(input_file=input_file)
            input_ids, attention_mask = token_result if token_result else (None, None)

            return ProcessingResult(
                file_path=input_file,
                success=input_ids is not None,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        except Exception as exc:
            return ProcessingResult(
                file_path=input_file,
                success=False,
                error_message=str(exc),
            )
