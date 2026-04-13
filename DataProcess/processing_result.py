"""
Data classes for ASM processing results
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProcessingResult:
    """Data class to hold processing results."""

    file_path: str
    success: bool
    input_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the dataclass to the persisted dataset format."""
        return {
            "file_path": self.file_path,
            "input_ids": self.input_ids,
        }
