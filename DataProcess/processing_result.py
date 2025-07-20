"""
Data classes for processing results
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ProcessingResult:
    """Data class to hold processing results"""
    file_path: str
    success: bool
    ddg_graph: List[Tuple[int, int, int, int]] = None
    cfg_graph: List[Tuple[int, int, int, int, float]] = None
    input_ids: List[int] = None
    attention_mask: List[int] = None
    error_message: Optional[str] = None

    # 定义 dict 的魔术方法
    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary"""
        return {
            "file_path": self.file_path,
            "ddg_graph": self.ddg_graph,
            "cfg_graph": self.cfg_graph,
            "input_ids": self.input_ids,
        }
