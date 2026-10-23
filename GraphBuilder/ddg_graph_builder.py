import pygraphviz as pgv
import networkx as nx
import re
import os
import sys
from Tokenizer.normalizer import normalize_file, normalize_string
from transformers import PreTrainedTokenizerFast
import bisect

from Tokenizer.ir_tokenizer import load_tokenizer


_LABEL_LINE = re.compile(r'^\s*(?:"[^"]*"|[-$.\w]+):\s*$')


def split_instructions(normalized_ir: str):
    """Return (start, end, text) character spans of the instructions in function bodies, in order.

    A `switch` / `indirectbr` that opens a `[` case list continues up to the line starting with `]`,
    so each span is one whole instruction even when it covers several lines.
    """
    spans = []
    offset = 0
    inside_function = False
    open_start, open_parts = None, []
    for line in normalized_ir.splitlines(keepends=True):
        body = line.rstrip('\r\n')
        stripped = body.strip()
        if open_start is not None:
            open_parts.append(stripped)
            if stripped.startswith(']'):
                spans.append((open_start, offset + len(body), ' '.join(open_parts)))
                open_start, open_parts = None, []
        elif body.startswith('define '):
            inside_function = True
        elif body.startswith('}'):
            inside_function = False
        elif inside_function and stripped and not _LABEL_LINE.match(body):
            if stripped.endswith('['):
                open_start, open_parts = offset + len(body) - len(body.lstrip()), [stripped]
            else:
                spans.append((offset + len(body) - len(body.lstrip()), offset + len(body), stripped))
        offset += len(line)
    if open_start is not None:
        raise ValueError("unterminated multi-line instruction in normalized IR")
    return spans


class DataDependencyGraphBuilder:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.normalized_instrumented_dict = {}
        self.id_range_map = {}
        self.node_id_inst_id_map = {}
        self.matrix_edges = []
        
        # Pre-compile regex patterns for better performance
        self.my_id_pattern = re.compile(r'!my_id\s+!(\d+)')
        self.my_id_remove_pattern = re.compile(r', !my_id\s+!(\d+)')
        self.id_pattern = re.compile(r'\(ID:\s*(\d+)\)')
        
        # Cache for tokenization results
        self._tokenization_cache = {}
    
    def _read_dot_file(self, dot_file: str) -> pgv.AGraph:
        g = pgv.AGraph(dot_file)
        return g
    
    def _build_instrumented_dict(self, instrumented_file: str):
        """Instructions of the instrumented IR in order, as (my_id or None, text without the my_id tag)."""
        self.instrumented_instructions = []
        for _, _, text in split_instructions(normalize_file(instrumented_file)):
            match = self.my_id_pattern.search(text)
            # Remove: ", !my_id !57"
            refined = self.my_id_remove_pattern.sub('', text)
            self.instrumented_instructions.append((match.group(1) if match else None, refined))
            if match:
                self.normalized_instrumented_dict[match.group(1)] = refined

    def _get_tokenization_result(self, text: str):
        """Cache tokenization results to avoid repeated computation"""
        if text not in self._tokenization_cache:
            self._tokenization_cache[text] = self.tokenizer(text, return_offsets_mapping=True)
        return self._tokenization_cache[text]

    def _char_span_to_token_range(self, start_index: int, end_index: int, offsets):
        search_start_idx = bisect.bisect_left(offsets, (start_index, 0))
        scan_start_idx = max(0, search_start_idx - 1)
        token_idx_start = None
        token_idx_end = None
        # More efficient token range finding
        for token_idx in range(scan_start_idx, len(offsets)):
                start, end = offsets[token_idx]

                # Early Break if we are past the end of the target range
                if start >= end_index:
                    break
                    
                # Filter out (0, 0) and judge the overlap
                if start < end and end > start_index:
                    if token_idx_start is None:
                        token_idx_start = token_idx
                    token_idx_end = token_idx

        if token_idx_start is None or token_idx_end is None:
            return None, None

        return token_idx_start, token_idx_end
    
    def _build_id_range_map(self, origin_ir: str):
        """Pair the k-th instrumented instruction with the k-th instruction of the origin IR.

        The DDG pass only attaches metadata, so both files list the same instructions in the same order;
        locating an instruction by its text would pick the first of several identical instructions.
        Any disagreement between the two listings fails the function instead of producing wrong edges.
        """
        original_normalized = normalize_file(origin_ir)
        if not original_normalized:
            raise ValueError(f"empty normalized IR: {origin_ir}")
        origin_instructions = split_instructions(original_normalized)
        if len(origin_instructions) != len(self.instrumented_instructions):
            raise ValueError(f"instrumented IR has {len(self.instrumented_instructions)} instructions, "
                             f"origin IR has {len(origin_instructions)}")

        # Tokenize once and reuse for all mappings
        offsets = self._get_tokenization_result(original_normalized)['offset_mapping']

        for (instruct_number, instrumented_text), (start, end, text) in zip(self.instrumented_instructions,
                                                                          origin_instructions):
            if instrumented_text != text:
                raise ValueError(f"instrumented IR does not match origin IR: {instrumented_text!r} != {text!r}")
            if instruct_number is None:
                continue
            start_idx, end_idx = self._char_span_to_token_range(start, end, offsets)
            if end_idx is not None:
                self.id_range_map[instruct_number] = (start_idx, end_idx)
    
    def _build_node_mapping(self, graph: pgv.AGraph):
        for node in graph.nodes():
            node_id = node.get_name()
            node_label = node.attr['label']
            instruction_number = None
            
            # Try extracting from !my_id !57 first (more efficient)
            if "!my_id" in node_label:
                match = self.my_id_pattern.search(node_label)
                if match:
                    instruction_number = match.group(1)
            
            # Try extracting from (ID: 57) if !my_id not found
            if instruction_number is None:
                match = self.id_pattern.search(node_label)
                if match:
                    instruction_number = match.group(1)
            
            if instruction_number is not None:
                self.node_id_inst_id_map[node_id] = instruction_number
    
    def _extract_edges(self, graph: pgv.AGraph):
        # Use set for automatic deduplication during construction
        edges_set = set()
        
        for edge in graph.edges():
            src, dst = edge
            src_inst_id = self.node_id_inst_id_map.get(src)
            dst_inst_id = self.node_id_inst_id_map.get(dst)

            if src_inst_id is not None and dst_inst_id is not None:
                edges_set.add((src_inst_id, dst_inst_id))

        # Build matrix edges more efficiently
        for src_inst_id, dst_inst_id in edges_set:
            src_range = self.id_range_map.get(src_inst_id)
            dst_range = self.id_range_map.get(dst_inst_id)
            
            if src_range is not None and dst_range is not None:
                src_start, src_end = src_range
                dst_start, dst_end = dst_range
                self.matrix_edges.append((src_start, src_end, dst_start, dst_end))
    
    def clear_cache(self):
        """Clear tokenization cache to free memory"""
        self._tokenization_cache.clear()
    
    def generate_ddg_matrix(self, dot_file: str, instrumented_file: str, origin_ir: str):
        # Fresh state per function; the returned list must not be cleared by a later call
        self.matrix_edges = []
        self.normalized_instrumented_dict = {}
        self.id_range_map = {}
        self.node_id_inst_id_map = {}
        self.clear_cache()
        
        graph = self._read_dot_file(dot_file)
        
        # Build all necessary mappings
        self._build_instrumented_dict(instrumented_file)
        self._build_id_range_map(origin_ir)
        self._build_node_mapping(graph)
        
        # Extract edges and build matrix
        self._extract_edges(graph)
        
        # Clear cache after processing to free memory
        self.clear_cache()
        
        return self.matrix_edges


def main():
    dot_file = "/home/damaoooo/Downloads/regraphv2/DataProcess/ddg_exporter/build/id_graph.movelim.dot"
    instrumented = "/home/damaoooo/Downloads/regraphv2/DataProcess/ddg_exporter/build/instrumented.ll"
    origin_ir = "/home/damaoooo/Downloads/regraphv2/DataProcess/ddg_exporter/test.ll"
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/output_tokenizer/llvm_ir_bpe.json"

    tokenizer = load_tokenizer(tokenizer_path)

    builder = DataDependencyGraphBuilder(tokenizer)
    matrix_edges = builder.generate_ddg_matrix(dot_file, instrumented, origin_ir)
    
    print(f"Generated {len(matrix_edges)} matrix edges")


if __name__ == "__main__":
    main()