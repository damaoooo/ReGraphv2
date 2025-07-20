import pygraphviz as pgv
import networkx as nx
import re
from Tokenizer.normalizer import normalize_file, normalize_string
import os
import sys
from collections import defaultdict
from transformers import PreTrainedTokenizerFast
from Tokenizer.ir_tokenizer import load_tokenizer


class CFGGraphBuilder:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, llvm_ir_path, dot_file_path):
        """
        Initialize the CFG Graph Builder.
        
        Args:
            tokenizer: PreTrainedTokenizerFast instance for tokenization
            llvm_ir_path: Path to the LLVM IR file
            dot_file_path: Path to the dot graph file
        """
        self.llvm_ir_path = llvm_ir_path
        self.dot_file_path = dot_file_path
        
        self.tokenizer = tokenizer
        self.normalized_ir = normalize_file(llvm_ir_path)
        self.graph = self._read_dot_file()
        
        self.node_id_to_label = {}
        self.label_to_token_position = {}
        
        # Pre-compile regex to capture everything up to and including the first colon
        self._label_pattern = re.compile(r'^(.*?:)', re.DOTALL)
        self._probability_pattern = re.compile(r'(\d+(?:\.\d+)?)%')
        
        # Cache tokenization results
        self._tokenized_cache = None
        
    def _read_dot_file(self):
        """Read and return the dot file as a graph."""
        return pgv.AGraph(self.dot_file_path)
    
    def _extract_node_labels(self):
        """Extract labels from graph nodes with optimized regex."""
        for node in self.graph.nodes():
            node_id = node.get_name()
            node_label = node.attr['label'].strip()
            if not node_label:
                print(f"Node {node_id} has an empty label.")
                continue
            node_label = node_label.split(":")[0].replace("\\n", "") + ":"
            self.node_id_to_label[node_id] = node_label
    
    def _get_tokenized_data(self):
        """Cache tokenization results to avoid repeated computation."""
        if self._tokenized_cache is None:
            self._tokenized_cache = self.tokenizer(self.normalized_ir, return_offsets_mapping=True)
        return self._tokenized_cache
    
    def _map_labels_to_tokens(self):
        """Map node labels to token positions with optimized search."""
        tokenized = self._get_tokenized_data()
        offset_mapping = tokenized['offset_mapping']
        input_ids = tokenized['input_ids']
        
        assert len(input_ids) == len(offset_mapping), "Input IDs and offset mapping lengths do not match."
        
        # Create a sorted list of (start_pos, label, node_id) for efficient processing
        label_positions = []
        for node_id, label in self.node_id_to_label.items():
            start_idx = self.normalized_ir.find(label)
            if start_idx != -1:
                label_positions.append((start_idx, start_idx + len(label), label, node_id))
            else:
                print(f"Label '{label}' not found in normalized IR.")
        
        # Sort by start position for efficient processing
        label_positions.sort()
        
        # Use binary search approach to map tokens
        token_idx = 0
        for start_idx, end_idx, label, node_id in label_positions:
            tokens = []
            
            # Start from current token position to avoid redundant searches
            while token_idx < len(offset_mapping) and offset_mapping[token_idx][1] <= start_idx:
                token_idx += 1
            
            # Collect tokens within the label range
            current_idx = token_idx
            while current_idx < len(offset_mapping):
                start, end = offset_mapping[current_idx]
                if start >= start_idx and end <= end_idx:
                    tokens.append(current_idx)
                elif start >= end_idx:
                    break
                current_idx += 1
            
            self.label_to_token_position[label] = tokens
    
    def _extract_probability_from_edge(self, edge):
        """Extract probability from edge tooltip attribute with pre-compiled regex."""
        label = edge.attr.get('label', '')
        if label:
            probability = float(self._probability_pattern.search(label).group(1)) 
            return probability / 100.0
        else:
            return 1.0
    
    def build_cfg_edges(self):
        """
        Build CFG edges with token positions and probabilities.
        
        Returns:
            List of tuples: (source_tokens, target_tokens, probability)
        """
        self._extract_node_labels()
        self._map_labels_to_tokens()
        
        cfg_edges = []
        
        # Convert to set for O(1) lookup
        valid_labels = set(self.label_to_token_position.keys())
        
        for edge in self.graph.edges():
            source = edge[0]
            target = edge[1]
            
            source_label = self.node_id_to_label.get(source, "")
            target_label = self.node_id_to_label.get(target, "")
            
            # Early termination if labels are missing
            if source_label not in valid_labels or target_label not in valid_labels:
                print(f"Edge from {source} to {target} has missing labels.")
                continue
            
            # Only extract probability if we have valid labels
            probability = self._extract_probability_from_edge(edge)
            source_tokens = self.label_to_token_position[source_label]
            target_tokens = self.label_to_token_position[target_label]
            src_start = min(source_tokens)
            src_end = max(source_tokens)
            
            dst_start = min(target_tokens)
            dst_end = max(target_tokens)

            cfg_edges.append((src_start, src_end, dst_start, dst_end, probability))

        return cfg_edges


# Example usage:
if __name__ == "__main__":
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/output_tokenizer/llvm_ir_bpe.json"
    llvm_ir_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/ddg_exporter/test.ll"
    dot_file_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/cfg_exporter/build/cfg_e3534b9842b57a4818957c6c557cdac394ae11db4923d464e33a551568f9860b.dot"
    tokenizer = load_tokenizer(tokenizer_path)
    builder = CFGGraphBuilder(tokenizer, llvm_ir_path, dot_file_path)
    cfg_edges = builder.build_cfg_edges()
    
    for edge in cfg_edges:
        print(f"Source tokens: {edge[0], edge[1]}, Target tokens: {edge[2], edge[3]}, Probability: {edge[4]}")
