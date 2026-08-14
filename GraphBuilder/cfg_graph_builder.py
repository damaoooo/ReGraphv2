import re
import pygraphviz as pgv
from transformers import PreTrainedTokenizerFast

from Tokenizer.ir_tokenizer import load_tokenizer
from Tokenizer.normalizer import normalize_file


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
        self.label_to_char_range = {}
        self.label_to_token_range = {}
        
        self._dot_label_pattern = re.compile(r'^\s*(?P<label>(?:"[^"]+"|[^:]+):)')
        self._ir_block_pattern = re.compile(r'^\s*(?P<label>(?:"[^"]+"|[^:]+):)\s*$')
        self._function_end_pattern = re.compile(r'(?m)^}')
        self._probability_pattern = re.compile(r'(\d+(?:\.\d+)?)%')
        
        # Cache tokenization results
        self._tokenized_cache = None
        
    def _read_dot_file(self):
        """Read and return the dot file as a graph."""
        return pgv.AGraph(self.dot_file_path)
    
    def _extract_node_labels(self):
        """Extract basic block labels from CFG nodes."""
        for node in self.graph.nodes():
            node_id = node.get_name()
            raw_label = node.attr.get('label', '')
            if not raw_label:
                print(f"Node {node_id} has an empty label.")
                continue

            node_text = raw_label.replace("\\n", "\n").replace('\\"', '"').lstrip()
            first_line = node_text.splitlines()[0].strip() if node_text else ""
            label_match = self._dot_label_pattern.match(first_line)
            if label_match is None:
                print(f"Unable to extract a basic block label from node {node_id}.")
                continue

            self.node_id_to_label[node_id] = label_match.group('label').strip()
    
    def _get_tokenized_data(self):
        """Cache tokenization results to avoid repeated computation."""
        if self._tokenized_cache is None:
            self._tokenized_cache = self.tokenizer(self.normalized_ir, return_offsets_mapping=True)
        return self._tokenized_cache
    
    def _find_function_end(self, search_start: int) -> int:
        """Find the closing brace for the current function body."""
        function_tail = self.normalized_ir[search_start:]
        function_end_match = self._function_end_pattern.search(function_tail)
        if function_end_match is not None:
            return search_start + function_end_match.start()
        return len(self.normalized_ir)

    def _build_block_char_ranges(self):
        """Build character spans for each basic block in normalized IR."""
        self.label_to_char_range.clear()
        block_positions = []
        char_offset = 0

        for line in self.normalized_ir.splitlines(keepends=True):
            label_match = self._ir_block_pattern.match(line.strip())
            if label_match is not None:
                block_positions.append((label_match.group('label').strip(), char_offset))
            char_offset += len(line)

        if not block_positions:
            print("No basic block labels found in normalized IR.")
            return

        function_end = self._find_function_end(block_positions[-1][1])
        for index, (label, start_idx) in enumerate(block_positions):
            end_idx = block_positions[index + 1][1] if index + 1 < len(block_positions) else function_end
            self.label_to_char_range[label] = (start_idx, end_idx)

    def _char_range_to_token_range(self, offset_mapping, start_idx: int, end_idx: int):
        """Convert a character span to its token span."""
        token_start = None
        token_end = None

        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue

            if token_start is None and end > start_idx:
                token_start = token_idx

            if start < end_idx:
                token_end = token_idx
            elif start >= end_idx and token_end is not None:
                break

        if token_start is None or token_end is None:
            return None

        return token_start, token_end

    def _map_labels_to_tokens(self):
        """Map CFG node labels to full basic block token spans."""
        self.label_to_token_range.clear()
        tokenized = self._get_tokenized_data()
        offset_mapping = tokenized['offset_mapping']
        input_ids = tokenized['input_ids']
        
        assert len(input_ids) == len(offset_mapping), "Input IDs and offset mapping lengths do not match."

        self._build_block_char_ranges()

        for label in set(self.node_id_to_label.values()):
            char_range = self.label_to_char_range.get(label)
            if char_range is None:
                print(f"Basic block label '{label}' not found in normalized IR.")
                continue

            token_range = self._char_range_to_token_range(offset_mapping, char_range[0], char_range[1])
            if token_range is None:
                print(f"Unable to map basic block label '{label}' to tokens.")
                continue

            self.label_to_token_range[label] = token_range
    
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
        self.node_id_to_label.clear()
        self._extract_node_labels()
        self._map_labels_to_tokens()
        
        cfg_edges = []
        
        # Convert to set for O(1) lookup
        valid_labels = set(self.label_to_token_range.keys())
        
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
            src_start, src_end = self.label_to_token_range[source_label]
            dst_start, dst_end = self.label_to_token_range[target_label]

            cfg_edges.append((src_start, src_end, dst_start, dst_end, probability))

        return cfg_edges


# Example usage:
if __name__ == "__main__":
    tokenizer_path = "/path/to/rell/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
    llvm_ir_path = "/path/to/LinkerLocalityTest/code_sample/code_sample/sample_code_functions/792f695c859f77c25bb5261bff3310baf4781796.ll"
    dot_file_path = "/path/to/LinkerLocalityTest/code_sample/code_sample/sample_code_functions/cfg_a4047ae365ca99672328c618e5b052b8343d31879b0a1090eda19891175de006.dot"
    tokenizer = load_tokenizer(tokenizer_path)
    builder = CFGGraphBuilder(tokenizer, llvm_ir_path, dot_file_path)
    cfg_edges = builder.build_cfg_edges()
    
    for edge in cfg_edges:
        print(f"Source tokens: {edge[0], edge[1]}, Target tokens: {edge[2], edge[3]}, Probability: {edge[4]}")
