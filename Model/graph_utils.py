from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _clip_span(start: Any, end: Any, seq_len: int) -> Optional[Tuple[int, int]]:
    if seq_len <= 0:
        return None
    start_idx = max(0, min(int(start), seq_len - 1))
    end_idx = max(start_idx, min(int(end), seq_len - 1))
    return start_idx, end_idx


def build_batched_span_graph_tensors(
    graphs: Sequence[Sequence[Sequence[Any]]],
    seq_lengths: Sequence[int],
    include_edge_attr: bool,
) -> Dict[str, Optional[torch.Tensor]]:
    node_spans: List[List[int]] = []
    node_batch: List[int] = []
    edge_pairs: List[List[int]] = []
    edge_attrs: List[List[float]] = []
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        seq_len = int(seq_lengths[graph_idx]) if graph_idx < len(seq_lengths) else 0
        span_to_local: Dict[Tuple[int, int], int] = {}
        local_edges: Dict[Tuple[int, int], float] = {}

        for edge in graph:
            if edge is None or len(edge) < 4:
                continue

            src_span = _clip_span(edge[0], edge[1], seq_len)
            dst_span = _clip_span(edge[2], edge[3], seq_len)
            if src_span is None or dst_span is None:
                continue

            if src_span not in span_to_local:
                span_to_local[src_span] = len(span_to_local)
            if dst_span not in span_to_local:
                span_to_local[dst_span] = len(span_to_local)

            src_idx = span_to_local[src_span]
            dst_idx = span_to_local[dst_span]
            attr_value = float(edge[4]) if include_edge_attr and len(edge) >= 5 else 1.0
            local_edges.setdefault((src_idx, dst_idx), attr_value)

        if not span_to_local:
            continue

        ordered_nodes = sorted(span_to_local.items(), key=lambda item: item[1])
        for span, _ in ordered_nodes:
            node_spans.append([span[0], span[1]])
            node_batch.append(graph_idx)

        for (src_idx, dst_idx), attr_value in local_edges.items():
            edge_pairs.append([node_offset + src_idx, node_offset + dst_idx])
            if include_edge_attr:
                edge_attrs.append([attr_value])

        node_offset += len(span_to_local)

    if node_spans:
        node_spans_tensor = torch.tensor(node_spans, dtype=torch.long)
        node_batch_tensor = torch.tensor(node_batch, dtype=torch.long)
    else:
        node_spans_tensor = torch.empty((0, 2), dtype=torch.long)
        node_batch_tensor = torch.empty((0,), dtype=torch.long)

    if edge_pairs:
        edge_index_tensor = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

    if include_edge_attr:
        if edge_attrs:
            edge_attr_tensor: Optional[torch.Tensor] = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_attr_tensor = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_attr_tensor = None

    return {
        "node_spans": node_spans_tensor,
        "node_batch": node_batch_tensor,
        "edge_index": edge_index_tensor,
        "edge_attr": edge_attr_tensor,
    }
