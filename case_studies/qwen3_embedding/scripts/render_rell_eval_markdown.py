#!/usr/bin/env python3
"""Render ReLL evaluate.py rich-console output into compact markdown tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER_RE = re.compile(r"^[0-9][0-9,]*$")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def extract_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else None


def split_table_line(line: str) -> list[str]:
    if not any(char in line for char in "│┃║"):
        return []
    parts = re.split(r"[│┃║]", line)
    return [part.strip() for part in parts if part.strip()]


def parse_rich_tables(text: str) -> tuple[list[str], list[list[str]], list[list[str]]]:
    recall_header: list[str] = []
    recall_rows: list[list[str]] = []
    mrr_rows: list[list[str]] = []
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "Recall@K" in line:
            current = "recall"
            continue
        if "MRR@P" in line and "Pool" in line:
            current = "mrr"
            continue

        cells = split_table_line(line)
        if not cells or current is None:
            continue

        if current == "recall":
            if cells[0] == "Pool Size":
                recall_header = cells
            elif NUMBER_RE.match(cells[0]):
                recall_rows.append(cells)
        elif current == "mrr" and NUMBER_RE.match(cells[0]) and len(cells) >= 2:
            mrr_rows.append(cells[:2])

    return recall_header, recall_rows, mrr_rows


def markdown_table(header: list[str], rows: list[list[str]], right_align_first: bool = True) -> list[str]:
    if not header:
        return []
    align = ["---:" if right_align_first else "---"] + ["---:" for _ in header[1:]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return lines


def row_by_pool(rows: list[list[str]], pool_size: str) -> list[str] | None:
    normalized = pool_size.replace(",", "")
    for row in rows:
        if row and row[0].replace(",", "") == normalized:
            return row
    return None


def build_markdown(args: argparse.Namespace, log_text: str) -> str:
    clean_log = strip_ansi(log_text)
    embedding_shape = extract_first(r"嵌入向量(?:生成|加载)完毕，形状为:\s*\(([^)]+)\)", clean_log)
    anchor_count = extract_first(r"将评估所有\s+([0-9,]+)\s+个锚点", clean_log)
    sampled_count = extract_first(r"随机采样\s*([0-9,]+)\s*个进行评估", clean_log)
    mrr10 = extract_first(r"MRR@10:\s*([0-9.]+)", clean_log)
    mrr30 = extract_first(r"MRR@30:\s*([0-9.]+)", clean_log)
    recall_header, recall_rows, mrr_rows = parse_rich_tables(clean_log)

    pool10000_recall = row_by_pool(recall_rows, "10000")
    pool10000_mrr = row_by_pool(mrr_rows, "10000")
    recall1_10000 = None
    if pool10000_recall and recall_header and "Recall@1" in recall_header:
        recall1_10000 = pool10000_recall[recall_header.index("Recall@1")]
    mrrp_10000 = pool10000_mrr[1] if pool10000_mrr and len(pool10000_mrr) > 1 else None

    lines = [
        f"# {args.title}",
        "",
        f"- Endpoint: `{args.endpoint}`",
        f"- Model ID: `{args.model_id}`",
        f"- Dataset pool: `{args.dataset}`",
        f"- Positive map: `{args.positive_map}`",
        f"- Cache file: `{args.cache}`",
        f"- Log file: `{args.log}`",
        f"- Command: `{args.command}`",
    ]

    if embedding_shape:
        lines.append(f"- Embedding shape: `({embedding_shape})`")
    if anchor_count or sampled_count:
        lines.append(f"- Anchors evaluated: `{anchor_count or sampled_count}`")
    if mrr10:
        lines.append(f"- MRR@10: `{mrr10}`")
    if mrr30:
        lines.append(f"- MRR@30: `{mrr30}`")

    lines.extend(["", "## Summary", ""])
    summary_header = ["Setting", "Recall@1 @ Pool 10,000", "MRR@P @ Pool 10,000", "MRR@10", "MRR@30"]
    summary_row = [
        args.summary_name,
        recall1_10000 or "n/a",
        mrrp_10000 or "n/a",
        mrr10 or "n/a",
        mrr30 or "n/a",
    ]
    lines.extend(markdown_table(summary_header, [summary_row], right_align_first=False))

    if recall_header and recall_rows:
        lines.extend(["", "## Recall@K", ""])
        lines.extend(markdown_table(recall_header, recall_rows))

    if mrr_rows:
        lines.extend(["", "## MRR@P", ""])
        lines.extend(markdown_table(["Pool Size", "MRR@P"], mrr_rows))

    lines.extend(
        [
            "",
            "## Raw Log",
            "",
            "```text",
            clean_log.rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary-name", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--positive-map", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    output_path = Path(args.output)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(args, log_text), encoding="utf-8")
    print(f"Wrote markdown report: {output_path}")


if __name__ == "__main__":
    main()
