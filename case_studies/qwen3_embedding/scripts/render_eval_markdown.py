#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def extract_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else None


def build_markdown(args, log_text: str) -> str:
    embedding_shape = extract_first(r"嵌入向量(?:生成|加载)完毕，形状为:\s*\(([^)]+)\)", log_text)
    anchor_count = extract_first(r"将评估所有\s+([0-9,]+)\s+个锚点", log_text)
    mrr10 = extract_first(r"MRR@10:\s*([0-9.]+)", log_text)
    mrr30 = extract_first(r"MRR@30:\s*([0-9.]+)", log_text)

    lines = [
        f"# {args.title}",
        "",
        f"- Generated from log: `{args.log}`",
        f"- Output markdown: `{args.output}`",
        f"- Endpoint: `{args.endpoint}`",
        f"- Model ID: `{args.model_id}`",
        f"- Cache file: `{args.cache}`",
        f"- Command: `{args.command}`",
    ]

    if embedding_shape:
        lines.append(f"- Embedding shape: `({embedding_shape})`")
    if anchor_count:
        lines.append(f"- Anchors evaluated: `{anchor_count}`")
    if mrr10:
        lines.append(f"- MRR@10: `{mrr10}`")
    if mrr30:
        lines.append(f"- MRR@30: `{mrr30}`")

    lines.extend(
        [
            "",
            "## Raw log",
            "",
            "```text",
            log_text.rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an evaluation log into a markdown report.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    output_path = Path(args.output)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(args, log_text), encoding="utf-8")


if __name__ == "__main__":
    main()
