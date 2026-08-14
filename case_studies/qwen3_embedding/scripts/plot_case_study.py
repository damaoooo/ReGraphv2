#!/usr/bin/env python3
"""Plot the Qwen3 case-study curves from frozen Markdown result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


SETTINGS = (
    (
        "0.6B ASM",
        "qwen3_0p6b_official_asm_common_oc_csv_hashdedup.md",
        "#4C566A",
        "-",
        "o",
    ),
    (
        "0.6B ASM + FT",
        "qwen3_0p6b_asm_ft_original_train_asm_prompt.md",
        "#7B8794",
        "--",
        "s",
    ),
    (
        "4B ASM",
        "qwen3_4b_official_asm_common_oc_csv_hashdedup.md",
        "#A3A3A3",
        ":",
        "^",
    ),
    (
        "0.6B IR",
        "qwen3_0p6b_official_ir_raw_csv_len128_text_hashdedup.md",
        "#2F6B5F",
        "-.",
        "D",
    ),
    (
        "0.6B IR + FT",
        "qwen3_0p6b_ft_oc_trained_local_tei_csv_len128_text_hashdedup.md",
        "#8B5A3C",
        "-",
        "v",
    ),
)


def table_column(path: Path, heading: str, value_column: str) -> tuple[list[int], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(heading)
    except ValueError as exc:
        raise ValueError(f"Missing {heading!r} in {path}") from exc

    header_index = next(
        index for index in range(start + 1, len(lines)) if lines[index].startswith("|")
    )
    columns = [item.strip() for item in lines[header_index].strip("|").split("|")]
    pool_index = columns.index("Pool Size")
    value_index = columns.index(value_column)

    pools: list[int] = []
    values: list[float] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [item.strip() for item in line.strip("|").split("|")]
        pools.append(int(cells[pool_index].replace(",", "")))
        values.append(float(cells[value_index]))
    if not pools:
        raise ValueError(f"No rows found under {heading!r} in {path}")
    return pools, values


def plot_metric(results_dir: Path, output: Path, heading: str, column: str) -> None:
    fig, axis = plt.subplots(figsize=(3.35, 2.45))
    for label, filename, color, linestyle, marker in SETTINGS:
        pools, values = table_column(results_dir / filename, heading, column)
        axis.plot(
            pools,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=[len(pools) - 1],
            linewidth=1.25,
            markersize=3.2,
        )

    axis.set_xscale("log", base=2)
    axis.set_xlabel("Candidate pool size")
    axis.set_ylabel(column)
    axis.grid(axis="y", color="0.88", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.5)
    axis.legend(frameon=False, fontsize=6.8, ncol=2, handlelength=2.2)
    fig.tight_layout(pad=0.35)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    case_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=case_dir / "results")
    parser.add_argument("--output-dir", type=Path, default=case_dir / "figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(args.results_dir, args.output_dir / "case_recall.pdf", "## Recall@K", "Recall@1")
    plot_metric(args.results_dir, args.output_dir / "case_mrr.pdf", "## MRR@P", "MRR@P")


if __name__ == "__main__":
    main()
