from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "Binaries" / "Dataset-1"
LIFT_PATH = PROJECT_ROOT / "Binaries" / "Dataset-1-re"
OUTPUT_SVG = PROJECT_ROOT / "paper" / "figures" / "binary_pipeline_sankey.svg"

CANVAS_WIDTH = 252
CANVAS_HEIGHT = 172
MARGIN_X = 16
NODE_WIDTH = 7
MAIN_NODE_TOP = 34
MAIN_TOTAL_HEIGHT = 86
LOSS_NODE_TOP = 134
CURVE_STRENGTH = 0.44

TEXT_PRIMARY = "#2F2A26"
TEXT_MUTED = "#514A44"
NODE_STROKE = "#867D75"
MAIN_NODE_OPACITY = "0.78"
LOSS_NODE_OPACITY = "0.76"
MAIN_LINK_OPACITY = "0.44"
LOSS_LINK_OPACITY = "0.50"

MAIN_NODE_COLORS = ["#A9B7A0", "#8FA8A2", "#93A3B2", "#9FB2A9"]
MAIN_LINK_COLORS = ["#BBC7B4", "#AABBB7", "#A8B7C3"]
LOSS_NODE_COLORS = ["#C8B2A7", "#C1AD97", "#B99B92"]
LOSS_LINK_COLORS = ["#DECBC2", "#D7CCBC", "#D2BDB7"]


def count_binary_files(root: Path) -> tuple[int, int]:
    total_cnt = 0
    i64_cnt = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".i64":
            i64_cnt += 1
            continue
        total_cnt += 1

    return total_cnt, i64_cnt


def count_pipeline_outputs(root: Path) -> tuple[int, int, int]:
    lift_count = 0
    reopt_count = 0
    recompile_count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".ll":
            lift_count += 1
        elif path.suffix == ".bc":
            reopt_count += 1
        elif path.suffix == ".re":
            recompile_count += 1

    return lift_count, reopt_count, recompile_count


def pct(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else 0.0


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_count(value: int) -> str:
    return f"{value:,}"


def link_path(
    x0: float,
    y0_top: float,
    y0_bottom: float,
    x1: float,
    y1_top: float,
    y1_bottom: float,
    curve: float = CURVE_STRENGTH,
) -> str:
    dx = x1 - x0
    c0 = x0 + dx * curve
    c1 = x1 - dx * curve
    return (
        f"M {x0:.2f} {y0_top:.2f} "
        f"C {c0:.2f} {y0_top:.2f} {c1:.2f} {y1_top:.2f} {x1:.2f} {y1_top:.2f} "
        f"L {x1:.2f} {y1_bottom:.2f} "
        f"C {c1:.2f} {y1_bottom:.2f} {c0:.2f} {y0_bottom:.2f} {x0:.2f} {y0_bottom:.2f} Z"
    )


def add_text_block(
    parent: Element,
    x: float,
    y: float,
    lines: list[str],
    anchor: str = "start",
    font_size: float = 26,
    line_gap: float = 1.28,
    fill: str = TEXT_PRIMARY,
    weight: str = "400",
) -> None:
    # Avoid <tspan dy="..."> because Inkscape's PDF+LaTeX export can
    # mis-handle tspan positioning/line spacing compared with browsers.
    for index, line in enumerate(lines):
        line_y = y + index * font_size * line_gap
        text = SubElement(
            parent,
            "text",
            {
                "x": f"{x:.2f}",
                "y": f"{line_y:.2f}",
                "text-anchor": anchor,
                "font-size": f"{font_size}",
                "font-family": "'Times New Roman', Times, serif",
                "fill": fill,
                "font-weight": weight,
            },
        )
        text.text = line


def build_svg(total_cnt: int, lift_count: int, reopt_count: int, recompile_count: int) -> Element:
    if total_cnt <= 0:
        raise ValueError("No binary files were found, cannot draw Sankey.")

    if lift_count > total_cnt or reopt_count > lift_count or recompile_count > reopt_count:
        raise ValueError("Pipeline counts must be monotonic for a standard Sankey layout.")

    loss_before_lift = total_cnt - lift_count
    loss_before_reopt = lift_count - reopt_count
    loss_before_recompile = reopt_count - recompile_count

    scale = MAIN_TOTAL_HEIGHT / total_cnt

    counts = [total_cnt, lift_count, reopt_count, recompile_count]
    losses = [loss_before_lift, loss_before_reopt, loss_before_recompile]
    x_positions = [MARGIN_X, 74, 132, 190]

    root = Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{CANVAS_WIDTH}pt",
            "height": f"{CANVAS_HEIGHT}pt",
            "viewBox": f"0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}",
        },
    )

    main_nodes: list[dict[str, float | str | list[str]]] = []
    for index, (x, count, color) in enumerate(zip(x_positions, counts, MAIN_NODE_COLORS)):
        height = count * scale
        main_nodes.append(
            {
                "x": x,
                "y": MAIN_NODE_TOP,
                "w": NODE_WIDTH,
                "h": height,
                "fill": color,
                "label_anchor": "start" if index < len(counts) - 1 else "end",
            }
        )

    loss_nodes: list[dict[str, float | str | list[str]]] = []
    for index, (x, count, color) in enumerate(zip(x_positions[1:], losses, LOSS_NODE_COLORS)):
        loss_nodes.append(
            {
                "x": x,
                "y": LOSS_NODE_TOP,
                "w": NODE_WIDTH,
                "h": count * scale,
                "fill": color,
                "label_anchor": "start" if index < len(losses) - 1 else "end",
            }
        )

    links_group = SubElement(root, "g", {"fill": "none"})

    for idx in range(3):
        source = main_nodes[idx]
        target = main_nodes[idx + 1]
        loss_target = loss_nodes[idx]
        passed_height = counts[idx + 1] * scale
        lost_height = losses[idx] * scale

        sx = float(source["x"]) + float(source["w"])
        tx = float(target["x"])
        lx = float(loss_target["x"])

        sy = float(source["y"])
        ty = float(target["y"])
        ly = float(loss_target["y"])

        pass_path = link_path(
            sx,
            sy,
            sy + passed_height,
            tx,
            ty,
            ty + passed_height,
        )
        SubElement(
            links_group,
            "path",
            {
                "d": pass_path,
                "fill": MAIN_LINK_COLORS[idx],
                "fill-opacity": MAIN_LINK_OPACITY,
                "stroke": MAIN_LINK_COLORS[idx],
                "stroke-opacity": MAIN_LINK_OPACITY,
                "stroke-width": "0.20",
            },
        )

        loss_path = link_path(
            sx,
            sy + passed_height,
            sy + passed_height + lost_height,
            lx,
            ly,
            ly + lost_height,
        )
        SubElement(
            links_group,
            "path",
            {
                "d": loss_path,
                "fill": LOSS_LINK_COLORS[idx],
                "fill-opacity": LOSS_LINK_OPACITY,
                "stroke": LOSS_LINK_COLORS[idx],
                "stroke-opacity": LOSS_LINK_OPACITY,
                "stroke-width": "0.26",
            },
        )

    nodes_group = SubElement(root, "g")

    for node in main_nodes:
        SubElement(
            nodes_group,
            "rect",
            {
                "x": f"{float(node['x']):.2f}",
                "y": f"{float(node['y']):.2f}",
                "width": f"{float(node['w']):.2f}",
                "height": f"{float(node['h']):.2f}",
                "fill": str(node["fill"]),
                "fill-opacity": MAIN_NODE_OPACITY,
                "stroke": NODE_STROKE,
                "stroke-opacity": "0.88",
                "stroke-width": "0.35",
            },
        )

    for node in loss_nodes:
        SubElement(
            nodes_group,
            "rect",
            {
                "x": f"{float(node['x']):.2f}",
                "y": f"{float(node['y']):.2f}",
                "width": f"{float(node['w']):.2f}",
                "height": f"{max(float(node['h']), 0.8):.2f}",
                "fill": str(node["fill"]),
                "fill-opacity": LOSS_NODE_OPACITY,
                "stroke": NODE_STROKE,
                "stroke-opacity": "0.86",
                "stroke-width": "0.32",
            },
        )

    main_labels = [
        ["Total", fmt_count(total_cnt)],
        ["Lifted", f"{fmt_count(lift_count)} ({fmt_pct(pct(lift_count, total_cnt))})"],
        ["Reopt.", f"{fmt_count(reopt_count)} ({fmt_pct(pct(reopt_count, lift_count))})"],
        ["Recompile", f"{fmt_count(recompile_count)} ({fmt_pct(pct(recompile_count, reopt_count))})"],
    ]

    for node, lines in zip(main_nodes, main_labels):
        label_x = float(node["x"]) + float(node["w"]) / 2
        label_y = 12
        add_text_block(
            root,
            label_x,
            label_y,
            lines,
            anchor="middle",
            font_size=8.4,
            line_gap=1.08,
            fill=TEXT_PRIMARY,
            weight="600",
        )

    loss_labels = [
        ["Lost@Lift", f"{fmt_count(loss_before_lift)} ({fmt_pct(pct(loss_before_lift, total_cnt))})"],
        ["Lost@Reopt", f"{fmt_count(loss_before_reopt)} ({fmt_pct(pct(loss_before_reopt, lift_count))})"],
        ["Lost@Recomp", f"{fmt_count(loss_before_recompile)} ({fmt_pct(pct(loss_before_recompile, reopt_count))})"],
    ]

    for node, lines in zip(loss_nodes, loss_labels):
        label_x = float(node["x"]) + float(node["w"]) / 2
        label_y = LOSS_NODE_TOP - 9
        add_text_block(
            root,
            label_x,
            label_y,
            lines,
            anchor="middle",
            font_size=7.2,
            line_gap=1.08,
            fill=TEXT_MUTED,
            weight="500",
        )

    return root


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Dataset directory not found: {SRC_PATH}")
    if not LIFT_PATH.exists():
        raise FileNotFoundError(f"Lifted output directory not found: {LIFT_PATH}")

    total_cnt, i64_cnt = count_binary_files(SRC_PATH)
    lift_count, reopt_count, recompile_count = count_pipeline_outputs(LIFT_PATH)

    print("Total binary files:", total_cnt)
    print("Total i64 files:", i64_cnt)
    print("Lifted files:", lift_count)
    print("Reoptimized files:", reopt_count)
    print("Recompiled files:", recompile_count)

    svg = build_svg(total_cnt, lift_count, reopt_count, recompile_count)
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    ElementTree(svg).write(OUTPUT_SVG, encoding="utf-8", xml_declaration=True)
    print(f"Sankey SVG saved to: {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
