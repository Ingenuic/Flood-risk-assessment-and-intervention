"""Create a three-panel comparison of four hazard-intervention strategies.

The script reads existing experiment outputs only; it does not rerun the
intervention experiment or modify any input file.

Example
-------
python plot_four_strategy_figure.py --input-dir data --output-dir outputs

Required input files
--------------------
- four_strategy_scenario_results.csv
- selected_grids_and_scores.csv
- selected_grid_counts.csv

Dependencies
------------
Python 3.10+; NumPy, pandas, Matplotlib, and Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


STRATEGIES = [
    "route_informed",
    "grid_directed",
    "static_risk_directed",
    "hazard_directed",
]
STRATEGY_LABELS = {
    "route_informed": "Route-informed",
    "grid_directed": "Integrated-risk grid",
    "static_risk_directed": "Static-risk grid",
    "hazard_directed": "Hazard-directed",
}
STRATEGY_COLORS = {
    "route_informed": "#AC5148",
    "grid_directed": "#607F98",
    "static_risk_directed": "#348D80",
    "hazard_directed": "#C69A35",
}
BUDGETS = ["top1", "top5", "top10"]
BUDGET_LABELS = ["Top 1%", "Top 5%", "Top 10%"]
SOURCE_FILES = [
    "four_strategy_scenario_results.csv",
    "selected_grids_and_scores.csv",
    "selected_grid_counts.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot four hazard-intervention strategies from CSV outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the three required CSV files (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for figures, source tables, caption, and QA files.",
    )
    parser.add_argument(
        "--stem",
        default="four_strategy_intervention",
        help="Base name for exported figure files.",
    )
    parser.add_argument(
        "--dpi", type=int, default=600, help="Resolution for PNG and TIFF exports."
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: input_dir / name for name in SOURCE_FILES}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path} ({name})")

    hashes_before = {name: sha256(path) for name, path in paths.items()}
    scenarios = pd.read_csv(paths[SOURCE_FILES[0]])
    selected = pd.read_csv(paths[SOURCE_FILES[1]])
    counts = pd.read_csv(paths[SOURCE_FILES[2]])

    require_columns(
        scenarios,
        {
            "strategy",
            "selection",
            "hazard_reduction",
            "selected_grid_count",
            "baseline_system_pdr",
            "selected_grid_pdr_basis",
            "system_pdr_reduction",
            "system_pdr_reduction_pct",
            "new_system_pdr",
        },
        SOURCE_FILES[0],
    )
    require_columns(
        selected,
        {"strategy", "selection", "grid_id", "selected_grid_count"},
        SOURCE_FILES[1],
    )
    require_columns(counts, {"selection", "selected_grid_count"}, SOURCE_FILES[2])

    budget_counts = np.array(
        [
            int(
                counts.loc[counts["selection"].eq(budget), "selected_grid_count"].iloc[0]
            )
            for budget in BUDGETS
        ]
    )

    checks: dict[str, bool] = {}

    def check(name: str, value: bool) -> None:
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)

    check(
        "36 unique strategy-budget-hazard scenarios",
        len(scenarios) == 36
        and not scenarios.duplicated(
            ["strategy", "selection", "hazard_reduction"]
        ).any(),
    )
    check("exactly four expected strategies", set(scenarios["strategy"]) == set(STRATEGIES))
    check(
        "exactly 10%, 20%, and 30% hazard reductions",
        set(scenarios["hazard_reduction"]) == {0.1, 0.2, 0.3},
    )

    for budget, grid_count in zip(BUDGETS, budget_counts):
        check(
            f"{budget}: equal grid count K={grid_count}",
            (
                scenarios.loc[
                    scenarios["selection"].eq(budget), "selected_grid_count"
                ]
                == grid_count
            ).all(),
        )
        for strategy in STRATEGIES:
            subset = selected[
                selected["strategy"].eq(strategy)
                & selected["selection"].eq(budget)
            ]
            check(
                f"{strategy}/{budget}: unique grid IDs and expected count",
                len(subset) == grid_count
                and subset["grid_id"].nunique() == grid_count
                and (subset["selected_grid_count"] == grid_count).all(),
            )

    check(
        "PDR percentage matches raw reduction and baseline",
        np.allclose(
            scenarios["system_pdr_reduction_pct"],
            100
            * scenarios["system_pdr_reduction"]
            / scenarios["baseline_system_pdr"],
            rtol=1e-12,
        ),
    )
    check(
        "new PDR matches baseline minus reduction",
        np.allclose(
            scenarios["new_system_pdr"],
            scenarios["baseline_system_pdr"]
            - scenarios["system_pdr_reduction"],
            rtol=1e-12,
        ),
    )
    check(
        "reduction matches hazard fraction times selected basis",
        np.allclose(
            scenarios["system_pdr_reduction"],
            scenarios["hazard_reduction"]
            * scenarios["selected_grid_pdr_basis"],
            rtol=1e-12,
        ),
    )

    for strategy in STRATEGIES:
        for budget in BUDGETS:
            subset = scenarios[
                scenarios["strategy"].eq(strategy)
                & scenarios["selection"].eq(budget)
            ]
            check(
                f"{strategy}/{budget}: linear hazard sensitivity",
                np.ptp(
                    subset["system_pdr_reduction_pct"]
                    / subset["hazard_reduction"]
                )
                < 1e-10,
            )

    panel_a = scenarios.copy()
    panel_a["strategy_label"] = panel_a["strategy"].map(STRATEGY_LABELS)
    panel_a["plot_color"] = panel_a["strategy"].map(STRATEGY_COLORS)
    panel_a["response_unit"] = "percent of baseline system PDR"
    panel_a.to_csv(table_dir / "panel_a_source.csv", index=False)

    panel_b = panel_a[np.isclose(panel_a["hazard_reduction"], 0.2)].copy()
    panel_b["pdr_decrease_pp_per_100_grids"] = (
        panel_b["system_pdr_reduction_pct"]
        * 100
        / panel_b["selected_grid_count"]
    )
    route_reduction = (
        panel_b[panel_b["strategy"].eq(STRATEGIES[0])]
        .set_index("selection")["system_pdr_reduction_pct"]
    )
    panel_b["route_to_strategy_multiplier"] = [
        route_reduction.loc[row.selection] / row.system_pdr_reduction_pct
        for row in panel_b.itertuples()
    ]
    panel_b.to_csv(table_dir / "panel_b_source.csv", index=False)

    overlap_rows = []
    for strategy in STRATEGIES[1:]:
        for budget, grid_count in zip(BUDGETS, budget_counts):
            route_set = set(
                selected.loc[
                    selected["strategy"].eq(STRATEGIES[0])
                    & selected["selection"].eq(budget),
                    "grid_id",
                ]
            )
            comparison_set = set(
                selected.loc[
                    selected["strategy"].eq(strategy)
                    & selected["selection"].eq(budget),
                    "grid_id",
                ]
            )
            shared = len(route_set & comparison_set)
            union = len(route_set | comparison_set)
            overlap_rows.append(
                {
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "reference_strategy": STRATEGIES[0],
                    "selection": budget,
                    "selected_grid_count": int(grid_count),
                    "shared_grid_count": shared,
                    "union_grid_count": union,
                    "shared_grid_percent": 100 * shared / grid_count,
                    "jaccard": shared / union,
                }
            )
    panel_c = pd.DataFrame(overlap_rows)
    check(
        "equal-K overlap/Jaccard identity",
        np.allclose(
            panel_c["jaccard"],
            panel_c["shared_grid_count"]
            / (2 * panel_c["selected_grid_count"] - panel_c["shared_grid_count"]),
        ),
    )
    check(
        "nine overlap cells in valid range",
        len(panel_c) == 9
        and panel_c["shared_grid_percent"].between(0, 100).all()
        and panel_c["jaccard"].between(0, 1).all(),
    )
    panel_c.to_csv(table_dir / "panel_c_source.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.titlesize": 8,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(188 / 25.4, 108 / 25.4))
    ax = fig.add_axes([0.072, 0.265, 0.285, 0.47])
    bx = fig.add_axes([0.445, 0.265, 0.205, 0.47])
    cx = fig.add_axes([0.77, 0.265, 0.205, 0.47])

    for axis in (ax, bx):
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.grid(True, color="#E4E6E8", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65)

    line_styles = {0.1: (0, (1, 2)), 0.2: "-", 0.3: (0, (4, 2))}
    for hazard_reduction in (0.1, 0.3, 0.2):
        for strategy in STRATEGIES:
            subset = (
                panel_a[
                    panel_a["strategy"].eq(strategy)
                    & np.isclose(panel_a["hazard_reduction"], hazard_reduction)
                ]
                .set_index("selection")
                .loc[BUDGETS]
            )
            ax.plot(
                range(3),
                subset["system_pdr_reduction_pct"],
                color=STRATEGY_COLORS[strategy],
                linestyle=line_styles[hazard_reduction],
                linewidth=1.65 if hazard_reduction == 0.2 else 0.85,
                alpha=1 if hazard_reduction == 0.2 else 0.75,
                marker="o" if hazard_reduction == 0.2 else None,
                markersize=3,
                markeredgewidth=0,
                zorder=4 if hazard_reduction == 0.2 else 2,
            )
    ax.set(
        xlim=(-0.12, 2.12),
        ylim=(0, 30),
        ylabel="System PDR reduction (%)",
        xlabel="Intervention budget",
    )
    ax.set_xticks(range(3), BUDGET_LABELS)
    ax.set_yticks(range(0, 31, 5))

    bar_width = 0.18
    for index, strategy in enumerate(STRATEGIES):
        subset = panel_b[panel_b["strategy"].eq(strategy)].set_index(
            "selection"
        ).loc[BUDGETS]
        bx.bar(
            np.arange(3) + (index - 1.5) * bar_width,
            subset["pdr_decrease_pp_per_100_grids"],
            bar_width,
            color=STRATEGY_COLORS[strategy],
            edgecolor="white",
            linewidth=0.25,
        )
    bx.set(
        xlim=(-0.55, 2.55),
        ylim=(0, 4),
        ylabel="System PDR decrease\n(pp per 100 grids)",
        xlabel="Intervention budget",
    )
    bx.set_xticks(range(3), BUDGET_LABELS)
    bx.set_yticks([0, 1, 2, 3, 4])

    overlap_matrix = (
        panel_c.pivot(
            index="strategy", columns="selection", values="shared_grid_percent"
        )
        .loc[STRATEGIES[1:], BUDGETS]
        .to_numpy()
    )
    jaccard_matrix = (
        panel_c.pivot(index="strategy", columns="selection", values="jaccard")
        .loc[STRATEGIES[1:], BUDGETS]
        .to_numpy()
    )
    color_map = LinearSegmentedColormap.from_list(
        "shared", ["#F6F4EE", "#DBB5AE", "#AC5148"]
    )
    image = cx.imshow(
        overlap_matrix,
        vmin=0,
        vmax=100,
        cmap=color_map,
        aspect="auto",
        interpolation="nearest",
    )
    cx.set_xticks(range(3), BUDGET_LABELS)
    cx.set_yticks(
        range(3),
        ["Integrated-risk\ngrid", "Static-risk\ngrid", "Hazard-\ndirected"],
    )
    for label, strategy in zip(cx.get_yticklabels(), STRATEGIES[1:]):
        label.set_color(STRATEGY_COLORS[strategy])
        label.set_fontsize(7)
    cx.tick_params(length=0, pad=4)
    cx.set_xlabel("Intervention budget", labelpad=6)
    for row in range(3):
        for column in range(3):
            cx.text(
                column,
                row,
                f"{overlap_matrix[row, column]:.1f}%\nJ={jaccard_matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                linespacing=1.55,
                color="white" if overlap_matrix[row, column] > 73 else "#161616",
            )
    cx.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    cx.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    cx.grid(which="minor", color="white", linewidth=1)
    cx.tick_params(which="minor", length=0)
    for spine in cx.spines.values():
        spine.set_visible(False)

    colorbar_axis = fig.add_axes([0.79, 0.125, 0.165, 0.025])
    colorbar = fig.colorbar(
        image, cax=colorbar_axis, orientation="horizontal", ticks=[0, 50, 100]
    )
    colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label("Shared grids / K (%)", fontsize=7, labelpad=2)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=STRATEGY_COLORS[strategy],
                lw=3,
                label=STRATEGY_LABELS[strategy],
            )
            for strategy in STRATEGIES
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.8,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#454545",
                linestyle=line_styles[value],
                lw=1.65 if value == 0.2 else 0.85,
                label=f"{int(value * 100)}% H reduction",
            )
            for value in (0.1, 0.2, 0.3)
        ],
        loc="upper left",
        bbox_to_anchor=(0.065, 0.89),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=2,
    )

    for x_position, panel, title in [
        (0.072, "a", "All intervention scenarios"),
        (0.445, "b", "20% H reduction"),
        (0.77, "c", "Overlap with route-informed"),
    ]:
        fig.text(
            x_position - 0.029,
            0.777,
            panel,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        fig.text(x_position, 0.782, title, fontsize=8, va="bottom")

    budget_text = ", ".join(
        f"{label}: K={count:,}" for label, count in zip(BUDGET_LABELS, budget_counts)
    )
    fig.text(
        0.072,
        0.065,
        f"Equal-grid-count comparison ({budget_text}).",
        fontsize=7.5,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.width and box.height and (
                box.x0 < -1
                or box.y0 < -1
                or box.x1 > canvas.width + 1
                or box.y1 > canvas.height + 1
            ):
                outside.append(artist.get_text())
    check("all visible text remains inside the figure canvas", not outside)
    check("36 line data points plotted", sum(len(line.get_xdata()) for line in ax.lines) == 36)
    check("12 grouped bars plotted", len(bx.patches) == 12)
    check("nine heatmap annotations plotted", len(cx.texts) == 9)

    for extension in ("png", "tiff", "svg", "pdf"):
        save_options: dict[str, object] = {"dpi": args.dpi}
        if extension == "tiff":
            save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(figure_dir / f"{args.stem}.{extension}", **save_options)
    plt.close(fig)

    with Image.open(figure_dir / f"{args.stem}.png") as image_file:
        pixels = image_file.size
    with Image.open(figure_dir / f"{args.stem}.tiff") as image_file:
        check(
            "TIFF is lossless RGB/RGBA and matches PNG dimensions",
            image_file.mode in {"RGB", "RGBA"} and image_file.size == pixels,
        )
    svg_text = (figure_dir / f"{args.stem}.svg").read_text(encoding="utf-8")
    check("SVG retains editable text", "<text " in svg_text)
    check("input files remain unchanged", hashes_before == {name: sha256(path) for name, path in paths.items()})

    caption = f"""# Figure caption

**Four intervention strategies under equal-grid-count budgets.** Route-informed, integrated-risk grid, static-risk grid, and hazard-directed selections are compared using the same number of grids at each budget ({budget_text}). **a,** Reduction in system PDR relative to the unmodified baseline for 10%, 20%, and 30% reductions in hazard within selected grids. Lines connect the three evaluated categorical budgets and do not imply intermediate evaluations. **b,** Efficiency under 20% hazard reduction, expressed as the reduction in system PDR (percentage points of baseline) per 100 selected grids. **c,** Overlap between each grid-selection strategy and the route-informed selection at the same budget. Cell colour and the first annotation show 100 × |S_comparator ∩ S_route| / K; the second annotation gives the Jaccard index. Outcomes are deterministic scenario evaluations; therefore no uncertainty bars or inferential tests are shown. Only hazard is reduced; exposure, vulnerability, demand, and route occurrences are held fixed without renormalization.

# Reproduction

Run `python plot_four_strategy_figure.py --input-dir <data-directory> --output-dir outputs`. The script reads existing CSV outputs only and does not rerun the intervention experiment.
"""
    (figure_dir / "figure_caption.md").write_text(caption, encoding="utf-8")

    qa = {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "figure_width_mm": 188,
        "figure_height_mm": 108,
        "raster_pixels": pixels,
        "dpi": args.dpi,
        "input_sha256": hashes_before,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "panel_c": panel_c.to_dict(orient="records"),
    }
    (figure_dir / "figure_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(json.dumps({"all_checks_passed": all(checks.values()), "output_dir": str(output_dir), "pixels": pixels}, indent=2))


if __name__ == "__main__":
    main()
