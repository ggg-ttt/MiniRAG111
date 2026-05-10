import argparse
import csv
import json
from pathlib import Path


try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib and numpy are required. "
        "Install them with `pip install matplotlib numpy`."
    ) from exc


DEFAULT_STYLE = {
    "r_max": 0.5,
    "r_ticks": [0.1, 0.2, 0.3, 0.4],
    "fill_alpha": 0.08,
    "line_width": 1.6,
    "grid_color": "#b7b7b7",
    "grid_alpha": 0.75,
    "spine_color": "#222222",
    "spine_width": 0.8,
    "font_size": 12,
    "label_size": 16,
    "tick_label_size": 10,
    "tick_padding": 2,
    "tick_format": "decimal",
    "legend_loc": "lower right",
    "legend_bbox": [1.10, 0.17],
    "legend_font_size": 10,
    "legend_frame_alpha": 0.92,
    "legend_edge_color": "#d8d8d8",
    "figure_size": [6.8, 6.0],
    "background_color": "white",
    "dpi": 300,
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty.")

    if "series" not in reader.fieldnames:
        raise ValueError("CSV must contain a `series` column.")

    categories = [name for name in reader.fieldnames if name != "series"]
    if not categories:
        raise ValueError("CSV must contain at least one category column.")

    series = []
    for row in rows:
        values = [float(row[category]) for category in categories]
        series.append(
            {
                "name": row["series"],
                "values": values,
            }
        )

    return {"categories": categories, "series": series}


def load_data(path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = load_json(path)
    elif suffix == ".csv":
        data = load_csv(path)
    else:
        raise ValueError("Only .json and .csv inputs are supported.")

    categories = data.get("categories", [])
    series = data.get("series", [])
    style = data.get("style", {})

    if not categories:
        raise ValueError("Input data must contain `categories`.")
    if not series:
        raise ValueError("Input data must contain `series`.")

    expected = len(categories)
    for item in series:
        values = item.get("values", [])
        if len(values) != expected:
            raise ValueError(
                "Series `%s` has %s values, expected %s."
                % (item.get("name", "<unnamed>"), len(values), expected)
            )

    return categories, series, style


def close_loop(values):
    return list(values) + [values[0]]


def format_tick_labels(ticks, tick_format):
    labels = []
    for tick in ticks:
        if tick_format == "percent":
            labels.append("%d%%" % round(tick * 100))
        else:
            labels.append(("{:.2f}".format(tick)).rstrip("0").rstrip("."))
    return labels


def plot_radar(categories, series, style, output_path):
    merged_style = dict(DEFAULT_STYLE)
    merged_style.update(style or {})

    plt.rcParams["font.size"] = merged_style["font_size"]

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=tuple(merged_style["figure_size"]),
        subplot_kw={"polar": True},
        dpi=merged_style["dpi"],
    )
    fig.patch.set_facecolor(merged_style["background_color"])
    ax.set_facecolor(merged_style["background_color"])

    ax.set_theta_offset(0.0)
    ax.set_theta_direction(1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=merged_style["label_size"])

    ax.set_ylim(0, merged_style["r_max"])
    ax.set_yticks(merged_style["r_ticks"])
    ax.set_yticklabels(
        format_tick_labels(merged_style["r_ticks"], merged_style["tick_format"]),
        fontsize=merged_style["tick_label_size"],
    )
    ax.tick_params(axis="y", pad=merged_style["tick_padding"])
    ax.yaxis.grid(True, color=merged_style["grid_color"], alpha=merged_style["grid_alpha"])
    ax.xaxis.grid(True, color=merged_style["grid_color"], alpha=merged_style["grid_alpha"])
    ax.spines["polar"].set_color(merged_style["spine_color"])
    ax.spines["polar"].set_linewidth(merged_style["spine_width"])

    for item in series:
        name = item["name"]
        values = close_loop(item["values"])
        color = item.get("color", "#4c78a8")
        linestyle = item.get("linestyle", "-")
        linewidth = item.get("linewidth", merged_style["line_width"])
        alpha = item.get("alpha", 1.0)
        fill = item.get("fill", True)
        fill_alpha = item.get("fill_alpha", merged_style["fill_alpha"])

        ax.plot(
            angles,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=name,
        )
        if fill:
            ax.fill(angles, values, color=color, alpha=fill_alpha)

    legend_bbox = merged_style.get("legend_bbox")
    if legend_bbox:
        bbox_to_anchor = tuple(legend_bbox)
    else:
        bbox_to_anchor = None

    ax.legend(
        loc=merged_style["legend_loc"],
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        facecolor="white",
        edgecolor=merged_style["legend_edge_color"],
        fontsize=merged_style["legend_font_size"],
        framealpha=merged_style["legend_frame_alpha"],
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Draw a radar chart from JSON or CSV input."
    )
    parser.add_argument("input", help="Path to the input JSON or CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        default="reproduce/result/radar_chart.png",
        help="Output image path. Defaults to reproduce/result/radar_chart.png",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    categories, series, style = load_data(input_path)
    plot_radar(categories, series, style, output_path)

    print("Radar chart saved to: %s" % output_path)


if __name__ == "__main__":
    main()
