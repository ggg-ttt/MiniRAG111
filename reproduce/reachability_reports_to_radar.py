import argparse
import json
import re
from pathlib import Path


SERIES_STYLE = {
    "mini": {
        "name": "MiniRAG",
        "color": "#e74c3c",
        "linestyle": "-",
        "fill": True,
        "fill_alpha": 0.08,
    },
    "low": {
        "name": "LightRAG",
        "color": "#60a5fa",
        "linestyle": "-",
        "fill": True,
        "fill_alpha": 0.08,
    },
    "hybrid": {
        "name": "HybridRAG",
        "color": "#57c76f",
        "linestyle": "-",
        "fill": True,
        "fill_alpha": 0.12,
    },
}


def parse_report_path(path_str):
    path = Path(path_str).expanduser().resolve()
    name = path.name.lower()
    match = re.match(r"(mini|low|hybrid)_k(\d+)_all_reachability_report\.json$", name)
    if not match:
        raise ValueError("Unsupported report filename: %s" % path.name)
    method = match.group(1)
    k = int(match.group(2))
    return path, method, k


def load_report(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_extra_metric(extra_args):
    extras = {}
    for raw in extra_args:
        try:
            category, method, value = raw.split(":", 2)
        except ValueError as exc:
            raise ValueError(
                "Invalid --extra-metric format: %s. Expected category:method:value" % raw
            ) from exc

        method = method.lower()
        if method not in SERIES_STYLE:
            raise ValueError("Unsupported method in --extra-metric: %s" % method)

        extras.setdefault(category, {})[method] = float(value)
    return extras


def build_output(report_paths, metric_key, extra_metrics):
    grouped = {}
    ks = set()

    for raw_path in report_paths:
        path, method, k = parse_report_path(raw_path)
        report = load_report(path)
        if metric_key not in report:
            raise ValueError("Metric `%s` not found in %s" % (metric_key, path))
        grouped.setdefault(method, {})[k] = float(report[metric_key])
        ks.add(k)

    ordered_ks = sorted(ks)
    categories = ["k=%s" % k for k in ordered_ks]
    categories.extend(extra_metrics.keys())

    series = []
    for method in ["mini", "low", "hybrid"]:
        if method not in grouped:
            continue
        values = []
        for k in ordered_ks:
            if k not in grouped[method]:
                raise ValueError("Missing %s report for k=%s" % (method, k))
            values.append(grouped[method][k])

        for category, metric_values in extra_metrics.items():
            if method not in metric_values:
                raise ValueError("Missing %s value for extra metric `%s`" % (method, category))
            values.append(metric_values[method])

        item = dict(SERIES_STYLE[method])
        item["values"] = values
        series.append(item)

    max_value = max(max(item["values"]) for item in series) if series else 1.0
    r_max = min(1.0, round(max(1.0, max_value + 0.05), 2))
    if r_max <= 0.5:
        ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]

    return {
        "categories": categories,
        "style": {
            "r_max": r_max,
            "r_ticks": ticks,
            "legend_loc": "lower right",
            "legend_bbox": [1.1, 0.10],
            "figure_size": [6.4, 5.8],
            "font_size": 12,
            "fill_alpha": 0.10,
            "line_width": 1.8,
        },
        "series": series,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert reachability report JSON files to radar-plot JSON."
    )
    parser.add_argument("reports", nargs="+", help="Reachability report JSON files.")
    parser.add_argument(
        "-m",
        "--metric",
        default="reachability_rate_over_answer_mappable",
        help="Metric key to plot. Defaults to reachability_rate_over_answer_mappable",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="reproduce/result/reachability_radar.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--extra-metric",
        action="append",
        default=[],
        help="Append an extra axis with format category:method:value, e.g. AEC:mini:0.939",
    )
    args = parser.parse_args()

    extra_metrics = parse_extra_metric(args.extra_metric)
    payload = build_output(args.reports, args.metric, extra_metrics)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Radar JSON saved to: %s" % output_path)


if __name__ == "__main__":
    main()
