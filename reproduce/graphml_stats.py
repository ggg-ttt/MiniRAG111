import argparse
import csv
import html
import json
import os
import re
from collections import Counter, defaultdict
from typing import Iterable

import networkx as nx


GRAPH_FIELD_SEP = "<SEP>"


METRIC_LABELS = {
    "entity_count": "实体数",
    "relation_count": "关系数",
    "average_node_degree": "平均节点度",
    "largest_connected_component_size": "最大连通分量大小",
    "unique_entities_per_relation": "每种关系连接的唯一实体数",
    "unique_entities_per_relation_json": "每种关系连接的唯一实体数(JSON)",
    "avg_unique_entities_per_relation": "每种关系连接的唯一实体数平均值",
    "relation_diversity": "关系多样性",
    "duplicate_entity_rate": "重复实体率",
    "duplicate_relation_description_rate": "重复关系表述率",
}


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def normalize_entity_name(value: str) -> str:
    return normalize_text(value).upper()



def split_relation_labels(keywords: str) -> list[str]:
    raw = normalize_text(keywords)
    if not raw:
        return []

    labels = set()
    for phrase in raw.split(GRAPH_FIELD_SEP):
        phrase = normalize_text(phrase)
        if not phrase:
            continue
        head = normalize_text(phrase.split(",")[0])
        if head:
            labels.add(head)
    return sorted(labels)



def largest_component_size(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    undirected = graph.to_undirected()
    return max((len(component) for component in nx.connected_components(undirected)), default=0)



def duplicate_rate(values: Iterable[str], normalizer) -> float:
    normalized = [normalizer(value) for value in values if normalizer(value)]
    total = len(normalized)
    if total == 0:
        return 0.0
    counts = Counter(normalized)
    duplicate_items = sum(count for count in counts.values() if count > 1)
    return duplicate_items / total



def collect_stats(graph: nx.Graph) -> list[dict[str, object]]:
    entity_count = graph.number_of_nodes()
    relation_count = graph.number_of_edges()
    average_degree = (sum(dict(graph.degree()).values()) / entity_count) if entity_count else 0.0
    lcc_size = largest_component_size(graph)

    relation_to_entities: dict[str, set[str]] = defaultdict(set)
    all_relation_labels: list[str] = []
    edge_descriptions: list[str] = []

    for source, target, data in graph.edges(data=True):
        source_name = normalize_entity_name(source)
        target_name = normalize_entity_name(target)

        description = normalize_text(data.get("description", ""))
        if description:
            edge_descriptions.append(description)

        labels = split_relation_labels(data.get("keywords", ""))
        all_relation_labels.extend(labels)
        for label in labels:
            relation_to_entities[label].add(source_name)
            relation_to_entities[label].add(target_name)

    relation_diversity = len(set(all_relation_labels)) / relation_count if relation_count else 0.0
    duplicate_entity_rate = duplicate_rate(graph.nodes(), normalize_entity_name)
    duplicate_relation_description_rate = duplicate_rate(edge_descriptions, normalize_text)

    # 计算每种关系连接的唯一实体数的平均值
    unique_entity_counts = [len(entities) for entities in relation_to_entities.values()]
    avg_unique_entities_per_relation = sum(unique_entity_counts) / len(unique_entity_counts) if unique_entity_counts else 0.0

    rows: list[dict[str, object]] = [
        {"metric": "entity_count", "metric_zh": METRIC_LABELS["entity_count"], "value": entity_count},
        {"metric": "relation_count", "metric_zh": METRIC_LABELS["relation_count"], "value": relation_count},
        {"metric": "average_node_degree", "metric_zh": METRIC_LABELS["average_node_degree"], "value": round(average_degree, 6)},
        {"metric": "largest_connected_component_size", "metric_zh": METRIC_LABELS["largest_connected_component_size"], "value": lcc_size},
        {"metric": "relation_diversity", "metric_zh": METRIC_LABELS["relation_diversity"], "value": round(relation_diversity, 6)},
        {"metric": "duplicate_entity_rate", "metric_zh": METRIC_LABELS["duplicate_entity_rate"], "value": round(duplicate_entity_rate, 6)},
        {
            "metric": "duplicate_relation_description_rate",
            "metric_zh": METRIC_LABELS["duplicate_relation_description_rate"],
            "value": round(duplicate_relation_description_rate, 6),
        },
        {
            "metric": "avg_unique_entities_per_relation",
            "metric_zh": METRIC_LABELS["avg_unique_entities_per_relation"],
            "value": round(avg_unique_entities_per_relation, 6),
        },
    ]

    for relation_label in sorted(relation_to_entities):
        rows.append(
            {
                "metric": "unique_entities_per_relation",
                "metric_zh": METRIC_LABELS["unique_entities_per_relation"],
                "relation_type": relation_label,
                "value": len(relation_to_entities[relation_label]),
            }
        )

    rows.append(
        {
            "metric": "unique_entities_per_relation_json",
            "metric_zh": METRIC_LABELS["unique_entities_per_relation_json"],
            "value": json.dumps(
                {
                    relation_label: len(entities)
                    for relation_label, entities in sorted(relation_to_entities.items())
                },
                ensure_ascii=False,
            ),
        }
    )
    return rows



def write_csv(rows: list[dict[str, object]], output_file: str) -> None:
    fieldnames = ["metric", "metric_zh", "relation_type", "value"]
    with open(output_file, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row.get("metric", ""),
                    "metric_zh": row.get("metric_zh", ""),
                    "relation_type": row.get("relation_type", ""),
                    "value": row.get("value", ""),
                }
            )



def main() -> None:
    #python reproduce\graphml_stats.py <你的graphml文件> 
    parser = argparse.ArgumentParser(description="Collect GraphML graph statistics and write them to CSV.")
    parser.add_argument("graphml", help="Input GraphML file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file path. Defaults to <graphml>_stats.csv",
    )
    args = parser.parse_args()

    graphml_path = args.graphml
    output_path = args.output or f"{os.path.splitext(graphml_path)[0]}_stats.csv"

    graph = nx.read_graphml(graphml_path)
    rows = collect_stats(graph)
    write_csv(rows, output_path)
    print(f"saved stats to: {output_path}")


if __name__ == "__main__":
    main()
