import argparse
import csv
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path
#用于图结构分析

GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
SEP_TOKEN = "<SEP>"


def clean_text(value):
    if value is None:
        return ""
    text = value.strip()
    while len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text.strip('"').strip()


def split_relation_tokens(raw_value):
    if not raw_value:
        return []

    tokens = []
    for part in raw_value.split(SEP_TOKEN):
        part = clean_text(part)
        if not part:
            continue

        for item in part.split(","):
            token = clean_text(item)
            if token:
                tokens.append(token)

    deduped = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def load_graphml(graphml_path, relation_field):
    tree = ET.parse(graphml_path)
    root = tree.getroot()

    key_map = {}
    for key in root.findall("g:key", GRAPHML_NS):
        key_id = key.attrib.get("id", "")
        key_for = key.attrib.get("for", "")
        attr_name = key.attrib.get("attr.name", "")
        key_map[key_id] = (key_for, attr_name)

    graph = root.find("g:graph", GRAPHML_NS)
    if graph is None:
        raise ValueError("GraphML graph element not found: %s" % graphml_path)

    nodes = {}
    edges = []

    for node in graph.findall("g:node", GRAPHML_NS):
        node_id = clean_text(node.attrib.get("id", ""))
        attrs = {}
        for data in node.findall("g:data", GRAPHML_NS):
            key_id = data.attrib.get("key", "")
            key_for, attr_name = key_map.get(key_id, ("", key_id))
            if key_for == "node":
                attrs[attr_name] = clean_text(data.text or "")
        nodes[node_id] = attrs

    for edge in graph.findall("g:edge", GRAPHML_NS):
        source = clean_text(edge.attrib.get("source", ""))
        target = clean_text(edge.attrib.get("target", ""))
        attrs = {}
        for data in edge.findall("g:data", GRAPHML_NS):
            key_id = data.attrib.get("key", "")
            key_for, attr_name = key_map.get(key_id, ("", key_id))
            if key_for == "edge":
                attrs[attr_name] = clean_text(data.text or "")

        relation_tokens = split_relation_tokens(attrs.get(relation_field, ""))
        if not relation_tokens:
            fallback = clean_text(attrs.get("description", ""))
            if fallback:
                relation_tokens = [fallback]

        edges.append(
            {
                "source": source,
                "target": target,
                "attrs": attrs,
                "relations": relation_tokens,
            }
        )

    return {"nodes": nodes, "edges": edges}


def weakly_connected_components(node_ids, undirected_neighbors):
    unvisited = set(node_ids)
    components = []

    while unvisited:
        start = next(iter(unvisited))
        component = set()
        queue = deque([start])
        unvisited.remove(start)

        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in undirected_neighbors.get(current, set()):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def compute_stats(graph_data, relation_field):
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    node_ids = set(nodes.keys())
    num_nodes = len(node_ids)
    num_edges = len(edges)

    in_degree = Counter()
    out_degree = Counter()
    undirected_neighbors = defaultdict(set)
    relation_to_nodes = defaultdict(set)
    pair_to_relations = defaultdict(set)
    relation_frequency = Counter()

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        out_degree[source] += 1
        in_degree[target] += 1

        if source != target:
            undirected_neighbors[source].add(target)
            undirected_neighbors[target].add(source)

        pair = tuple(sorted((source, target)))
        for relation in edge["relations"]:
            relation_to_nodes[relation].update((source, target))
            pair_to_relations[pair].add(relation)
            relation_frequency[relation] += 1

    total_degree_values = []
    for node_id in node_ids:
        total_degree_values.append(in_degree[node_id] + out_degree[node_id])

    avg_degree = statistics.fmean(total_degree_values) if total_degree_values else 0.0
    max_degree = max(total_degree_values, default=0)
    isolates = sum(1 for degree in total_degree_values if degree == 0)

    unique_relations = sorted(relation_to_nodes.keys())
    unique_relation_count = len(unique_relations)

    unique_entities_per_relation_values = [len(entity_ids) for entity_ids in relation_to_nodes.values()]
    if unique_entities_per_relation_values:
        unique_entities_per_relation_avg = statistics.fmean(unique_entities_per_relation_values)
    else:
        unique_entities_per_relation_avg = 0.0

    relation_diversity_values = [len(relations) for relations in pair_to_relations.values()]
    if relation_diversity_values:
        relation_diversity_avg = statistics.fmean(relation_diversity_values)
    else:
        relation_diversity_avg = 0.0

    components = weakly_connected_components(node_ids, undirected_neighbors) if node_ids else []
    component_sizes = sorted((len(component) for component in components), reverse=True)
    largest_component_size = component_sizes[0] if component_sizes else 0
    largest_component_ratio = (float(largest_component_size) / num_nodes) if num_nodes else 0.0

    possible_directed_edges = num_nodes * (num_nodes - 1)
    density = (float(num_edges) / possible_directed_edges) if possible_directed_edges else 0.0

    entity_types = set()
    nodes_with_type = 0
    for attrs in nodes.values():
        entity_type = clean_text(attrs.get("entity_type", ""))
        if entity_type:
            nodes_with_type += 1
            entity_types.add(entity_type)

    summary_rows = [
        {"metric": "graph_path", "value": ""},
        {"metric": "node_count", "value": num_nodes},
        {"metric": "edge_count", "value": num_edges},
        {"metric": "unique_relation_count", "value": unique_relation_count},
        {"metric": "avg_entity_degree", "value": round(avg_degree, 6)},
        {"metric": "max_entity_degree", "value": max_degree},
        {"metric": "isolated_node_count", "value": isolates},
        {"metric": "isolated_node_ratio", "value": round((float(isolates) / num_nodes) if num_nodes else 0.0, 6)},
        {"metric": "unique_entities_per_relation_avg", "value": round(unique_entities_per_relation_avg, 6)},
        {"metric": "relation_diversity_per_pair_avg", "value": round(relation_diversity_avg, 6)},
        {"metric": "weakly_connected_component_count", "value": len(components)},
        {"metric": "largest_weakly_connected_component_size", "value": largest_component_size},
        {"metric": "largest_weakly_connected_component_ratio", "value": round(largest_component_ratio, 6)},
        {"metric": "directed_density", "value": round(density, 10)},
        {"metric": "nodes_with_entity_type", "value": nodes_with_type},
        {"metric": "unique_entity_type_count", "value": len(entity_types)},
        {"metric": "relation_field_used", "value": relation_field},
    ]

    relation_rows = []
    for relation in unique_relations:
        relation_rows.append(
            {
                "relation": relation,
                "edge_mentions": relation_frequency[relation],
                "unique_entities": len(relation_to_nodes[relation]),
            }
        )

    return summary_rows, relation_rows


def write_csv(rows, output_path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_default_output_paths(graphml_path):
    stem = graphml_path.stem
    summary_path = graphml_path.with_name("%s_structure_stats.csv" % stem)
    relation_path = graphml_path.with_name("%s_relation_stats.csv" % stem)
    return summary_path, relation_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute MuSiQue-style graph structure statistics from a GraphML index."
    )
    parser.add_argument("graphml", help="Path to the GraphML file.")
    parser.add_argument(
        "--relation-field",
        default="keywords",
        help="Edge attribute used as the relation label. Defaults to 'keywords'.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="CSV path for summary metrics. Defaults to <graph>_structure_stats.csv",
    )
    parser.add_argument(
        "--relation-output",
        default="",
        help="CSV path for relation-level metrics. Defaults to <graph>_relation_stats.csv",
    )
    args = parser.parse_args()

    graphml_path = Path(args.graphml).expanduser().resolve()
    summary_output, relation_output = build_default_output_paths(graphml_path)
    if args.summary_output:
        summary_output = Path(args.summary_output).expanduser().resolve()
    if args.relation_output:
        relation_output = Path(args.relation_output).expanduser().resolve()

    graph_data = load_graphml(graphml_path, args.relation_field)
    summary_rows, relation_rows = compute_stats(graph_data, args.relation_field)
    summary_rows[0]["value"] = str(graphml_path)

    write_csv(summary_rows, summary_output, ["metric", "value"])
    write_csv(relation_rows, relation_output, ["relation", "edge_mentions", "unique_entities"])

    print("Summary CSV: %s" % summary_output)
    print("Relation CSV: %s" % relation_output)


if __name__ == "__main__":
    main()

