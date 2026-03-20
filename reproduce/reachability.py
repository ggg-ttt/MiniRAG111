#!/usr/bin/env python3
"""
Evaluate whether answer entities are reachable within k hops from LLM-extracted
start entities (from question text) on MiniRAG graph indexes.

Pipeline per sample:
1) Use LLM to extract start entities from question.
2) Check whether extracted start entities exist in vdb_entities_name.json.
3) Build graph from vdb_relationships.json and test whether answer entity is
   within k hops from any matched start node.

Usage:
python reproduce/reachability.py --qa-file reproduce/result/query_set_entity_answers_single.json --entities-file tests/Qwen3-4B-Instruct-2507_vllm_debug/vdb_entities_name.json --relations-file tests/Qwen3-4B-Instruct-2507_vllm_debug/vdb_relationships.json --k 1 2 3 --out-json reproduce/result/all_reachability_khop_report.json --out-csv reproduce/result/all_reachability_khop_samples.csv --prompt-mode light
Required env vars for LLM extraction (DashScope OpenAI-compatible API):
  DASHSCOPE_API_KEY
Optional:
  DASHSCOPE_BASE_URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  DASHSCOPE_MODEL    (default: qwen3-1.7b)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from openai import OpenAI
from tqdm import tqdm

PROMPTS = {}

system_prompt_mini = """---Role---

You are a helpful assistant tasked with identifying entities from the user's query.

---Goal---

Given the query, extract specific entities, details, or concrete terms from the query.

---Instructions---

- Output in JSON format.
- The JSON should have one key:
  - "entities_from_query" for specific entities or details. It must be extracted from the query.
- Keep the same language as the Query.
- No explanation.

######################
-Real Data-
######################
Query: {query}
######################
Output:

"""
PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]

system_prompt_light = PROMPTS["keywords_extraction"].format(
    examples="\n\n".join(PROMPTS["keywords_extraction_examples"]),
    query="{query}",
)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
        return t[1:-1].strip()
    return t


def normalize_name(text: str) -> str:
    """Aggressive normalization for fuzzy matching aliases and formatting variants."""
    t = strip_wrapping_quotes(text)
    t = t.upper()
    # Keep alnum only so variants like Li Hua / LiHua / LI_HUA align.
    t = re.sub(r"[^A-Z0-9]+", "", t)
    return t


def parse_entities_name(path: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    obj = load_json(path)
    data = obj.get("data", [])

    canonical_names: Set[str] = set()
    norm_to_names: Dict[str, Set[str]] = defaultdict(set)

    for item in data:
        raw = str(item.get("entity_name", "")).strip()
        if not raw:
            continue
        cleaned = strip_wrapping_quotes(raw)
        canonical_names.add(cleaned)
        norm_to_names[normalize_name(cleaned)].add(cleaned)

    return canonical_names, norm_to_names


def parse_relationships_graph(path: Path) -> Dict[str, Set[str]]:
    obj = load_json(path)
    data = obj.get("data", [])

    graph: Dict[str, Set[str]] = defaultdict(set)

    for item in data:
        src = str(item.get("src_id", "")).strip()
        tgt = str(item.get("tgt_id", "")).strip()
        if not src or not tgt:
            continue

        src_clean = strip_wrapping_quotes(src)
        tgt_clean = strip_wrapping_quotes(tgt)
        if not src_clean or not tgt_clean:
            continue

        # Undirected expansion for k-hop neighborhood reachability.
        graph[src_clean].add(tgt_clean)
        graph[tgt_clean].add(src_clean)

    return graph


def bfs_min_distance_within_k(
    graph: Dict[str, Set[str]], start_nodes: Iterable[str], max_k: int
) -> Dict[str, int]:
    """Return minimum hop distance to each visited node up to max_k."""
    min_dist: Dict[str, int] = {}
    dq: deque[Tuple[str, int]] = deque()

    for s in start_nodes:
        if s in min_dist:
            continue
        min_dist[s] = 0
        dq.append((s, 0))

    while dq:
        node, dist = dq.popleft()
        if dist == max_k:
            continue
        for nb in graph.get(node, ()):  # Missing nodes are treated as isolated.
            if nb in min_dist:
                continue
            next_dist = dist + 1
            min_dist[nb] = next_dist
            dq.append((nb, next_dist))

    return min_dist


def call_openai_compatible(
    question: str,
    api_key: str,
    base_url: str,
    model: str,
    prompt_mode: str = "mini",
    timeout_s: int = 60,
) -> List[str]:
    """Return extracted start entities as a list of strings."""

    if prompt_mode in {"light", "hybrid"}:
        system_prompt = system_prompt_light.format(query=question)
    else:
        system_prompt = system_prompt_mini.format(query=question)
    user_prompt = "Return only JSON."

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        extra_body={"enable_thinking": True},
        stream=True,
    )

    content_parts: List[str] = []
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            content_parts.append(delta.content)

    content = "".join(content_parts).strip()
    if not content:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM JSON response: {content[:500]}") from e
    if prompt_mode in {"light", "hybrid"}:
        # For light/hybrid prompt experiments, use low-level keywords or low+high keywords.
        low_level = parsed.get("low_level_keywords", [])
        high_level = parsed.get("high_level_keywords", [])
        if isinstance(low_level, list) and isinstance(high_level, list):
            if prompt_mode == "hybrid":
                raw_entities = low_level + high_level
            else:
                raw_entities = low_level
        elif isinstance(low_level, list):
            raw_entities = low_level
        elif isinstance(high_level, list):
            raw_entities = high_level
        else:
            raw_entities = []
    else:
        # Compatible with both legacy key and mini prompt key.
        raw_entities = parsed.get("start_entities", parsed.get("entities_from_query", []))

    if isinstance(raw_entities, str):
        return [x.strip() for x in re.split(r"[,\n;，、]+", raw_entities) if x.strip()]

    if isinstance(raw_entities, list):
        res: List[str] = []
        for x in raw_entities:
            if isinstance(x, str):
                t = x.strip()
                if t:
                    res.append(t)
        return res

    return []


def match_candidates(name: str, norm_to_names: Dict[str, Set[str]]) -> Set[str]:
    return set(norm_to_names.get(normalize_name(name), set()))


def ensure_prefixed_filename(path: Path, prefix: str) -> Path:
    if path.name.startswith(prefix):
        return path
    return path.with_name(f"{prefix}{path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate answer reachability within k hops.")
    parser.add_argument("--qa-file", required=True, type=Path)
    parser.add_argument("--entities-file", required=True, type=Path)
    parser.add_argument("--relations-file", required=True, type=Path)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3],
        help="One or more hop values, e.g. --k 1 2 3",
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument(
        "--prompt-mode",
        choices=["mini", "light", "hybrid"],
        default="mini",
        help="Prompt template used for start-node extraction.",
    )
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between LLM calls.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="For debugging: process first N samples only; 0 means all.",
    )
    args = parser.parse_args()

    k_values = []
    for k in args.k:
        if k < 0:
            print("ERROR: --k must contain non-negative integers.", file=sys.stderr)
            return 2
        if k not in k_values:
            k_values.append(k)
    max_k = max(k_values) if k_values else 0

    # Use mode-specific prefixed output artifacts for this experiment variant.
    if args.prompt_mode == "light":
        mode_prefix = "low_"
    elif args.prompt_mode == "hybrid":
        mode_prefix = "hybrid_"
    else:
        mode_prefix = "mini_"
    k_tag = "-".join(str(k) for k in k_values)
    out_prefix = f"{mode_prefix}k{k_tag}_"
    args.out_json = ensure_prefixed_filename(args.out_json, out_prefix)
    args.out_csv = ensure_prefixed_filename(args.out_csv, out_prefix)

    api_key = "sk-27bca34541014e628299970d1a6323b5"
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    model = os.getenv("DASHSCOPE_MODEL", "qwen3-1.7b").strip()

    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY is required for LLM extraction.", file=sys.stderr)
        return 2

    qa_obj = load_json(args.qa_file)
    if not isinstance(qa_obj, dict):
        print("ERROR: QA file must be a JSON object keyed by sample id.", file=sys.stderr)
        return 2

    _, norm_to_names = parse_entities_name(args.entities_file)
    graph = parse_relationships_graph(args.relations_file)
    sample_items = sorted(qa_obj.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))
    if args.max_samples > 0:
        sample_items = sample_items[: args.max_samples]

    rows = []

    total = 0
    answer_mappable = 0
    answer_hit_by_k: Dict[int, int] = {k: 0 for k in k_values}
    start_extracted = 0
    start_mapped = 0

    for sid, sample in tqdm(sample_items, desc="Evaluating", total=len(sample_items), unit="sample"):
        if not isinstance(sample, dict):
            continue
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        if not question or not answer:
            continue

        total += 1

        start_entities = call_openai_compatible(
            question=question,
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt_mode=args.prompt_mode,
        )
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

        if start_entities:
            start_extracted += 1

        matched_start_nodes: Set[str] = set()
        for s in start_entities:
            matched_start_nodes |= match_candidates(s, norm_to_names)

        start_in_entity_db = len(matched_start_nodes) > 0
        if start_in_entity_db:
            start_mapped += 1

        answer_candidates = match_candidates(answer, norm_to_names)
        answer_is_mappable = len(answer_candidates) > 0
        if answer_is_mappable:
            answer_mappable += 1

        reachable_by_k: Dict[int, bool] = {k: False for k in k_values}
        if answer_is_mappable and start_in_entity_db:
            min_dist = bfs_min_distance_within_k(graph, matched_start_nodes, max_k)
            min_answer_dist = min((min_dist.get(a) for a in answer_candidates), default=None)
            if min_answer_dist is not None:
                for k in k_values:
                    reachable_by_k[k] = min_answer_dist <= k

        if answer_is_mappable:
            for k in k_values:
                if reachable_by_k[k]:
                    answer_hit_by_k[k] += 1

        row = {
            "sample_id": sid,
            "question": question,
            "answer": answer,
            "llm_start_entities": json.dumps(start_entities, ensure_ascii=False),
            "matched_start_nodes": json.dumps(sorted(matched_start_nodes), ensure_ascii=False),
            "answer_candidates": json.dumps(sorted(answer_candidates), ensure_ascii=False),
            "start_in_entity_db": int(start_in_entity_db),
            "answer_mappable": int(answer_is_mappable),
        }
        for k in k_values:
            row[f"reachable_at_{k}"] = int(reachable_by_k[k])
        rows.append(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "question",
            "answer",
            "llm_start_entities",
            "matched_start_nodes",
            "answer_candidates",
            "start_in_entity_db",
            "answer_mappable",
        ] + [f"reachable_at_{k}" for k in k_values]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    per_k = {
        str(k): {
            "reachability_hit_count": answer_hit_by_k[k],
            "reachability_rate_over_answer_mappable": (answer_hit_by_k[k] / answer_mappable) if answer_mappable else 0.0,
            "reachability_rate_over_total": (answer_hit_by_k[k] / total) if total else 0.0,
        }
        for k in k_values
    }

    report = {
        "k_values": k_values,
        "total_samples": total,
        "start_entity_extracted_count": start_extracted,
        "start_entity_mapped_count": start_mapped,
        "start_entity_mapped_rate": (start_mapped / total) if total else 0.0,
        "answer_mappable_count": answer_mappable,
        "per_k": per_k,
        "output_csv": str(args.out_csv),
    }

    if len(k_values) == 1:
        only_k = k_values[0]
        report["k"] = only_k
        report["reachability_hit_count"] = answer_hit_by_k[only_k]
        report["reachability_rate_over_answer_mappable"] = per_k[str(only_k)]["reachability_rate_over_answer_mappable"]
        report["reachability_rate_over_total"] = per_k[str(only_k)]["reachability_rate_over_total"]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
