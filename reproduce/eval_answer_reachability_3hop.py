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
  python reproduce/eval_answer_reachability_3hop.py \
    --qa-file reproduce/result/query_set_entity_answers_single.json \
    --entities-file tests/qwen06b_old/vdb_entities_name.json \
    --relations-file tests/qwen06b_old/vdb_relationships.json \
    --k 3 \
    --out-json tests/qwen06b_old/reachability_3hop_report.json \
    --out-csv tests/qwen06b_old/reachability_3hop_samples.csv

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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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


def tokenize_name(text: str) -> List[str]:
    """Tokenize a name for lightweight alias/variant matching."""
    t = strip_wrapping_quotes(text).upper()
    toks = re.findall(r"[A-Z0-9]+", t)
    return [x for x in toks if x]


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


def bfs_within_k(graph: Dict[str, Set[str]], start_nodes: Iterable[str], k: int) -> Set[str]:
    visited: Set[str] = set()
    dq: deque[Tuple[str, int]] = deque()

    for s in start_nodes:
        if s in visited:
            continue
        visited.add(s)
        dq.append((s, 0))

    while dq:
        node, dist = dq.popleft()
        if dist == k:
            continue
        for nb in graph.get(node, ()):  # Missing nodes are treated as isolated.
            if nb in visited:
                continue
            visited.add(nb)
            dq.append((nb, dist + 1))

    return visited


def call_openai_compatible(
    question: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_s: int = 60,
) -> List[str]:
    """Return extracted start entities as a list of strings."""

    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Install it with 'pip install openai'.")

    system_prompt = (
        "You extract start entities from a question for graph retrieval. "
        "Return ONLY JSON with key 'start_entities' whose value is an array of strings. "
        "Include explicit named entities in the question that are natural starting points. "
        "No explanations."
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
            stream=True,
        )
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    content_parts: List[str] = []
    try:
        for chunk in completion:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            piece = getattr(delta, "content", None)
            if piece:
                content_parts.append(piece)
    except Exception as e:
        raise RuntimeError(f"LLM stream failed: {e}") from e

    content = "".join(content_parts).strip()
    if not content:
        return []

    # Some model outputs may be a raw JSON list or wrapped in markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if not m:
            return []
        parsed = json.loads(m.group(1))

    if isinstance(parsed, dict):
        arr = parsed.get("start_entities", [])
    elif isinstance(parsed, list):
        arr = parsed
    else:
        return []

    if not isinstance(arr, list):
        return []

    res = []
    for x in arr:
        if isinstance(x, str):
            t = x.strip()
            if t:
                res.append(t)
    return res


def load_cache(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        if isinstance(data, dict):
            cleaned: Dict[str, List[str]] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list):
                    cleaned[k] = [str(x) for x in v if isinstance(x, (str, int, float))]
            return cleaned
    except Exception:
        pass
    return {}


def save_cache(path: Path, cache: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def match_candidates(name: str, norm_to_names: Dict[str, Set[str]]) -> Set[str]:
    return set(norm_to_names.get(normalize_name(name), set()))


def match_answer_candidates(name: str, norm_to_names: Dict[str, Set[str]]) -> Set[str]:
    """
    Flexible answer matching:
    1) exact normalized match
    2) substring match on normalized forms (e.g., Nolan -> Christopher Nolan)
    3) token-subset match for minor mention variants/aliases
    """
    q_raw = strip_wrapping_quotes(name)
    q_norm = normalize_name(q_raw)
    if not q_norm:
        return set()

    matched = set(norm_to_names.get(q_norm, set()))

    q_tokens = set(tokenize_name(q_raw))

    # Substring fallback for abbreviated mentions.
    # Avoid extremely short strings to reduce false positives.
    if len(q_norm) >= 4:
        for cand_norm, cand_names in norm_to_names.items():
            if q_norm in cand_norm or cand_norm in q_norm:
                matched |= cand_names

    # Token-subset fallback for small formatting variants.
    # Example: "Wolfgang" can match "Wolfgang Schulz".
    if q_tokens:
        filtered_q_tokens = {t for t in q_tokens if len(t) >= 3}
        if filtered_q_tokens:
            for cand_names in norm_to_names.values():
                # All names in one bucket are equivalent normalized forms;
                # sample one representative for tokenization.
                rep = next(iter(cand_names))
                cand_tokens = set(tokenize_name(rep))
                if filtered_q_tokens.issubset(cand_tokens):
                    matched |= cand_names

    return matched


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate answer reachability within k hops.")
    parser.add_argument("--qa-file", required=True, type=Path)
    parser.add_argument("--entities-file", required=True, type=Path)
    parser.add_argument("--relations-file", required=True, type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("reproduce/result/question_start_entity_cache.json"),
    )
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between LLM calls.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="For debugging: process first N samples only; 0 means all.",
    )
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("DASHSCOPE_MODEL", "qwen3-1.7b")

    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY is required for LLM extraction.", file=sys.stderr)
        return 2

    qa_obj = load_json(args.qa_file)
    if not isinstance(qa_obj, dict):
        print("ERROR: QA file must be a JSON object keyed by sample id.", file=sys.stderr)
        return 2

    _, norm_to_names = parse_entities_name(args.entities_file)
    graph = parse_relationships_graph(args.relations_file)

    cache = load_cache(args.cache_file)

    sample_items = sorted(qa_obj.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))
    if args.max_samples > 0:
        sample_items = sample_items[: args.max_samples]

    rows = []

    total = 0
    answer_mappable = 0
    answer_hit_3hop = 0
    start_extracted = 0
    start_mapped = 0

    for sid, sample in sample_items:
        if not isinstance(sample, dict):
            continue
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        if not question or not answer:
            continue

        total += 1

        if question in cache:
            start_entities = cache[question]
        else:
            start_entities = call_openai_compatible(
                question=question,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            cache[question] = start_entities
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

        extracted_non_empty = len(start_entities) > 0
        if extracted_non_empty:
            start_extracted += 1

        # Map LLM outputs to graph entity names.
        matched_start_nodes: Set[str] = set()
        for s in start_entities:
            matched_start_nodes |= match_candidates(s, norm_to_names)

        start_in_entity_db = len(matched_start_nodes) > 0
        if start_in_entity_db:
            start_mapped += 1

        # Map answer to candidate graph entity nodes.
        answer_candidates = match_answer_candidates(answer, norm_to_names)
        answer_is_mappable = len(answer_candidates) > 0
        if answer_is_mappable:
            answer_mappable += 1

        reachable = False
        if answer_is_mappable and start_in_entity_db:
            nh = bfs_within_k(graph, matched_start_nodes, args.k)
            reachable = len(nh.intersection(answer_candidates)) > 0

        if answer_is_mappable and reachable:
            answer_hit_3hop += 1

        rows.append(
            {
                "sample_id": sid,
                "question": question,
                "answer": answer,
                "llm_start_entities": json.dumps(start_entities, ensure_ascii=False),
                "matched_start_nodes": json.dumps(sorted(matched_start_nodes), ensure_ascii=False),
                "answer_candidates": json.dumps(sorted(answer_candidates), ensure_ascii=False),
                "start_in_entity_db": int(start_in_entity_db),
                "answer_mappable": int(answer_is_mappable),
                f"reachable_at_{args.k}": int(reachable),
            }
        )

    save_cache(args.cache_file, cache)

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
            f"reachable_at_{args.k}",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "k": args.k,
        "total_samples": total,
        "start_entity_extracted_count": start_extracted,
        "start_entity_mapped_count": start_mapped,
        "start_entity_mapped_rate": (start_mapped / total) if total else 0.0,
        "answer_mappable_count": answer_mappable,
        "reachability_hit_count": answer_hit_3hop,
        "reachability_rate_over_answer_mappable": (answer_hit_3hop / answer_mappable) if answer_mappable else 0.0,
        "reachability_rate_over_total": (answer_hit_3hop / total) if total else 0.0,
        "output_csv": str(args.out_csv),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
