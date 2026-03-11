"""
统计 vdb_entities_name.json 中的实体信息。

用法:
    python analyze_vdb_entities.py [--path PATH] [--top N] [--show-samples N]
"""

import json
import os
import sys
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import random


# 默认 JSON 文件路径（相对于本脚本）
DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "Qwen3-4B-Instruct-2507_vllm", "vdb_entities_name.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="统计 vdb_entities_name.json 实体信息")
    parser.add_argument("--path", default=DEFAULT_JSON_PATH, help="JSON 文件路径")
    parser.add_argument("--top", type=int, default=20, help="显示频次最高的 N 个词（默认 20）")
    parser.add_argument("--show-samples", type=int, default=10, help="显示随机采样实体数量（默认 10）")
    return parser.parse_args()


def strip_quotes(name: str) -> str:
    """去除实体名称首尾的引号"""
    return name.strip('"').strip("'")


def main():
    args = parse_args()

    print(f"正在加载文件: {args.path}")
    file_size_mb = os.path.getsize(args.path) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB\n")

    with open(args.path, "r", encoding="utf-8") as f:
        db = json.load(f)

    embedding_dim = db.get("embedding_dim", "未知")
    records = db.get("data", [])

    print("=" * 60)
    print("【基本信息】")
    print(f"  embedding_dim : {embedding_dim}")
    print(f"  实体总数      : {len(records)}")

    # ── 提取字段 ──────────────────────────────────────────────────
    entity_names_raw = [r["entity_name"] for r in records]
    entity_names = [strip_quotes(n) for n in entity_names_raw]
    created_ats = [r["__created_at__"] for r in records]

    # ── 重复情况 ───────────────────────────────────────────────────
    name_counter = Counter(entity_names)
    dup_count = sum(1 for c in name_counter.values() if c > 1)
    dup_total = sum(c - 1 for c in name_counter.values() if c > 1)
    unique_count = len(name_counter)

    print()
    print("=" * 60)
    print("【实体名称统计】")
    print(f"  唯一实体数      : {unique_count}")
    print(f"  重复名称数      : {dup_count}  （冗余记录共 {dup_total} 条）")

    # ── 名称长度分布 ───────────────────────────────────────────────
    lengths = [len(n) for n in entity_names]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    min_len = min(lengths)
    max_len = max(lengths)
    length_counter = Counter(lengths)

    print()
    print("=" * 60)
    print("【实体名称长度分布】")
    print(f"  最短 / 最长 / 平均 : {min_len} / {max_len} / {avg_len:.1f} 字符")
    buckets = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, max_len + 1)]
    print(f"  {'长度区间':<12}{'实体数':>8}  {'占比':>7}  {'分布'}")
    for lo, hi in buckets:
        cnt = sum(v for k, v in length_counter.items() if lo <= k < hi)
        if cnt == 0:
            continue
        bar = "█" * min(int(cnt / len(lengths) * 40), 40)
        print(f"  [{lo:>3}, {hi:<4})   {cnt:>8}  {cnt/len(lengths)*100:>6.1f}%  {bar}")

    # ── 最短/最长实体举例 ──────────────────────────────────────────
    shortest = sorted([(len(n), n) for n in entity_names])[:5]
    longest = sorted([(len(n), n) for n in entity_names], reverse=True)[:5]
    print()
    print("  最短的 5 个实体名称:")
    for l, n in shortest:
        print(f"    ({l:>3} 字符)  {n}")
    print()
    print("  最长的 5 个实体名称:")
    for l, n in longest:
        print(f"    ({l:>3} 字符)  {n[:80]}{'...' if len(n) > 80 else ''}")

    # ── 时间分布 ───────────────────────────────────────────────────
    ts_min = min(created_ats)
    ts_max = max(created_ats)
    dt_min = datetime.fromtimestamp(ts_min).strftime("%Y-%m-%d %H:%M:%S")
    dt_max = datetime.fromtimestamp(ts_max).strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 60)
    print("【创建时间范围】")
    print(f"  最早 : {dt_min}")
    print(f"  最晚 : {dt_max}")

    hour_counter: dict = defaultdict(int)
    for ts in created_ats:
        hour_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H")
        hour_counter[hour_str] += 1
    if len(hour_counter) <= 48:
        print(f"  {'时间段 (小时)':<25}{'实体数':>8}  {'分布'}")
        max_h = max(hour_counter.values())
        for h in sorted(hour_counter):
            cnt = hour_counter[h]
            bar = "█" * min(int(cnt / max_h * 30), 30)
            print(f"  {h}      {cnt:>8}  {bar}")

    # ── 重复最多的实体 ─────────────────────────────────────────────
    if dup_count > 0:
        print()
        print("=" * 60)
        print(f"【重复次数最多的实体 Top {min(10, dup_count)}】")
        shown = 0
        for name, cnt in name_counter.most_common():
            if cnt > 1:
                print(f"  {cnt:>4}x  {name}")
                shown += 1
                if shown >= 10:
                    break

    # ── 高频词（词级别统计）────────────────────────────────────────
    word_counter: Counter = Counter()
    for name in entity_names:
        for word in name.split():
            word_counter[word.upper()] += 1

    print()
    print("=" * 60)
    print(f"【实体名称中出现频次最高的 {args.top} 个词】")
    print(f"  {'词':<35}{'出现次数':>10}")
    for word, cnt in word_counter.most_common(args.top):
        print(f"  {word:<35}{cnt:>10}")

    # ── ID 前缀统计 ────────────────────────────────────────────────
    prefixes: Counter = Counter()
    for r in records:
        prefix = r["__id__"].split("-")[0] if "-" in r["__id__"] else r["__id__"]
        prefixes[prefix] += 1
    print()
    print("=" * 60)
    print("【__id__ 前缀类型】")
    for prefix, cnt in prefixes.most_common():
        print(f"  {prefix:<25}{cnt:>8}")

    # ── 随机采样展示 ───────────────────────────────────────────────
    random.seed(42)
    sample_n = min(args.show_samples, len(entity_names))
    samples = random.sample(list(enumerate(entity_names)), sample_n)
    print()
    print("=" * 60)
    print(f"【随机采样实体（{sample_n} 条）】")
    for idx, name in sorted(samples):
        print(f"  [{idx:>5}]  {name}")

    print()
    print("=" * 60)
    print("统计完毕。")


if __name__ == "__main__":
    main()
