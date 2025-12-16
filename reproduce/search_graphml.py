#!/usr/bin/env python3
"""
Stream-search GraphML for a query string without loading whole file.
Usage: python search_graphml_stream.py <query> <file.graphml>
Optional: add --ignore-case to do case-insensitive search.
"""
import sys
import argparse
import xml.etree.ElementTree as ET


def parse_node_details(elem):
    """提取节点的所有data字段作为详细描述"""
    details = {}
    for d in elem.findall('.//{http://graphml.graphdrawing.org/xmlns}data'):
        key = d.get('key', 'unknown')
        text = d.text.strip() if d.text else ''
        if text:
            details[key] = text
    # 也尝试不带命名空间
    for d in elem.findall('.//data'):
        key = d.get('key', 'unknown')
        text = d.text.strip() if d.text else ''
        if text:
            details[key] = text
    return details


def parse_edge_details(elem):
    """提取边的所有data字段"""
    details = {}
    for d in elem.findall('.//{http://graphml.graphdrawing.org/xmlns}data'):
        key = d.get('key', 'unknown')
        text = d.text.strip() if d.text else ''
        if text:
            details[key] = text
    for d in elem.findall('.//data'):
        key = d.get('key', 'unknown')
        text = d.text.strip() if d.text else ''
        if text:
            details[key] = text
    return details


def search_graphml(query, fname, ignore_case=False, max_reports=1000):
    q = query.lower() if ignore_case else query
    found = []
    edges = []  # 存储所有边的信息
    count = 0
    
    # 第一遍：收集所有边和匹配的节点
    for event, elem in ET.iterparse(fname, events=("end",)):
        tag = elem.tag
        
        # 收集边信息
        if tag.endswith('}edge') or tag == 'edge':
            source = elem.get('source')
            target = elem.get('target')
            edge_details = parse_edge_details(elem)
            edges.append({
                'source': source,
                'target': target,
                'details': edge_details
            })
            elem.clear()
            continue
            
        if tag.endswith('}node') or tag == 'node':
            nid = elem.get('id')
            matched = False
            node_details = parse_node_details(elem)
            
            # check <data> children text
            for d in elem.findall('.//'):
                if d.text:
                    txt = d.text.strip()
                    if (ignore_case and q in txt.lower()) or (not ignore_case and q in txt):
                        matched = True
                        break
            # also check common attributes (e.g., label)
            if not matched:
                for k, v in elem.items():
                    if v:
                        if (ignore_case and q in v.lower()) or (not ignore_case and q in v):
                            matched = True
                            break
            
            if matched:
                count += 1
                if count <= max_reports:
                    found.append({'id': nid, 'details': node_details})
                    
            elem.clear()
    
    # 打印匹配结果
    if count == 0:
        print("No matches found.")
    else:
        print(f"Found {count} matching nodes (showing up to {max_reports}).\n")
        print("=" * 80)
        
        for node in found:
            nid = node['id']
            details = node['details']
            
            # 打印节点详细信息
            print(f"\n【实体】 ID: {nid}")
            print("-" * 40)
            print("【详细描述】")
            for key, value in details.items():
                # 截断过长内容
                display_val = value[:500] + "..." if len(value) > 500 else value
                print(f"  {key}: {display_val}")
            
            # 查找并打印相关关系
            print("\n【相关关系】")
            related_count = 0
            for edge in edges:
                if edge['source'] == nid or edge['target'] == nid:
                    related_count += 1
                    direction = "→" if edge['source'] == nid else "←"
                    other = edge['target'] if edge['source'] == nid else edge['source']
                    print(f"  {direction} {other}")
                    for ek, ev in edge['details'].items():
                        display_ev = ev[:200] + "..." if len(ev) > 200 else ev
                        print(f"      {ek}: {display_ev}")
            
            if related_count == 0:
                print("  (无相关关系)")
            
            print("=" * 80)
    
    return [n['id'] for n in found]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Stream-search GraphML for a query string')
    p.add_argument('--query', default ='HOUSE RULES')
    p.add_argument('--file', default='./tests/Qwen3-4B-Instruct-2507_vllm_debug/graph_chunk_entity_relation.graphml')
    p.add_argument('--ignore-case', action='store_true')
    p.add_argument('--max-reports', type=int, default=50, help='Max matches to print')
    args = p.parse_args()

    try:
        search_graphml(args.query, args.file, ignore_case=args.ignore_case, max_reports=args.max_reports)
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        sys.exit(2)
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        sys.exit(3)
