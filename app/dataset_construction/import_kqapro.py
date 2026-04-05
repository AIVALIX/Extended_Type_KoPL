"""
KQA Pro KB → Neo4j CSV 変換スクリプト

Usage:
    python app/dataset_construction/import_kqapro.py

Output:
    data/kqapro/import_nodes.csv
    data/kqapro/import_rels.csv
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path


def sanitize_label(name: str) -> str:
    """Neo4j ラベル用に文字列をサニタイズ"""
    # スペース→アンダースコア、特殊文字除去
    s = re.sub(r'[^a-zA-Z0-9_]', '_', name.replace(' ', '_'))
    # 先頭が数字の場合はプレフィックス追加
    if s and s[0].isdigit():
        s = 'T_' + s
    return s


def main():
    base = Path(__file__).resolve().parent.parent.parent / 'data' / 'kqapro'
    kb_path = base / 'kb.json'

    print(f"Loading {kb_path} ...")
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    concepts = kb['concepts']
    entities = kb['entities']
    print(f"Concepts: {len(concepts)}, Entities: {len(entities)}")

    # concept ID → name マッピング
    concept_names = {}
    for cid, c in concepts.items():
        concept_names[cid] = c['name']

    # ========= Nodes =========
    nodes = {}  # id -> {name, labels}

    # Concepts をノードとして追加
    for cid, c in concepts.items():
        label = sanitize_label(c['name'])
        nodes[cid] = {
            'name': c['name'],
            'labels': ['Concept', label],
        }

    # Entities をノードとして追加
    for eid, ent in entities.items():
        # instanceOf から最初のconcept名をラベルに使用
        labels = []
        for iof in ent.get('instanceOf', []):
            cname = concept_names.get(iof, '')
            if cname:
                labels.append(sanitize_label(cname))
        if not labels:
            labels = ['Entity']

        nodes[eid] = {
            'name': ent['name'],
            'labels': labels,
        }

    print(f"Total nodes: {len(nodes)}")

    # ========= Relations =========
    edges = []
    rel_counter = Counter()

    for eid, ent in entities.items():
        for rel in ent.get('relations', []):
            predicate = rel['predicate']
            obj = rel['object']
            direction = rel.get('direction', 'forward')

            # 正規化: スペース→アンダースコア
            rel_type = predicate.replace(' ', '_')

            if direction == 'forward':
                start_id = eid
                end_id = obj
            else:
                start_id = obj
                end_id = eid

            # objectがnodes内に存在する場合のみ追加
            if start_id in nodes and end_id in nodes:
                edges.append((start_id, end_id, rel_type))
                rel_counter[rel_type] += 1

    # 重複除去
    edges_unique = list(set(edges))
    print(f"Total edges: {len(edges)} -> deduplicated: {len(edges_unique)}")
    print(f"Unique relation types: {len(rel_counter)}")

    # ========= CSV出力 =========
    nodes_path = base / 'import_nodes.csv'
    rels_path = base / 'import_rels.csv'

    # Nodes CSV
    with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id:ID', 'name', ':LABEL'])
        for nid, node in nodes.items():
            label_str = ';'.join(node['labels'])
            writer.writerow([nid, node['name'], label_str])

    print(f"Written: {nodes_path} ({len(nodes)} nodes)")

    # Relations CSV
    with open(rels_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([':START_ID', ':END_ID', ':TYPE'])
        for start, end, rtype in edges_unique:
            writer.writerow([start, end, rtype])

    print(f"Written: {rels_path} ({len(edges_unique)} edges)")

    # 統計
    print(f"\nTop 10 relation types:")
    for r, c in rel_counter.most_common(10):
        print(f"  {r}: {c}")

    label_counter = Counter()
    for node in nodes.values():
        for l in node['labels']:
            label_counter[l] += 1
    print(f"\nTop 10 node labels:")
    for l, c in label_counter.most_common(10):
        print(f"  {l}: {c}")


if __name__ == '__main__':
    main()
