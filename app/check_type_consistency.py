#!/usr/bin/env python
import json
import sys

same_type = 0
diff_type = 0
details = []

for line in sys.stdin:
    d = json.loads(line)
    anchor_types = d.get('anchor_types', [])
    mid_nodes = d.get('mid_nodes_sample', [])
    answer_nodes = d.get('answer_nodes_sample', [])
    rel1 = d.get('rel1', '')
    rel2 = d.get('rel2', '')

    if not mid_nodes or not answer_nodes:
        continue

    mid_types = mid_nodes[0].get('types', [])
    ans_types = answer_nodes[0].get('types', [])

    # タイプを文字列に変換
    a_type = anchor_types[0] if isinstance(anchor_types, list) and anchor_types else str(anchor_types)
    m_type = mid_types[0] if isinstance(mid_types, list) and mid_types else str(mid_types)
    x_type = ans_types[0] if isinstance(ans_types, list) and ans_types else str(ans_types)

    # parent-childの場合にタイプが同じかどうかをチェック
    has_diff = False
    if rel1 == 'parent-child' and a_type != m_type:
        has_diff = True
    if rel2 == 'parent-child' and m_type != x_type:
        has_diff = True

    if has_diff:
        diff_type += 1
        details.append(f'{a_type} -[{rel1}]-> {m_type} -[{rel2}]-> {x_type}')
    else:
        same_type += 1

print(f'=== parent-childを含むパスのタイプ整合性 ===')
print(f'同タイプ間 (正常): {same_type}')
print(f'異タイプ間 (問題): {diff_type}')
print(f'問題率: {diff_type / (same_type + diff_type) * 100:.1f}%')
print()
print('=== 異タイプ間の例 (最初の15件) ===')
for d in details[:15]:
    print(f'  {d}')
