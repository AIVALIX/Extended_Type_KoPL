import json
from pathlib import Path

dataset_path = Path('/app/result/dataset_v3/two_hop.jsonl')
with open(dataset_path) as f:
    sample = json.loads(f.readline())

from pipeline.extended_type_kopl.pipeline import ExtendedTypeKoPLPipeline

pipeline = ExtendedTypeKoPLPipeline(
    kg_type='primekgqa',
    use_schema_relations=False,
    reranker_type='none',
    reranker_input_k=10,
)

question = sample['question']
gold_rels = [sample['rel1'], sample['rel2']]

print(f'Question: {question[:60]}...')
print(f'Gold: {gold_rels}')
print()

result = pipeline.run(question)

# Count paths with each relation combo
from collections import Counter
rel_combos = Counter()

for entry in result.processing_log:
    if '[stepwise-bfs]' in entry or '[global]' in entry:
        import re
        rels = re.findall(r'-\[([^\]]+)\]->', entry)
        if rels:
            rel_combos[tuple(rels)] += 1

print('Phase 2で生成されたパスのリレーション組み合わせ:')
for combo, cnt in rel_combos.most_common(20):
    marker = ' ← GOLD' if list(combo) == gold_rels else ''
    print(f'  {list(combo)}: {cnt}{marker}')
