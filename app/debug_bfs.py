import json
from pathlib import Path

# Load a sample
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

# Check relations between types
print('=== Relations between types ===')
print('drug -> disease:')
rels = pipeline._get_relations_between_types('drug', 'disease')
for rel, dir in rels[:10]:
    print(f'  {rel} ({dir})')

print('\ndisease -> drug:')
rels = pipeline._get_relations_between_types('disease', 'drug')
for rel, dir in rels[:10]:
    print(f'  {rel} ({dir})')

# Check BFS paths
print('\n=== BFS paths from drug to drug (via disease) ===')
paths = pipeline._find_paths_between_types('drug', 'drug', max_depth=2)
print(f'Found {len(paths)} paths')
for p in paths[:10]:
    rels = ' -> '.join(p.relations)
    types = ' -> '.join(p.types)
    dirs = ' '.join(p.directions)
    print(f'  {types}')
    print(f'    Relations: {rels}')
    print(f'    Directions: {dirs}')
