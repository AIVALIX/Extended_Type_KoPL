import json
import random
from pathlib import Path

# Load two_hop dataset
dataset_path = Path('/app/result/dataset_v3/two_hop.jsonl')
samples = []
with open(dataset_path) as f:
    for line in f:
        samples.append(json.loads(line))

random.seed(42)
sample = random.sample(samples, 1)[0]

from pipeline.extended_type_kopl.pipeline import ExtendedTypeKoPLPipeline

pipeline = ExtendedTypeKoPLPipeline(
    kg_type='primekgqa',
    use_schema_relations=False,
    reranker_type='llm',
    reranker_input_k=10,
)

question = sample['question']
gold_rels = [sample['rel1'], sample['rel2']]

print(f'Question: {question}')
print(f'Gold: {gold_rels}')
print()
print('Full log:')
print('=' * 60)

result = pipeline.run(question)
for entry in result.processing_log:
    print(entry)
