import json
import random
from pathlib import Path
import re
from collections import Counter

# Load two_hop dataset
dataset_path = Path('/app/result/dataset_v3/two_hop.jsonl')
samples = []
with open(dataset_path) as f:
    for line in f:
        samples.append(json.loads(line))

random.seed(42)
samples = random.sample(samples, 30)

from pipeline.extended_type_kopl.pipeline import ExtendedTypeKoPLPipeline

pipeline = ExtendedTypeKoPLPipeline(
    kg_type='primekgqa',
    use_schema_relations=False,
    reranker_type='llm',
    reranker_input_k=10,
)

gold_patterns = Counter()
predicted_patterns = Counter()
mismatch_details = []

for sample in samples:
    question = sample['question']
    gold_rels = [sample['rel1'], sample['rel2']]
    gold_key = ' -> '.join(gold_rels)
    gold_patterns[gold_key] += 1
    
    result = pipeline.run(question)
    log = result.processing_log
    
    # Extract top-1 candidate
    for entry in log:
        if 'score:' in entry and '-[' in entry:
            rels = re.findall(r'-\[([^\]]+)\]->', entry)
            if rels:
                pred_key = ' -> '.join(rels)
                predicted_patterns[pred_key] += 1
                
                if rels != gold_rels:
                    mismatch_details.append({
                        'gold': gold_rels,
                        'pred': rels,
                    })
                break

print('Gold relation patterns:')
for k, v in gold_patterns.most_common(10):
    print(f'  {k}: {v}')

print()
print('Predicted top-1 patterns:')
for k, v in predicted_patterns.most_common(10):
    print(f'  {k}: {v}')

print()
print('Mismatch patterns (gold => predicted):')
mismatch_counter = Counter()
for m in mismatch_details:
    key = str(m['gold']) + ' => ' + str(m['pred'])
    mismatch_counter[key] += 1

for k, v in mismatch_counter.most_common(10):
    print(f'  {k}: {v}')
