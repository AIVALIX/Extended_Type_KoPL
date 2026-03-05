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

gold_found_before = 0
gold_found_after = 0
gold_not_found = 0
total = 0

for sample in samples:
    question = sample['question']
    gold_rels = [sample['rel1'], sample['rel2']]
    
    result = pipeline.run(question)
    log = result.processing_log
    
    # Parse log to find before/after vector pruning
    before_pruning = []
    after_pruning = []
    section = None
    
    for entry in log:
        if 'BFS' in entry or 'Global paths' in entry or 'Stepwise paths' in entry:
            section = 'before'
        elif 'Vector' in entry and 'Pruning' in entry:
            section = 'after'
        elif 'Reranker' in entry:
            section = None
        
        if '-[' in entry:
            rels = re.findall(r'-\[([^\]]+)\]->', entry)
            if rels:
                if section == 'before':
                    before_pruning.append(rels)
                elif section == 'after' or 'score:' in entry:
                    after_pruning.append(rels)
    
    total += 1
    
    # Check if gold is in before/after
    found_before = any(rels == gold_rels for rels in before_pruning)
    found_after = any(rels == gold_rels for rels in after_pruning)
    
    if found_before:
        gold_found_before += 1
    if found_after:
        gold_found_after += 1
    if not found_before and not found_after:
        gold_not_found += 1
        print(f'Gold NOT found: {gold_rels}')
        print(f'  Before pruning ({len(before_pruning)}): {before_pruning[:5]}...')
        print(f'  After pruning ({len(after_pruning)}): {after_pruning[:3]}')
        print()

print('=' * 60)
print(f'Total: {total}')
print(f'Gold found BEFORE pruning: {gold_found_before}/{total} ({100*gold_found_before/total:.1f}%)')
print(f'Gold found AFTER pruning:  {gold_found_after}/{total} ({100*gold_found_after/total:.1f}%)')
print(f'Gold NOT found at all:     {gold_not_found}/{total} ({100*gold_not_found/total:.1f}%)')
