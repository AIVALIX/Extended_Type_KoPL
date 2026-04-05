"""
KQA Pro val.json → ETK用 JSONL 変換

全タイプの質問（entity, count, attr, relation, verify, select）を変換。

Usage:
    python app/dataset_construction/convert_kqapro.py

Output:
    data/kqapro/qa/val_entity.jsonl  (entity retrieval: What)
    data/kqapro/qa/val_all.jsonl     (全タイプ)
"""
import json
from pathlib import Path
from collections import Counter


# Last function → answer_type mapping
ANSWER_TYPE_MAP = {
    "What": "entity",
    "Count": "count",
    "QueryAttr": "attr",
    "QueryRelation": "relation",
    "QueryAttrQualifier": "attr",
    "QueryRelationQualifier": "relation",
    "QueryAttrUnderCondition": "attr",
    "VerifyStr": "verify",
    "VerifyNum": "verify",
    "VerifyYear": "verify",
    "VerifyDate": "verify",
    "SelectBetween": "select",
    "SelectAmong": "select",
    # These are intermediate, but sometimes appear as last function
    "Find": "entity",
    "Relate": "entity",
    "FilterConcept": "entity",
    "And": "entity",
    "Or": "entity",
}


def extract_entity_and_relations(program: list) -> tuple:
    """KoPLプログラムからエンティティとリレーションを抽出。

    Returns:
        (entity_name, relations_list)
    """
    entity_name = None
    relations = []

    for step in program:
        func = step.get('function', '')
        inputs = step.get('inputs', [])

        if func == 'Find' and inputs and entity_name is None:
            entity_name = inputs[0]
        elif func == 'Relate' and inputs:
            if len(inputs) >= 1:
                relations.append(inputs[0])

    return entity_name, relations


def extract_query_info(program: list) -> dict:
    """Extract answer-type specific info from the KoPL program."""
    info = {}
    last = program[-1] if program else {}
    func = last.get('function', '')
    inputs = last.get('inputs', [])

    if func in ('QueryAttr', 'QueryAttrUnderCondition'):
        info['query_key'] = inputs[0] if inputs else None
    elif func == 'QueryAttrQualifier':
        # QueryAttrQualifier(key, value, qualifier_key)
        if len(inputs) >= 3:
            info['query_key'] = inputs[2]  # qualifier key is the answer
    elif func in ('VerifyStr', 'VerifyNum', 'VerifyYear', 'VerifyDate'):
        info['verify_value'] = inputs[0] if inputs else None
        info['verify_op'] = inputs[1] if len(inputs) > 1 else '='
        # Find the query key from the preceding QueryAttr step
        deps = last.get('dependencies', [])
        if deps:
            dep_step = program[deps[0]]
            if dep_step['function'] in ('QueryAttr', 'QueryAttrUnderCondition'):
                info['query_key'] = dep_step['inputs'][0] if dep_step.get('inputs') else None
    elif func == 'SelectBetween':
        info['query_key'] = inputs[0] if inputs else None
        info['select_mode'] = inputs[1] if len(inputs) > 1 else None
        # Find the two entities from dependencies
        deps = last.get('dependencies', [])
        if len(deps) >= 2:
            for d in deps:
                dep_step = program[d]
                if dep_step['function'] == 'Find' and dep_step.get('inputs'):
                    if 'select_entity_a' not in info:
                        info['select_entity_a'] = dep_step['inputs'][0]
                    else:
                        info['select_entity_b'] = dep_step['inputs'][0]
    elif func == 'SelectAmong':
        info['query_key'] = inputs[0] if inputs else None
        info['select_mode'] = inputs[1] if len(inputs) > 1 else None

    return info


def main():
    base = Path(__file__).resolve().parent.parent.parent / 'data' / 'kqapro'
    val_path = base / 'val.json'
    out_dir = base / 'qa'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {val_path} ...")
    with open(val_path, 'r', encoding='utf-8') as f:
        val = json.load(f)

    print(f"Total questions: {len(val)}")

    entity_only = []
    all_samples = []
    type_counter = Counter()
    skip_counter = Counter()

    for q in val:
        program = q.get('program', [])
        answer = q.get('answer', '')
        if not program:
            skip_counter['no_program'] += 1
            continue

        last_func = program[-1].get('function', '')
        answer_type = ANSWER_TYPE_MAP.get(last_func)
        if not answer_type:
            skip_counter[f'unknown_func:{last_func}'] += 1
            continue

        entity_name, relations = extract_entity_and_relations(program)

        # Answers: always a list of strings
        if isinstance(answer, list):
            answers = [str(a) for a in answer]
        else:
            answers = [str(answer)]

        sample = {
            'question': q['question'],
            'entity': entity_name or '',
            'relation': relations[0] if relations else '',
            'answers': answers,
            'answer_nodes': [{'name': a} for a in answers],
            'gold_relations': relations,
            'answer_type': answer_type,
            'program_functions': [s['function'] for s in program],
        }

        # Add type-specific info
        query_info = extract_query_info(program)
        sample.update(query_info)

        type_counter[answer_type] += 1
        all_samples.append(sample)

        # Entity-only subset (backward compatible)
        if answer_type == 'entity' and entity_name and relations:
            entity_only.append(sample)

    # Save entity-only (backward compatible)
    out_entity = out_dir / 'val_entity.jsonl'
    with open(out_entity, 'w', encoding='utf-8') as f:
        for s in entity_only:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # Save all types
    out_all = out_dir / 'val_all.jsonl'
    with open(out_all, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\nEntity-only: {len(entity_only)} -> {out_entity}")
    print(f"All types:   {len(all_samples)} -> {out_all}")
    print(f"\nAnswer type distribution:")
    for t, c in type_counter.most_common():
        pct = 100 * c / len(all_samples)
        print(f"  {t}: {c} ({pct:.1f}%)")

    if skip_counter:
        print(f"\nSkipped:")
        for reason, c in skip_counter.most_common():
            print(f"  {reason}: {c}")


if __name__ == '__main__':
    main()
