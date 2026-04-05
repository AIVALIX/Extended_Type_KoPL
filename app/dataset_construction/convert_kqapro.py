"""
KQA Pro val.json → ETK用 JSONL 変換

ETK対応のmulti-hop entity retrieval質問のみ抽出。

Usage:
    python app/dataset_construction/convert_kqapro.py

Output:
    data/kqapro/qa/val_entity.jsonl
"""
import json
from pathlib import Path
from collections import Counter


def extract_entity_and_relations(program: list, kb_entities: dict) -> tuple:
    """KoPLプログラムからエンティティとリレーションを抽出。

    Returns:
        (entity_name, relations_list) or (None, None) if not extractable
    """
    entity_name = None
    relations = []

    for step in program:
        func = step.get('function', '')
        inputs = step.get('inputs', [])

        if func == 'Find' and inputs and entity_name is None:
            entity_name = inputs[0]
        elif func == 'Relate' and inputs:
            # Relate のinputsは [relation_name, direction]
            if len(inputs) >= 1:
                relations.append(inputs[0])
        elif func == 'FilterConcept' and inputs:
            # FilterConcept はtype制約（リレーションではない）
            pass

    return entity_name, relations


def is_entity_answer(program: list) -> bool:
    """プログラムの回答がエンティティ名かどうか判定"""
    if not program:
        return False
    last_func = program[-1].get('function', '')
    # エンティティ回答を返す関数
    return last_func in ('Find', 'Relate', 'FilterConcept', 'And', 'Or', 'What', 'QueryRelation')


def main():
    base = Path(__file__).resolve().parent.parent.parent / 'data' / 'kqapro'
    val_path = base / 'val.json'
    kb_path = base / 'kb.json'
    out_dir = base / 'qa'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'val_entity.jsonl'

    print(f"Loading {val_path} ...")
    with open(val_path, 'r', encoding='utf-8') as f:
        val = json.load(f)

    print(f"Loading {kb_path} for entity name lookup ...")
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    # entity ID → name マッピング
    id_to_name = {}
    for eid, ent in kb['entities'].items():
        id_to_name[eid] = ent['name']
    for cid, c in kb['concepts'].items():
        id_to_name[cid] = c['name']

    print(f"Total questions: {len(val)}")

    # フィルタ: ETK対応の質問のみ
    converted = []
    func_counter = Counter()
    hop_counter = Counter()

    for q in val:
        program = q.get('program', [])
        answer = q.get('answer', '')

        if not is_entity_answer(program):
            continue

        entity_name, relations = extract_entity_and_relations(program, kb['entities'])

        if not entity_name:
            continue

        # 回答がエンティティ名のリストか確認
        # KQA Pro のanswerは文字列（エンティティ名）
        answers = [answer] if isinstance(answer, str) else answer

        # リレーションが少なくとも1つ必要
        if not relations:
            continue

        hop_counter[len(relations)] += 1
        last_func = program[-1].get('function', '') if program else ''
        func_counter[last_func] += 1

        sample = {
            'question': q['question'],
            'entity': entity_name,
            'relation': relations[0] if len(relations) == 1 else relations[0],
            'answers': answers,
            'answer_nodes': [{'name': a} for a in answers],
            'gold_relations': relations,
            'program_functions': [s['function'] for s in program],
        }

        # multi-hop の場合は各リレーションを個別キーに
        for i, r in enumerate(relations):
            if i == 0:
                sample['relation'] = r
            else:
                sample[f'relation{i+1}'] = r

        converted.append(sample)

    # 保存
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in converted:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\nConverted: {len(converted)}/{len(val)} questions ({len(converted)/len(val)*100:.1f}%)")
    print(f"Written: {out_path}")

    print(f"\nHop distribution:")
    for h, c in sorted(hop_counter.items()):
        print(f"  {h}-hop: {c}")

    print(f"\nLast function distribution:")
    for func, c in func_counter.most_common(10):
        print(f"  {func}: {c}")

    # サンプル表示
    print(f"\nSample entries:")
    for s in converted[:5]:
        print(f"  Q: {s['question']}")
        print(f"  Entity: {s['entity']}, Relations: {s['gold_relations']}, Answer: {s['answers']}")
        print()


if __name__ == '__main__':
    main()
