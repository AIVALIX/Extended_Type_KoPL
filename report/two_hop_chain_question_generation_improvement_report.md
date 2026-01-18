# Two-Hop Chain 質問生成改善レポート

**作成日時**: 2026-01-18 05:57
**作業期間**: 2026-01-18 03:00 - 05:30
**対象ファイル**: `app/dataset_construction/generate_questions.py`

---

## 1. 背景と問題

### 1.1 初期状態の評価結果

| Dataset | Hits@1 | Hits@5 | Hits@10 | MRR | PathAcc |
|---------|--------|--------|---------|-----|---------|
| two_hop_chain | 26% | - | - | - | 87% |
| two_anchor_intersection | 75% | - | - | - | 93% |
| three_anchor_intersection | 86% | - | - | - | 99% |

**問題点**: two_hop_chainのHits@1が他と比較して著しく低い

### 1.2 原因分析

two_hop_chainの質問形式を調査した結果、以下の問題を発見:

- **質問例**: "What are subtypes of exit from reproductive diapause?"
- **問題**: 中間ノード（intermediate entity）が質問に含まれていない
- **データ構造**:
  ```
  anchor: "exit from reproductive diapause"
  mid_nodes_sample: [{"name": "exit from diapause", ...}]
  answer: ["dauer exit", "dormancy process"]
  ```
- **期待される質問**: anchorとmid（中間ノード）の両方を含む質問

---

## 2. 試行内容

### 2.1 試行1: 両エンティティを含む質問生成

**日時**: 2026-01-18 03:30
**変更内容**: `_pattern_description`関数を修正し、中間ノード情報をプロンプトに追加

```python
# 変更前
"anchor_name: {row.get('anchor_name')}"

# 変更後
"Entity 1 (anchor): {anchor_name}"
"Entity 2 (intermediate): {mid_name}"
"STRICT REQUIREMENT: The question MUST explicitly name BOTH entities"
```

**結果**:
- 質問に両エンティティは含まれるようになった
- しかし「How does X relate to Y?」形式の質問が生成された
- **問題**: Type KoPLがA→Bの1-hopクエリとして解釈し、回答がBになってしまう

**生成例**:
```
Q: "How does exit from reproductive diapause relate to exit from diapause?"
予測回答: "exit from diapause" (中間ノード)
正解回答: "dauer exit", "dormancy process" (最終ノード)
```

### 2.2 試行2: バリデーションによる不適切な質問のリジェクト

**日時**: 2026-01-18 04:00
**変更内容**: `_validate_two_hop_question`メソッドを追加

```python
def _validate_two_hop_question(self, question: str, row: Dict[str, Any]) -> bool:
    # 両エンティティを含むかチェック
    anchor_in_q = anchor_name.lower() in q_lower
    mid_in_q = mid_name.lower() in q_lower

    if not (anchor_in_q and mid_in_q):
        return False

    # A-B関係を聞く質問はリジェクト
    bad_patterns = [
        "how does", "how is", "how are",
        "relate to", "related to", "relationship between",
        "connection between", "link between",
        "what role", "what is the role"
    ]
    for pattern in bad_patterns:
        if pattern in q_lower:
            return False

    return True
```

**リトライ機構**:
- バリデーション失敗時は最大3回リトライ
- リトライ時はより強い指示を含むプロンプトを使用

### 2.3 試行3: プロンプトの明確化

**日時**: 2026-01-18 04:30
**変更内容**: プロンプトで推奨フォーマットを明示

```python
return (
    "Pattern: TWO-HOP CHAIN\n\n"
    "SIMPLE QUESTION FORMAT REQUIRED:\n"
    f"  'What are the {question_verb} of {mid_name}, a {rel1} of {anchor_name}?'\n\n"
    "MUST INCLUDE:\n"
    f"  1. \"{anchor_name}\" (the starting point)\n"
    f"  2. \"{mid_name}\" (the intermediate entity)\n"
    f"  3. Ask for {ans_types_text} (the answers we want)\n\n"
    ...
    "DO NOT ASK:\n"
    f"  - 'How does X relate to Y?' or 'What is the relationship between X and Y?'\n"
    f"  - 'What role does X play in Y?' or 'What is the significance of X?'"
)
```

---

## 3. 評価結果

### 3.1 各試行後の結果比較

| 試行 | Hits@1 | PathAcc | 備考 |
|------|--------|---------|------|
| 初期状態 | 26% | 87% | 単純な質問形式 |
| 試行1後 | 25% | 60% | "relate to"形式 |
| 試行2+3後 | 35% | 35% | 改善版プロンプト |

### 3.2 詳細分析（試行2+3後、20サンプル）

**成功サンプル (7/20)**:
- 質問形式: "Which X are subcategories of Y, which is a subcategory of Z?"
- Type KoPLが正しく2-hopとして解釈
- 正しいパス `['parent-child', 'parent-child']` を発見

**失敗サンプル (13/20)**:
- 3回のリトライ後も適切な質問が生成できず
- 主な原因: データ自体の意味的不整合

### 3.3 成功・失敗サンプル例

**成功例**:
```
Q: "Which biological processes are subclasses of exit from diapause,
    itself a subclass of exit from reproductive diapause?"
Gold: ['parent-child', 'parent-child']
Pred: ['parent-child', 'parent-child']
Path Match: True, Hits@10: True
```

**失敗例（データ品質問題）**:
```
Q: "What is the role of disulfide oxidoreductase activity in the
    development of pedal digit 3 metatarsal endochondral element?"
Gold: ['parent-child', 'parent-child']
Pred: None
Path Match: False, Hits@10: False

原因: anchor=Anatomy, mid=MolecularFunction という意味的に不整合なパス
```

---

## 4. 発見されたデータ品質問題

### 4.1 意味的に不整合なパス

two_hop_chainデータに以下のような意味的に不整合なパスが含まれている:

| anchor_type | mid_type | relation | 問題 |
|-------------|----------|----------|------|
| Anatomy | MolecularFunction | parent-child | 解剖学構造が分子機能の親になることはない |
| MolecularFunction | Disease | parent-child | 分子機能が疾患の親になることはない |

**例**:
```json
{
  "anchor_name": "pedal digit 3 metatarsal endochondral element",
  "anchor_types": ["anatomy"],
  "mid_nodes_sample": [{"name": "disulfide oxidoreductase activity", "types": ["molecular_function"]}],
  "rel1": "parent-child"
}
```

### 4.2 影響

- LLMが意味的に整合する質問を生成できない
- バリデーションを通過する質問が生成されても、Type KoPLが正しく解釈できない
- データ生成段階でのフィルタリングが必要

---

## 5. 変更されたファイル

### 5.1 generate_questions.py

**変更箇所**:
1. `_pattern_description`関数: two_hop_chain用プロンプトの大幅改修
2. `_validate_two_hop_question`メソッド: 新規追加
3. `generate`メソッド: リトライ機構追加

**コード差分（主要部分）**:
```python
# _validate_two_hop_question (新規追加)
def _validate_two_hop_question(self, question: str, row: Dict[str, Any]) -> bool:
    is_two_hop = ("rel1" in row and "rel2" in row and
                  row.get("mid_nodes_sample") and len(row.get("mid_nodes_sample", [])) > 0)
    if not is_two_hop:
        return True
    # ... バリデーションロジック

# generate メソッド（リトライ追加）
max_retries = 3
for attempt in range(max_retries):
    res = self._model.invoke(prompt)
    question = res.question.strip()
    if self._validate_two_hop_question(question, row):
        return question
    # リトライプロンプト生成
```

### 5.2 evaluate_all_datasets.py

**変更箇所**: `--dataset`オプション追加

```python
p.add_argument("--dataset", type=str, nargs="+",
               choices=["two_hop_chain", "two_anchor_intersection", "three_anchor_intersection"],
               help="Specific dataset(s) to evaluate (default: all)")
```

---

## 6. 生成されたデータセット

| ファイル | 日時 | 件数 | 説明 |
|----------|------|------|------|
| `two_hop_chain_train_with_questions_en.jsonl` | 05:07 | 100件 | 最新版（改善プロンプト） |
| `two_hop_chain_train_with_questions_en_old.jsonl` | 03:01 | 100件 | 旧版（バックアップ） |

---

## 7. 結論と今後の課題

### 7.1 結論

- **Hits@1の改善**: 25% → 35% (+10%)
- **PathAccの悪化**: 60% → 35% (-25%)
- **トレードオフ**: 質問の複雑化によりType KoPL生成の難易度が上昇

### 7.2 根本的な問題

1. **データ品質**: two_hop_chainデータに意味的に不整合なパスが多数含まれる
2. **質問形式のジレンマ**:
   - 単純な質問 → PathAcc高いがHits@1低い
   - 複雑な質問 → 両エンティティ含むがType KoPL解釈が困難

### 7.3 今後の改善案

1. **データ生成段階でのフィルタリング**: 意味的に整合するパスのみを抽出
2. **Type KoPL生成の改善**: 複雑な質問にも対応できるようプロンプト改善
3. **質問形式の最適化**: PathAccとHits@1の両方を向上させる質問形式の探索

---

## 8. 実行コマンド

```bash
# 質問生成（改善版）
docker compose exec app python dataset_construction/generate_questions.py \
  --in result/dataset_splits_v2/two_hop_chain/train.jsonl \
  --out result/dataset_construction_v2/two_hop_chain_train_with_questions_en.jsonl \
  --lang en --max-rows 100 --parallel

# 評価（two_hop_chainのみ）
docker compose exec app python pipeline/evaluate_all_datasets.py \
  --dataset two_hop_chain --num-samples 20 --lang en

# デバッグ
docker compose exec app python pipeline/debug_pipeline.py \
  --data result/dataset_construction_v2/two_hop_chain_train_with_questions_en.jsonl \
  --type two_hop_chain --index 0
```

---

## 9. 追加調査: PrimeKGのデータ品質問題

**調査日時**: 2026-01-18 06:00

### 9.1 parent-childリレーションの分析

PrimeKGの`parent-child`リレーションが異なるノードタイプ間でも使用されていることが判明。

```
=== parent-child リレーションのノードタイプ組み合わせ ===
biological_process        <--> biological_process       :  117,608 (正常)
disease                   <--> disease                  :   61,660 (正常)
gene/protein              <--> gene/protein             :   53,656 (正常)
gene/protein              <--> disease                  :   29,104 (問題)
disease                   <--> molecular_function       :    7,544 (問題)
anatomy                   <--> disease                  :    3,658 (問題)
...
```

**統計**:
- 同タイプ間: 311,532件 (55%)
- 異タイプ間: 251,956件 (45%)

### 9.2 two_hop_chainデータへの影響

500サンプルを分析した結果：

| 分類 | 件数 | 割合 |
|------|------|------|
| 同タイプ間 (正常) | 322 | 64.4% |
| 異タイプ間 (問題) | 178 | **35.6%** |

**問題のあるパスの例**:
```
anatomy -[parent-child]-> molecular_function -[parent-child]-> disease
molecular_function -[parent-child]-> disease -[parent-child]-> gene/protein
effect/phenotype -[parent-child]-> effect/phenotype -[parent-child]-> gene/protein
```

### 9.3 問題の原因

`Cypher_query.py`の`build_two_hop_chain_cypher`関数にノードタイプの制約がない:

```cypher
// 問題のあるクエリ（現状）
MATCH (a)-[r1]-(z)-[r2]-(x)
WHERE a <> z AND z <> x AND a <> x
// ノードタイプの制約がない！
```

### 9.4 推奨される修正

**Option 1: Cypherクエリでフィルタリング**

```cypher
// parent-childの場合は同タイプのみに制限
MATCH (a)-[r1]-(z)-[r2]-(x)
WHERE a <> z AND z <> x AND a <> x
  AND (type(r1) <> 'parent-child' OR labels(a)[0] = labels(z)[0])
  AND (type(r2) <> 'parent-child' OR labels(z)[0] = labels(x)[0])
```

**Option 2: 後処理でフィルタリング**

```python
# データ生成後にフィルタリング
def is_valid_path(sample):
    if sample['rel1'] == 'parent-child':
        if sample['anchor_types'][0] != sample['mid_nodes_sample'][0]['types'][0]:
            return False
    if sample['rel2'] == 'parent-child':
        if sample['mid_nodes_sample'][0]['types'][0] != sample['answer_nodes_sample'][0]['types'][0]:
            return False
    return True
```

### 9.5 結論

**two_hop_chainの品質問題の約36%はデータ生成段階に起因する**。質問生成プロンプトの改善だけでは限界があり、根本的な解決にはデータ生成ロジックの修正が必要。
