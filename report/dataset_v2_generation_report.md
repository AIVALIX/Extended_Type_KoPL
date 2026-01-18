# Dataset v2 生成・評価レポート

**作成日時**: 2026-01-18
**作業内容**: スキーマベースのデータセット生成システム構築と評価

---

## 1. 背景

### 1.1 前回までの問題点

前回のレポート（`two_hop_chain_question_generation_improvement_report.md`）で以下の問題を特定:

- PrimeKGの`parent-child`リレーションが異なるノードタイプ間で使用されていた（45%が異タイプ間）
- two_hop_chainデータの35.6%が意味的に不整合なパスを含んでいた
- 質問生成の改善だけでは限界があり、データ生成ロジックの修正が必要

### 1.2 対応方針

1. PrimeKGデータを再インポート（`parent-child` → `parent_child`、同タイプ間のみ）
2. スキーマベースのデータ生成システムを構築
3. 意味のあるパステンプレートのみを使用

---

## 2. 実装内容

### 2.1 作成ファイル

| ファイル | 説明 |
|----------|------|
| `app/dataset_construction/schema_v2.py` | クリーンなスキーマ定義と質問テンプレート |
| `app/dataset_construction/generate_dataset_v2.py` | Answer-firstアプローチのデータ生成 |
| `app/pipeline/evaluate_v2_datasets.py` | v2データセット用評価スクリプト |
| `app/pipeline/debug_v2_pipeline.py` | v2データセット用デバッグスクリプト |

### 2.2 スキーマ定義 (schema_v2.py)

#### ノードタイプ（10種類）
```
anatomy, biological_process, cellular_component, disease,
drug, effect/phenotype, exposure, gene/protein,
molecular_function, pathway
```

#### クエリテンプレート数
| クエリタイプ | テンプレート数 |
|-------------|---------------|
| one_hop | 10 |
| two_hop | 9 |
| two_intersection | 7 |
| three_intersection | 3 |

#### two_hopテンプレート例
```python
{
    "name": "drug_target_gene_disease",
    "path": ("drug", "target", "gene/protein", "associated_disease", "disease"),
    "question_templates": [
        "What diseases are associated with genes targeted by {anchor}?",
        "Which diseases involve genes that {anchor} targets?",
    ],
}
```

### 2.3 データ生成 (generate_dataset_v2.py)

#### Answer-firstアプローチ
従来のAnchor-firstではなく、Answer-firstで逆引き生成:

```cypher
-- 2-hop例: Answer(x) -> Mid(z) -> Anchor(a) の順で逆引き
MATCH (x:{x_label})<-[r2:{rel2}]-(z:{z_label})<-[r1:{rel1}]-(a:{a_label})
WHERE a <> z AND z <> x AND a <> x
WITH a, z, collect(DISTINCT x)[0..{max_answers}] AS answers
WHERE size(answers) >= {min_answers} AND size(answers) <= {max_answers}
RETURN elementId(a) AS anchor_id, a.name AS anchor_name, ...
```

#### LLMパラフレーズ機能
- `--paraphrase`オプションでLLMによる質問パラフレーズを有効化
- 並列処理対応（`--max-workers`）
- テンプレートから自然な質問文を生成

### 2.4 生成されたデータセット

`result/dataset_v2/`に保存:

| ファイル | サンプル数 |
|----------|-----------|
| one_hop.jsonl | 1,000 |
| two_hop.jsonl | 800 |
| two_intersection.jsonl | 700 |
| three_intersection.jsonl | 100 |
| **合計** | **2,600** |

---

## 3. 評価結果

### 3.1 最終評価結果（各20サンプル）

| Dataset | Hits@1 | Hits@10 | PathAcc | MRR |
|---------|--------|---------|---------|-----|
| one_hop | 85.0% | 85.0% | 85.0% | 85.0% |
| two_hop | 40.0% | 70.0% | 90.0% | 46.9% |
| two_intersection | 100.0% | 100.0% | 70.0% | 100.0% |
| three_intersection | 100.0% | 100.0% | 95.0% | 100.0% |

### 3.2 分析

#### one_hop (85%)
- 良好な結果
- 単純なクエリのためType KoPLが正しく解釈できている

#### two_hop (40% Hits@1, 90% PathAcc)
- **PathAccは高い（90%）** → パス自体は正しく特定できている
- **Hits@1が低い（40%）** → 最終的な回答取得に問題

**原因調査（debug_v2_pipeline.pyによる）**:
```
Question: Which diseases are linked to genes targeted by Tacrolimus?
Gold Relations: ["target", "associated_disease"]

Generated Type KoPL:
  relation_name='targets'    ← 正しくは 'target'
  relation_name='related_to' ← 正しくは 'associated_disease'
  target_type='Gene'         ← 正しくは 'gene/protein'
```

Type KoPLが生成するリレーション名・タイプ名がスキーマと一致しない場合がある。

#### two_intersection / three_intersection (100%)
- 非常に良好
- 複数アンカーの交差クエリは正しく処理できている

---

## 4. 発見された課題

### 4.1 Type KoPLのリレーション名不一致

LLMが生成するType KoPLのリレーション名がスキーマと一致しない:

| LLM生成 | 正しいスキーマ |
|---------|---------------|
| `targets` | `target` |
| `related_to` | `associated_disease` |
| `Gene` | `gene/protein` |

### 4.2 対策案

1. **スキーマプロンプトの改善**: `--use-schema-prompt`オプションでスキーマ情報をプロンプトに追加
2. **リレーション名の正規化**: LLM出力を正規化するポストプロセス追加
3. **Few-shotの改善**: 正しいリレーション名を使用した例を追加

---

## 5. 使用方法

### 5.1 データセット生成

```bash
# パラフレーズなし（高速）
docker compose exec app python dataset_construction/generate_dataset_v2.py \
  --output-dir result/dataset_v2 \
  --num-samples 100

# パラフレーズあり（自然な質問）
docker compose exec app python dataset_construction/generate_dataset_v2.py \
  --output-dir result/dataset_v2 \
  --num-samples 100 \
  --paraphrase \
  --lang en
```

### 5.2 評価

```bash
# 全データセット評価
docker compose exec app python pipeline/evaluate_v2_datasets.py \
  --num-samples 50

# 特定データセットのみ
docker compose exec app python pipeline/evaluate_v2_datasets.py \
  --dataset one_hop two_hop \
  --num-samples 100
```

### 5.3 デバッグ

```bash
# 単一サンプル
docker compose exec app python pipeline/debug_v2_pipeline.py \
  --data result/dataset_v2/two_hop.jsonl --index 0

# 複数サンプル
docker compose exec app python pipeline/debug_v2_pipeline.py \
  --data result/dataset_v2/two_hop.jsonl --index 0 1 2 3 4

# サマリーのみ
docker compose exec app python pipeline/debug_v2_pipeline.py \
  --data result/dataset_v2/two_hop.jsonl --index 0 1 2 3 4 -q

# スキーマプロンプト有効
docker compose exec app python pipeline/debug_v2_pipeline.py \
  --data result/dataset_v2/two_hop.jsonl --index 0 --use-schema-prompt
```

---

## 6. 今後の改善案

### 6.1 短期的改善
1. Type KoPL生成プロンプトにスキーマ情報を追加
2. リレーション名のファジーマッチング実装
3. テンプレート数の増加

### 6.2 中期的改善
1. two_hop専用のType KoPL生成ロジック
2. 中間ノード情報の活用（質問に含める）
3. 評価メトリクスの細分化（パス正解率、リレーション正解率など）

---

## 7. 結論

- スキーマベースのデータ生成システムを構築完了
- one_hop、two_intersection、three_intersectionは良好な精度
- two_hopはPathAccは高いがHits@1が低い → Type KoPLのリレーション名生成が課題
- 評価・デバッグツールを整備し、問題の特定が容易に

---

## 8. 関連ファイル

- `app/dataset_construction/schema_v2.py` - スキーマ定義
- `app/dataset_construction/generate_dataset_v2.py` - データ生成
- `app/pipeline/evaluate_v2_datasets.py` - 評価スクリプト
- `app/pipeline/debug_v2_pipeline.py` - デバッグスクリプト
- `result/dataset_v2/` - 生成データセット
