# MetaQA Evaluation Report - February 2026

## Overview

MetaQA（映画ドメインのKGQAベンチマーク）における Extended Type-KoPL (ETK) と SAFE パイプラインの比較評価結果をまとめる。

## Evaluation Settings

- **Dataset**: MetaQA (1-hop, 2-hop, 3-hop)
- **Samples**: 100 per dataset (random sampling)
- **Few-shot Examples**: 3 examples (1-hop, 2-hop, 3-hop 各1つ)
- **LLM**: GPT-4o-mini (default)
- **Workers**: 16 (parallel execution)

### Few-shot Examples (ETK/SAFE共通)

```
1. 1-hop: "Who directed Titanic?"
2. 2-hop: "Who directed the movies that Tom Hanks starred in?"
3. 3-hop: "Who starred in the movies written by the writers of The Matrix?"
```

## Results

### 1. ETK vs SAFE Comparison (3 Few-shot Examples)

| Dataset | ETK Accuracy | SAFE Accuracy | ETK F1 | SAFE F1 |
|---------|-------------|---------------|--------|---------|
| 1-hop   | **100%**    | **100%**      | 100%   | 100%    |
| 2-hop   | **100%**    | 89%           | 100%   | 88.6%   |
| 3-hop   | **92%**     | 88%           | 91.8%  | 88.0%   |

**Key Findings**:
- ETKが全データセットでSAFEを上回る
- 特に2-hop (+11%) と 3-hop (+4%) で差が顕著
- 1-hopは両手法とも100%で同等

### 2. LLM Reranker Effect (ETK)

| Dataset | Without Reranker | With Reranker | Improvement |
|---------|-----------------|---------------|-------------|
| 1-hop   | 100%            | 100%          | 0%          |
| 2-hop   | 100%            | 100%          | 0%          |
| 3-hop   | 87%             | **90%**       | **+3%**     |

**Key Findings**:
- LLM Rerankerは3-hopでのみ効果あり (+3%)
- 1-hop/2-hopは既に高精度のため改善余地なし
- Rerankerの追加LLMコールのコストを考慮すると、3-hop以上の複雑なクエリでのみ使用推奨

### 3. PathAcc Analysis

| Dataset | ETK PathAcc | SAFE PathAcc | Note |
|---------|-------------|--------------|------|
| 1-hop   | 98%         | 96%          | High |
| 2-hop   | 56%         | 55%          | Low  |
| 3-hop   | 4%          | 2%           | Very Low |

**Note**: PathAccが低いのはMetaQAのgold relationsの品質問題による。例えば「directors」を問う質問に対してgold relationが「HAS_TAGS」となっているケースがある。Precision/Recallベースの評価がより信頼性が高い。

## Algorithm Comparison

### Extended Type-KoPL (ETK)
- **Approach**: LLMでType-KoPLプログラムを生成 → スキーマベースのパス探索
- **Strengths**:
  - 少数のfew-shot例で高い汎化性能
  - スキーマ制約による効率的な探索
- **Weaknesses**:
  - スキーマ定義が必要

### SAFE
- **Approach**: LLMで擬似クエリグラフ生成 → ADJによるスキーマ補正 → サブグラフマッチング
- **Strengths**:
  - 自然言語リレーションを使用（LLMが理解しやすい）
  - ADJによる誤り補正機能
- **Weaknesses**:
  - 多様なfew-shot例が必要
  - 2-hop以上でファンアウトが発生しやすい

## Conclusions

1. **MetaQAではETKがSAFEを上回る**
   - シンプルなスキーマ（映画ドメイン）でETKの優位性が発揮
   - 3つのfew-shot例で十分な汎化

2. **LLM Rerankerは複雑なクエリで有効**
   - 3-hopで+3%改善
   - 単純なクエリでは不要

3. **Few-shot設計の重要性**
   - 3-hopの例に`WRITTEN_BY`パターンを含めることで3-hop精度が向上
   - ドメイン固有のパターンをカバーする例の選択が重要

## Recommendations

- MetaQAのような単純なドメインでは**ETK推奨**
- 3-hop以上のクエリが多い場合は**LLM Reranker併用**を検討
- Few-shot例は各ホップ数で**異なるリレーションパターン**を含めるべき
