# PrimeKGQA Evaluation Report - February 2026

## Overview

PrimeKGQA（バイオメディカルドメインのKGQAベンチマーク）における Extended Type-KoPL (ETK) パイプラインの評価結果をまとめる。特に、LLM Rerankerの効果とRELATION GUIDEの効果について分析。

## Evaluation Settings

- **Dataset**: PrimeKGQA (one_hop, two_hop)
- **Samples**: 100 per dataset (random sampling)
- **Few-shot Examples**: 3 examples (1-hop, 2-hop, intersection 各1つ)
- **LLM**: GPT-4o-mini (default)
- **Workers**: 16 (parallel execution)

### Few-shot Examples (ETK)

```
1. 1-hop: "What diseases is Metformin indicated for?"
2. 2-hop: "What phenotypes are present in diseases treated by Aspirin?"
3. Intersection: "What genes are targeted by both Aspirin and Ibuprofen?"
```

## Results

### 1. Baseline ETK Performance (3 Few-shot, No Reranker)

| Dataset | Accuracy | F1 | PathAcc |
|---------|----------|-----|---------|
| one_hop | 75% | 74.8% | 73% |
| two_hop | 88% | 87.5% | 56% |

**Key Findings**:
- 2-hopの方が1-hopより精度が高いという逆転現象
- 原因: 類似リレーションの混同（indication↔off-label use, phenotype present↔phenotype absent）
- 1-hopは単一のリレーション選択が必要なため、混同の影響が直接精度に反映

### 2. RELATION GUIDE Effect

RELATION GUIDEは、類似したリレーションを区別するための明示的な説明テキスト：

```
IMPORTANT RELATION GUIDE:
- "indication": diseases the drug IS APPROVED to treat
- "off-label use": diseases the drug is used for but NOT officially approved
- "phenotype present": symptoms that ARE observed in the disease
- "phenotype absent": symptoms that are NOT observed in the disease
```

| Dataset | Without Guide | With Guide | Improvement |
|---------|--------------|------------|-------------|
| one_hop | 75% | **90%** | **+15%** |
| two_hop | 88% | 89% | +1% |

**Key Findings**:
- RELATION GUIDEは1-hopで大幅改善（+15%）
- 2-hopでは改善効果は限定的
- 類似リレーションの明示的説明がLLMの選択精度を向上

### 3. LLM Reranker Effect

LLM Rerankerは、候補パスをLLMで再評価して最適なパスを選択：

| Dataset | Without Reranker | With Reranker | Improvement |
|---------|-----------------|---------------|-------------|
| one_hop | 75% | **96%** | **+21%** |
| two_hop | 88% | **99%** | **+11%** |

| Dataset | PathAcc (No Reranker) | PathAcc (With Reranker) |
|---------|----------------------|-------------------------|
| one_hop | 73% | **100%** |
| two_hop | 56% | **99%** |

**Key Findings**:
- LLM Rerankerは両データセットで劇的な改善
- 特に1-hopで+21%、2-hopで+11%
- PathAccも大幅改善（1-hop: 73%→100%, 2-hop: 56%→99%）
- Rerankerがリレーション選択の誤りを効果的に補正

### 4. Configuration Comparison Summary

| Configuration | one_hop | two_hop | Notes |
|--------------|---------|---------|-------|
| Baseline (3 few-shot) | 75% | 88% | 1-hop < 2-hop 逆転 |
| + RELATION GUIDE | 90% | 89% | 1-hop改善 |
| + LLM Reranker | **96%** | **99%** | 最高性能 |

## Algorithm Analysis

### Why 1-hop < 2-hop Without Reranker?

PrimeKGQAでは、類似したリレーションペアが存在：
- `indication` vs `off-label use`（承認済み適応症 vs 適応外使用）
- `phenotype present` vs `phenotype absent`（存在する症状 vs 存在しない症状）
- `contraindication` vs `indication`（禁忌 vs 適応症）

1-hopクエリでは：
- 単一のリレーション選択が必要
- 類似リレーションの混同が直接エラーに

2-hopクエリでは：
- 複数のリレーションの組み合わせ
- 一部の混同があっても最終回答に影響しにくい

### Why LLM Reranker Works So Well?

1. **候補パスの文脈理解**: Rerankerは質問と候補パス全体を見て判断
2. **類似リレーションの区別**: 文脈から適切なリレーションを選択可能
3. **エラー補正**: 初期選択の誤りを効果的に修正

## Comparison with MetaQA

| Aspect | PrimeKGQA | MetaQA |
|--------|-----------|--------|
| Domain | Biomedical | Movie |
| Schema Complexity | High (10 node types, 30+ relations) | Low (3 node types, 9 relations) |
| Similar Relations | Many pairs | Few |
| Reranker Effect | **Dramatic** (+21%, +11%) | Moderate (+3% on 3-hop only) |
| Baseline Performance | Lower (75-88%) | Higher (92-100%) |

**Key Insight**: ドメインの複雑さとリレーションの類似性がRerankerの効果に影響

## Conclusions

1. **PrimeKGQAではLLM Rerankerが必須**
   - ベースライン75%→Reranker96%（1-hop）
   - 複雑なスキーマでの類似リレーション区別に効果的

2. **RELATION GUIDEも有効だがRerankerほどではない**
   - 1-hopで+15%改善
   - Rerankerとの併用は未検証（Reranker単体で十分な精度）

3. **1-hop < 2-hopの逆転現象は類似リレーションが原因**
   - バイオメディカルドメイン特有の問題
   - Rerankerで解消可能

## Recommendations

- PrimeKGQAでは**LLM Reranker必須**
- 類似リレーションが多いドメインでは、Rerankerのコストは正当化される
- ベースライン精度が低い場合、few-shot例やRELATION GUIDEより**Rerankerが効果的**
- 本番運用では、1-hopクエリでも2-hopクエリでもRerankerを有効化すべき

## Future Work

1. **Rerankerのコスト最適化**: 候補数の削減、小型LLMの使用
2. **RELATION GUIDE + Rerankerの併用効果**: 相乗効果があるか検証
3. **intersection/three_intersectionでの評価**: より複雑なクエリでの検証
4. **他のバイオメディカルKGでの検証**: DrugBank、UMLS等での汎用性確認
