# パイプライン精度評価レポート

**作成日時**: 2026-01-19
**評価対象**: Extended Type-KoPL, SAFE, KGT パイプライン
**サンプル数**: 各データセット 20件

---

## 1. 概要

### 1.1 評価目的

3つのグラフQAパイプライン（Extended Type-KoPL, SAFE, KGT）の精度を、複数のデータセット形式で比較評価する。

### 1.2 評価対象パイプライン

| パイプライン | 特徴 |
|-------------|------|
| **Extended Type-KoPL** | LLMによるKoPL生成 + タイプ制約付きグラフ探索 |
| **SAFE** | ADJ (隣接探索) + APSP (全点間最短経路) によるサブグラフマッチング |
| **KGT** | スキーマベースパス探索 + ベクトル類似度ランキング |

### 1.3 評価指標

| 指標 | 説明 |
|------|------|
| **Accuracy (Hits@1)** | 正解エンティティが予測結果に含まれる割合 |
| **Recall** | 正解エンティティのうち、予測できた割合 |
| **Precision** | 予測エンティティのうち、正解だった割合 |
| **F1** | Precision と Recall の調和平均 |

### 1.4 データセット形式

| 形式 | 説明 | 例 |
|------|------|-----|
| **one_hop** | 1つのリレーションで回答に到達 | "Aspirinが標的とする遺伝子は？" |
| **two_hop** | 2つのリレーションを経由して回答に到達 | "Aspirinの標的遺伝子が関連する疾患は？" |
| **two_intersection** | 2つの条件の積集合 | "AとBの両方に関連するCは？" |
| **three_intersection** | 3つの条件の積集合 | "A、B、Cの全てに関連するDは？" |

---

## 2. データセット別評価結果

### 2.1 One-Hop (1ホップ)

単一のリレーションで回答に到達する最も基本的なクエリ形式。

| Pipeline | Accuracy | Recall | Precision | F1 |
|----------|----------|--------|-----------|-----|
| Extended Type-KoPL | **100.0%** | 100.0% | 100.0% | 100.0% |
| SAFE | **100.0%** | 96.9% | 100.0% | 98.0% |
| KGT | **100.0%** | 85.2% | 100.0% | 88.6% |

**所見**:
- 全パイプラインで Accuracy 100% を達成
- Extended Type-KoPL が全指標で最高スコア
- KGT は Recall がやや低い（85.2%）が、Precision は 100%

---

### 2.2 Two-Hop (2ホップ)

2つのリレーションを経由するチェーン型クエリ。中間ノードを正しく特定する必要がある。

| Pipeline | Accuracy | Recall | Precision | F1 |
|----------|----------|--------|-----------|-----|
| Extended Type-KoPL | **70.0%** | 66.8% | 32.6% | 38.0% |
| SAFE | **70.0%** | 37.7% | 50.1% | 35.5% |
| KGT | **70.0%** | 47.7% | 46.4% | 40.4% |

**所見**:
- 全パイプラインで Accuracy 70%（1-hop より難易度が高い）
- Extended Type-KoPL は Recall が最も高いが、Precision が低い（多くの候補を返す傾向）
- KGT が F1 スコアで最も高い（40.4%）

---

### 2.3 Two-Intersection (2条件積集合)

2つの条件を同時に満たすエンティティを検索するクエリ。

| Pipeline | Accuracy | Recall | Precision | F1 |
|----------|----------|--------|-----------|-----|
| Extended Type-KoPL | **100.0%** | 100.0% | 100.0% | 100.0% |
| SAFE | - | - | - | - |
| KGT | 85.0% | 85.0% | 73.1% | 76.3% |

**所見**:
- SAFE は intersection クエリ未対応
- Extended Type-KoPL が全指標で 100% を達成
- KGT は Accuracy 85%、Precision がやや低い（73.1%）

---

### 2.4 Three-Intersection (3条件積集合)

3つの条件を同時に満たすエンティティを検索する最も複雑なクエリ形式。

| Pipeline | Accuracy | Recall | Precision | F1 |
|----------|----------|--------|-----------|-----|
| Extended Type-KoPL | **100.0%** | 100.0% | 100.0% | 100.0% |
| SAFE | - | - | - | - |
| KGT | 85.0% | 85.0% | 56.3% | 63.2% |

**所見**:
- SAFE は intersection クエリ未対応
- Extended Type-KoPL が全指標で 100% を達成
- KGT は条件が増えるほど Precision が低下（56.3%）

---

## 3. パイプライン総合評価

### 3.1 全データセット平均

| Pipeline | Accuracy | Recall | Precision | F1 | 対応データセット |
|----------|----------|--------|-----------|-----|-----------------|
| Extended Type-KoPL | **92.5%** | 91.7% | 83.2% | 84.5% | 全4種類 |
| SAFE | 85.0% | 67.3% | 75.1% | 66.8% | one_hop, two_hop |
| KGT | 85.0% | 75.7% | 69.0% | 67.1% | 全4種類 |

### 3.2 データセット形式別の最高精度パイプライン

| データセット | 最高精度パイプライン | Accuracy |
|-------------|---------------------|----------|
| one_hop | 全パイプライン同率 | 100.0% |
| two_hop | 全パイプライン同率 | 70.0% |
| two_intersection | Extended Type-KoPL | 100.0% |
| three_intersection | Extended Type-KoPL | 100.0% |

---

## 4. パイプライン特性分析

### 4.1 Extended Type-KoPL

**強み**:
- intersection クエリで圧倒的な精度（100%）
- 全体的に高い Recall（正解を逃さない）

**弱み**:
- two_hop での Precision が低い（32.6%）
- 多くの候補を返す傾向があり、ノイズが含まれる

**適用シーン**: 正解を確実に含めたい場合、intersection クエリ

### 4.2 SAFE

**強み**:
- 実装がシンプル（ADJ + APSP ベース）
- one_hop, two_hop で安定した性能

**弱み**:
- intersection クエリ未対応
- two_hop での Recall が低い（37.7%）

**適用シーン**: シンプルなチェーン型クエリ（one_hop, two_hop）

### 4.3 KGT

**強み**:
- 全データセット形式に対応
- バランスの取れた Precision/Recall

**弱み**:
- intersection クエリでの Precision 低下
- one_hop での Recall がやや低い（85.2%）

**適用シーン**: 汎用的な用途、精度とカバレッジのバランスが重要な場合

---

## 5. 技術的修正事項

### 5.1 有向グラフ検索への統一

評価中に発見された問題と修正:

| 問題 | 原因 | 修正内容 |
|------|------|----------|
| SAFE: PathAcc 100% だが Accuracy 70% | Cypher が無向検索 `-[r]-` | 有向検索 `-[r]->` に変更 |
| KGT: 有向検索後に Accuracy 10% に低下 | KGSchema が無向グラフ | 有向グラフに変更 |

### 5.2 修正ファイル一覧

| ファイル | 修正内容 |
|----------|----------|
| `safe_pipeline.py` | `_intersection_search`, `_multi_hop_search`, `_dfs_search` を有向検索に変更 |
| `kgt_pipeline.py` | `_build_fallback_cypher` を有向検索に変更、`KGSchema.add_edge` を有向グラフに変更 |
| `extended_type_kopl_pipeline.py` | `_retrieve_entities_for_path` を有向検索に変更 |
| `type_kopl_pipeline.py` | 複数メソッドを有向検索に変更 |

---

## 6. 結論と推奨事項

### 6.1 結論

1. **Extended Type-KoPL** が総合的に最も高い精度（Accuracy 92.5%）
2. **intersection クエリ** では Extended Type-KoPL が唯一 100% を達成
3. **one_hop クエリ** は全パイプラインで 100% 達成（基本性能は同等）
4. **two_hop クエリ** は全パイプラインで 70%（改善の余地あり）

### 6.2 推奨事項

| ユースケース | 推奨パイプライン |
|-------------|-----------------|
| 汎用的なグラフQA | Extended Type-KoPL |
| シンプルなクエリのみ | SAFE（実装がシンプル） |
| Precision 重視 | KGT（one_hop, two_hop） |

### 6.3 今後の改善方針

1. **two_hop 精度向上**: 中間ノードの絞り込みロジック改善
2. **SAFE の intersection 対応**: ADJ/APSP ロジックの拡張
3. **KGT の intersection 精度向上**: パスランキングアルゴリズムの改善

---

## 7. 付録: 評価実行方法

```bash
# Docker コンテナ内で実行
docker exec python-primekgqa-experiment python pipeline/run_evaluation.py --all --num-samples 20

# 特定のパイプラインのみ
docker exec python-primekgqa-experiment python pipeline/run_evaluation.py --pipeline kgt --num-samples 20

# 特定のデータセットのみ
docker exec python-primekgqa-experiment python pipeline/run_evaluation.py --pipeline safe --dataset one_hop --num-samples 20
```
