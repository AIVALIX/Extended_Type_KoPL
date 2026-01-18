# KGT・SAFE Pipeline 実装・評価レポート

**作成日時**: 2026-01-18
**作業内容**: KGTおよびSAFEアルゴリズムの実装と評価

---

## 1. 背景

### 1.1 前回までの状況

`dataset_v2_generation_report.md`で報告した通り、Type KoPLベースのパイプラインでは以下の結果が得られていた：

| Dataset | Hits@1 | PathAcc |
|---------|--------|---------|
| one_hop | 85.0% | 85.0% |
| two_hop | 40.0% | 90.0% |
| two_intersection | 100.0% | 70.0% |
| three_intersection | 100.0% | 95.0% |

**課題**: two_hopでPathAccは高い(90%)がHits@1が低い(40%) → LLMが生成するリレーション名がスキーマと一致しない

### 1.2 対応方針

論文ベースの2つのアルゴリズムを実装し、比較評価を行う：

1. **KGT (Knowledge Graph Transformer)**: スキーマベースのパス探索 + ベクトル類似度
2. **SAFE (Semantic-Aware Subgraph Retrieval Framework)**: ADJ + APSP によるサブグラフマッチング

---

## 2. KGT Pipeline 実装

### 2.1 アルゴリズム概要

KGTは4つのフェーズで構成される：

```
Question → [Phase 1: Question Analysis]
        → [Phase 2: Schema-Based Path Finding]
        → [Phase 3: Cypher Generation]
        → [Phase 4: Subgraph Retrieval & Answer]
```

### 2.2 Phase 1: Question Analysis

LLMを使用して質問から以下を抽出：
- Head Entity（起点エンティティ）
- Head Entity Type（起点のノードタイプ）
- Tail Entity Type（回答のノードタイプ）

```python
class QuestionAnalysis(BaseModel):
    head_entity_name: str
    head_entity_type: str
    tail_entity_type: str
```

### 2.3 Phase 2: Schema-Based Path Finding

**スキーマグラフ構築**:
- `schema_v2.py`のSCHEMA_GRAPHからノードタイプとリレーションを読み込み
- 無向グラフとして構築（双方向探索可能）

**パス探索（DFS）**:
```python
def find_all_paths(self, start_type: str, end_type: str, max_depth: int = 3) -> List[SchemaPath]:
    """DFSで全てのパスを探索（最短パスのみではなく全パス）"""
```

当初BFSを使用していたが、最短パス（1-hop）しか見つからない問題があったためDFSに変更。

**ベクトル類似度によるランキング**:
- 各スキーマパスをテキスト化（例: "drug → target → gene/protein → associated_disease → disease"）
- OpenAI Embeddingsでベクトル化
- 質問とのコサイン類似度でランキング

### 2.4 Phase 3: Cypher Generation

最適パスからCypherクエリを生成：

```python
def _build_fallback_cypher(self, analysis: QuestionAnalysis, path: SchemaPath) -> str:
    # 無向マッチング（-[r:relation]-）を使用
    # PrimeKGのリレーションは方向が不定なため
```

**重要な修正点**: 当初`-[r:relation]->`（有向）を使用していたが、結果が0件になる問題があったため`-[r:relation]-`（無向）に変更。

### 2.5 Phase 4: Subgraph Retrieval

Cypherクエリを実行し、結果からエンティティを抽出。

### 2.6 実装ファイル

| ファイル | 説明 |
|----------|------|
| `app/pipeline/kgt_pipeline.py` | KGTパイプライン本体 |
| `app/pipeline/evaluate_kgt_pipeline.py` | 評価スクリプト |
| `app/pipeline/debug_kgt_pipeline.py` | デバッグスクリプト |

---

## 3. SAFE Pipeline 実装

### 3.1 アルゴリズム概要

SAFEは2つのフェーズで構成される：

```
Question → [Phase 1: ADJ (Approximate Distance Join)]
        → [Phase 2: Ranked Semantic Subgraph Matching]
```

### 3.2 Phase 1: ADJ (Approximate Distance Join)

**Step 1: 擬似クエリグラフ生成**

LLMを使用して質問から擬似エッジを生成：

```python
class PseudoEdge:
    src_node: str      # ソースノード名（"?"は未知）
    src_type: str      # ソースノードタイプ
    relation: str      # リレーション（自然言語）
    tgt_node: str      # ターゲットノード名
    tgt_type: str      # ターゲットノードタイプ
    is_anchor: bool    # 既知エンティティか
```

**Step 2: 候補スキーマエッジ選定**

各擬似エッジに対して、ベクトル類似度で上位k個のスキーマエッジを選定。

**Step 3: Floyd-Warshall APSP**

スキーマグラフ上で全点対最短経路（All-Pairs Shortest Path）を計算：

```python
def build_apsp(self):
    """Floyd-Warshallで全点対最短経路を計算"""
    # O(n^3) だがノードタイプ数が少ないため問題なし
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
```

**Step 4: エッジ距離による結合**

隣接するスキーマエッジ間の距離が閾値δ以下の組み合わせのみを候補クエリグラフとして選定。

### 3.3 Phase 2: Ranked Semantic Subgraph Matching

**1-hopクエリ**: 単純なCypher検索

**2-hopクエリ（チェーンパターン）**: 2ホップパスを単一Cypherで検索

```python
cypher = f"""
MATCH (a:{src_type})-[r1:{rel1}]-(mid:{mid_type})-[r2:{rel2}]-(ans:{tgt_type})
WHERE a.name = $anchor_name AND a <> mid AND mid <> ans AND a <> ans
RETURN DISTINCT a.name, mid.name, ans.name
"""
```

### 3.4 制限事項

**純粋なSAFEはIntersectionクエリに非対応**:
- SAFEは単一アンカーからのサブグラフマッチングを想定
- 複数アンカーの交差（AND）クエリには対応していない

### 3.5 実装ファイル

| ファイル | 説明 |
|----------|------|
| `app/pipeline/safe_pipeline.py` | SAFEパイプライン本体 |
| `app/pipeline/evaluate_safe_pipeline.py` | 評価スクリプト |
| `app/pipeline/debug_safe_pipeline.py` | デバッグスクリプト |

---

## 4. 評価結果

### 4.1 最終比較（Hits@1、各20サンプル）

| Dataset | Type KoPL | KGT | SAFE |
|---------|-----------|-----|------|
| one_hop | 85.0% | **100.0%** | 95.0% |
| two_hop | 40.0% | 40.0% | **50.0%** |
| two_intersection | **100.0%** | 90.0% | 20.0% |
| three_intersection | **100.0%** | 55.0% | 50.0% |

### 4.2 各パイプラインの特性

| 特性 | Type KoPL | KGT | SAFE |
|------|-----------|-----|------|
| **強み** | Intersection | one_hop | one_hop, two_hop |
| **弱み** | two_hop | Intersection | Intersection |
| **アプローチ** | LLM→KoPL→Cypher | Schema BFS+Vector | ADJ+APSP |
| **LLM依存度** | 高（KoPL生成） | 中（分析のみ） | 中（擬似グラフ生成） |

### 4.3 分析

**one_hop**:
- KGT (100%) > SAFE (95%) > Type KoPL (85%)
- スキーマベースのパス選択がLLMのリレーション名生成より安定

**two_hop**:
- SAFE (50%) > KGT = Type KoPL (40%)
- チェーンパターンではADJによる補正が効果的

**Intersection (two/three)**:
- Type KoPL (100%) >> KGT (90%/55%) >> SAFE (20%/50%)
- Type KoPLはKoPL言語でAND条件を自然に表現可能
- KGT/SAFEは単一パス探索のため、複数アンカーの交差に弱い

---

## 5. 実装上の課題と解決

### 5.1 KGT: BFSが最短パスのみ返す問題

**問題**: BFSで探索すると1-hopパスのみが返され、2-hopパスが見つからない

**解決**: DFSベースの`find_all_paths`を実装し、max_depthまでの全パスを探索

### 5.2 KGT: Cypherの方向性問題

**問題**: 有向マッチング(`->`)を使用すると結果が0件

**解決**: 無向マッチング(`-`)に変更。PrimeKGのリレーションは方向が不定なため

### 5.3 SAFE: スキーマグラフが空

**問題**: Neo4jからスキーマを取得しようとすると空

**解決**: `schema_v2.py`のSCHEMA_GRAPHから直接構築

### 5.4 SAFE: Pydantic Structured Output

**問題**: OpenAIの新しいstructured output形式でエラー

**解決**: ネストしたPydanticモデルを定義

```python
class PseudoEdgeSchema(BaseModel):
    src_node: str
    src_type: str
    relation: str
    tgt_node: str
    tgt_type: str
    is_anchor: bool = False

class PseudoQueryGraphResponse(BaseModel):
    edges: List[PseudoEdgeSchema] = Field(default=[])
```

---

## 6. 使用方法

### 6.1 KGT Pipeline

```bash
# 評価
docker compose exec app python pipeline/evaluate_kgt_pipeline.py --num-samples 20

# デバッグ（データファイルから）
docker compose exec app python pipeline/debug_kgt_pipeline.py \
  --data result/dataset_v2/two_hop.jsonl --index 0 1 2

# デバッグ（直接質問）
docker compose exec app python pipeline/debug_kgt_pipeline.py \
  --question "What diseases are associated with BRCA1?" --entity "BRCA1"
```

### 6.2 SAFE Pipeline

```bash
# 評価
docker compose exec app python pipeline/evaluate_safe_pipeline.py --num-samples 20

# デバッグ（データファイルから）
docker compose exec app python pipeline/debug_safe_pipeline.py \
  --data result/dataset_v2/two_hop.jsonl --index 0 1 2

# デバッグ（直接質問）
docker compose exec app python pipeline/debug_safe_pipeline.py \
  --question "What diseases are associated with BRCA1?" --entity "BRCA1"
```

---

## 7. 今後の改善案

### 7.1 Intersectionクエリ対応

- KGT/SAFEに複数アンカー検出と交差検索ロジックを追加
- クエリタイプ（chain vs intersection）の自動判定

### 7.2 ハイブリッドアプローチ

- クエリタイプに応じてパイプラインを切り替え
  - one_hop, two_hop → KGT or SAFE
  - intersection → Type KoPL

### 7.3 パス選択の改善

- Rerankingモデルの導入
- Few-shot examplesの活用

---

## 8. 結論

- **one_hop**: KGTが最も高精度（100%）、スキーマベースのアプローチが有効
- **two_hop**: SAFEが最も高精度（50%）だが、まだ改善の余地あり
- **intersection**: Type KoPLが圧倒的に強い（100%）、KGT/SAFEは非対応
- 単一のパイプラインで全クエリタイプに対応するのは困難、ハイブリッドアプローチが有効

---

## 9. 関連ファイル

### 実装
- `app/pipeline/kgt_pipeline.py` - KGTパイプライン
- `app/pipeline/safe_pipeline.py` - SAFEパイプライン
- `app/pipeline/evaluate_kgt_pipeline.py` - KGT評価
- `app/pipeline/evaluate_safe_pipeline.py` - SAFE評価
- `app/pipeline/debug_kgt_pipeline.py` - KGTデバッグ
- `app/pipeline/debug_safe_pipeline.py` - SAFEデバッグ

### データ
- `result/dataset_v2/one_hop.jsonl`
- `result/dataset_v2/two_hop.jsonl`
- `result/dataset_v2/two_intersection.jsonl`
- `result/dataset_v2/three_intersection.jsonl`

### 関連レポート
- `report/dataset_v2_generation_report.md` - データセットv2生成レポート
- `report/two_hop_chain_question_generation_improvement_report.md` - 質問生成改善レポート
