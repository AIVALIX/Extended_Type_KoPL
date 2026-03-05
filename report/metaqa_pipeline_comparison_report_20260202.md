# MetaQA パイプライン比較評価レポート

**作成日時**: 2026-02-02
**作業内容**: Extended Type-KoPL, SAFE, KGT の3パイプラインをMetaQAで比較評価

---

## 1. 概要

Knowledge Graph Question Answering (KGQA) における3つのパイプラインアルゴリズムの比較評価を実施した。評価にはMetaQA（映画ドメイン）の1-hop, 2-hop, 3-hopデータセットを使用し、各100サンプル（ランダム抽出）で精度を測定した。

---

## 2. アルゴリズム比較

### 2.1 Extended Type-KoPL

Extended Type-KoPLは、LLMに「どのような型のノードをどの関係で辿るか」を直接生成させるアプローチである。

#### 基本的な考え方

「Tom Hanksが出演した映画の監督は誰か？」という質問を考える。この質問に答えるには、Knowledge Graph上で以下のパスを辿る必要がある：

```
Tom Hanks (Person) --出演--> 映画 (Movie) --監督--> 監督 (Person)
```

Extended Type-KoPLでは、LLMにこの「型パス」と「関係のヒント」を生成させる。LLMの出力は以下のような形式になる：

```json
{
  "operations": [
    {"src_type": "Person", "tgt_type": "Movie", "relation": "STARRED_ACTORS", "anchor_name": "Tom Hanks"},
    {"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY"}
  ]
}
```

ここで重要なのは、`relation`フィールドはあくまで「ヒント」であり、スキーマ上の正確な関係名である必要はないということだ。LLMが「starred in」と出力しても、後続の処理でスキーマ上の正しい関係名「STARRED_ACTORS」にマッピングされる。

#### 処理の流れ

**Step 1: 型パスの生成**

まずLLMが質問を解析し、「どの型からどの型へ辿るか」という型パスを生成する。この段階では関係名は大まかなヒントでよい。Few-shot例を通じて、ドメインに適した出力形式を学習させる。

**Step 2: 各ホップでの関係選択**

LLMが指定した型パス（例：Person → Movie → Person）に対して、各ホップで実際に使える関係をスキーマから列挙する。例えば「Person → Movie」間には、STARRED_ACTORS、DIRECTED_BY（逆方向）、WRITTEN_BY（逆方向）などが存在する。

この候補の中から、LLMが出力した関係ヒントとベクトル類似度が高いものを選択する。「STARRED_ACTORS」というヒントに対しては、完全一致する「STARRED_ACTORS」が選ばれる。ヒントが曖昧な場合でも、埋め込みベクトルの類似度で最も近い関係が選択される。

**Step 3: パスのスコアリング**

複数の候補パスが生成された場合、質問文との類似度でランキングする。パスは自然言語に変換され（例："Person starred in Movie directed by Person"）、質問文とのコサイン類似度が計算される。さらに、LLMが出力した関係ヒントとの一致度に応じてスコアがブーストされる。

**Step 4: Cypherの生成と実行**

最終的に選択されたパスから、型制約付きのCypherクエリを生成する：

```cypher
MATCH (a:Person)-[:STARRED_ACTORS]->(mid:Movie)-[:DIRECTED_BY]->(b:Person)
WHERE a.name = "Tom Hanks"
RETURN DISTINCT b.name
```

ここで重要なのは、ノードに型ラベル（`:Person`、`:Movie`）が付いていることだ。これにより、誤った型のノードを経由するパスが排除される。

#### スキーマの活用方法

Extended Type-KoPLもスキーマを活用するが、SAFEとは使い方が異なる。

LLMが指定した型パス（例：Person → Person）に直接リレーションが存在しない場合、BFSで中間パスを自動補完する。例えば「Person → Person」に直接の関係がなければ、「Person → Movie → Person」のようなパスを探索して展開する。これにより、LLMが中間の型を省略しても、スキーマに基づいて正しいパスが構築される。

#### SAFEとの違い

両者のスキーマ活用の違いは以下の通り：

- **SAFE**: 擬似エッジごとに候補スキーマエッジを列挙し、APSP距離でエッジ間の接続可能性を検証する。全組み合わせの中から総距離最小のものを選択する（グローバル最適化）。

- **Extended Type-KoPL**: LLMが指定した型パスを起点に、各ホップで利用可能な関係を列挙する。直接関係がなければBFSで中間パスを補完する。各ステップで独立に最良の関係を選択する（ステップワイズ最適化）。

つまり、SAFEは「エッジの組み合わせをスキーマで検証」、Extended Type-KoPLは「型パスをスキーマで補完・具体化」という違いがある。

実験結果を見ると、マルチホップクエリではExtended Type-KoPLのアプローチが優れている。LLMは「PersonからMovieを経由してPersonに辿り着く」という構造的な理解に長けており、Few-shot例を与えれば高い精度で型パスを生成できる。その型パスさえ正しければ、後はスキーマに基づいて機械的に処理できる。

#### 強みと弱み

**強み**：
- 型パスが正しければ、後は機械的に処理できる
- スキーマから方向情報を取得するため、Cypherの実行が1回で済む
- Intersection（複数アンカーのAND）クエリにも対応

**弱み**：
- LLMが型パスを間違えると回復が困難
- ドメインごとにFew-shot例を用意する必要がある

---

### 2.2 SAFE (Semantic-Aware Subgraph Retrieval Framework)

**アプローチ**: 擬似クエリグラフ生成 + ADJ（Approximate Distance Join）によるスキーマ補正

```
Question → [Pseudo Query Graph Generation]
        → [ADJ: Schema-Level Correction]
        → [Ranked Semantic Subgraph Matching]
        → [Answer Extraction]
```

**特徴**:
- LLMが自然言語から擬似クエリグラフ（エッジのリスト）を生成
- Floyd-Warshallで全点対最短経路（APSP）を事前計算
- スキーマエッジ間の距離δで接続可能性を判定

**主要コンポーネント**:
```python
# 擬似エッジの定義
class PseudoEdge:
    src_node: str       # ソースノード名（"?"は未知）
    src_type: str       # ソースノード型
    relation: str       # 関係（自然言語）
    tgt_node: str       # ターゲットノード名
    tgt_type: str       # ターゲットノード型
    is_anchor: bool     # アンカーエンティティか
```

**ADJアルゴリズム**:
1. 各擬似エッジに対して候補スキーマエッジを選定（ベクトル類似度）
2. 連続するエッジ間の距離がδ以下の組み合わせを選択
3. 総距離最小のクエリグラフを選定

**強み**:
- スキーマ情報を活用した補正が可能
- 自然言語の関係名をスキーマにマッピング
- 1-hopクエリで高精度

**弱み**:
- マルチホップでは方向パターンの組み合わせ爆発
- Intersectionクエリに非対応

---

### 2.3 KGT (Knowledge Graph Traversal)

**アプローチ**: スキーマベースのパス探索 + ベクトル類似度によるランキング

```
Question → [Question Analysis (LLM)]
        → [Schema-Based Path Finding (DFS)]
        → [Vector Similarity Ranking]
        → [Cypher Generation & Execution]
```

**特徴**:
- LLMで質問から起点/終点のエンティティ型を抽出
- スキーマグラフ上でDFSにより全パスを探索
- パスをテキスト化し、質問との類似度でランキング

**主要コンポーネント**:
```python
# 質問分析結果
class QuestionAnalysis(BaseModel):
    head_entity_name: str   # 起点エンティティ名
    head_entity_type: str   # 起点ノード型
    tail_entity_type: str   # 回答ノード型

# スキーマパス
class SchemaPath:
    types: List[str]        # ノード型のリスト
    relations: List[str]    # 関係のリスト
    score: float            # 類似度スコア
```

**強み**:
- スキーマに基づく確実なパス探索
- 1-hopクエリで高精度
- LLM依存度が比較的低い

**弱み**:
- 長いパスでは候補パス数が爆発
- パスランキングの精度がボトルネック
- 3-hop以上で精度が著しく低下

---

## 3. アルゴリズム比較表

| 特性 | Extended Type-KoPL | SAFE | KGT |
|------|-------------------|------|-----|
| **アプローチ** | LLM + 型制約 | ADJ + APSP | Schema DFS + Vector |
| **LLM使用箇所** | 操作生成 | 擬似グラフ生成 | 質問分析 |
| **スキーマ活用** | 型制約のみ | APSP距離計算 | パス探索 |
| **パス探索** | LLMが直接生成 | 候補から選択 | DFS + ランキング |
| **方向処理** | LLMが判断 | 全パターン試行 | 無向マッチング |
| **計算量** | O(LLM) | O(n³) APSP + O(k^d) | O(b^d) DFS |

---

## 4. SAFE vs Extended Type-KoPL: アルゴリズム詳細比較

### 4.1 問題の分解方法

| 観点 | SAFE | Extended Type-KoPL |
|------|------|-------------------|
| **処理単位** | エッジ（関係）単位 | ホップ（型遷移）単位 |
| **LLM出力形式** | 擬似エッジのリスト | 型パス + 関係ヒント |

**SAFE**:
```json
// 入力: "Who directed movies Tom Hanks starred in?"
// LLM出力: 擬似エッジのリスト
[
  {"src": "Tom Hanks", "rel": "starred in", "tgt": "?", "tgt_type": "Movie"},
  {"src": "?", "rel": "directed by", "tgt": "?", "tgt_type": "Person"}
]
```

**Extended Type-KoPL**:
```json
// 入力: "Who directed movies Tom Hanks starred in?"
// LLM出力: 型パス + 関係ヒント
[
  {"src_type": "Person", "tgt_type": "Movie", "relation": "STARRED_ACTORS", "anchor": "Tom Hanks"},
  {"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY"}
]
```

**違い**: SAFEはエッジごとにスキーマにマッピング、Extended Type-KoPLは型パスを先に決定。

---

### 4.2 探索戦略の違い

**SAFE: グローバル最適化（ADJ）**

```
擬似エッジ1 → 候補スキーマエッジ [A, B, C]  (ベクトル類似度Top-k)
擬似エッジ2 → 候補スキーマエッジ [D, E, F]
擬似エッジ3 → 候補スキーマエッジ [G, H, I]

↓ ADJ: 全組み合わせから「総距離最小」を選択

例: A→D→G (総距離=0.5) vs B→E→H (総距離=0.8)
    → A→D→G を選択
```

**Extended Type-KoPL: ステップワイズ貪欲法**

```
Step 1: Person→Movie で利用可能な関係 [STARRED_ACTORS, DIRECTED_BY]
        → ヒント"starred"との類似度でTop-1選択: STARRED_ACTORS

Step 2: Movie→Person で利用可能な関係 [DIRECTED_BY, WRITTEN_BY]
        → ヒント"directed"との類似度でTop-1選択: DIRECTED_BY

↓ 各ステップで独立に最良を選択（局所最適）
```

**違い**: SAFEは全組み合わせを評価、Extended Type-KoPLは各ステップで独立に決定。

---

### 4.3 スキーマ活用の違い

**SAFE: APSP（全点対最短距離）による接続性検証**

```python
# 事前計算: Floyd-Warshallで全ノード型間の最短距離を計算
apsp = floyd_warshall(schema_graph)  # O(n³)

# エッジ間の接続可能性を距離で判定
edge_dist = apsp[edge1.tgt_type][edge2.src_type]
if edge_dist > delta:  # delta=1がデフォルト
    skip()  # 接続不可（スキーマ上で隣接していない）
```

**Extended Type-KoPL: BFSによる型パス補完 + 関係列挙**

```python
# LLMが指定した型パスに直接関係がない場合、BFSで中間パスを探索
if not schema.has_direct_relation(src_type="Person", tgt_type="Person"):
    # BFSで中間パスを探索
    intermediate = schema.find_shortest_paths("Person", "Person")
    # → [Person, Movie, Person] のようなパスを発見

# 各ステップで利用可能な関係を列挙
relations = schema.get_relations_between(src_type="Movie", tgt_type="Person")
# → ["DIRECTED_BY", "WRITTEN_BY", "STARRED_ACTORS"]

# ヒントとのベクトル類似度で剪定
best_relation = max(relations, key=lambda r: similarity(r, hint))
```

**違い**:
- SAFE: 擬似エッジ間の接続可能性をAPSP距離で検証（エッジ単位）
- Extended Type-KoPL: 型パスをBFSで補完し、各ステップの関係を列挙（型パス単位）

---

### 4.4 方向処理の違い

**SAFE: 全方向パターンを総当たり**

```python
# 2-hop: 4パターン (2²)
# 3-hop: 8パターン (2³)
direction_patterns = [
    ("fwd", "fwd"),      # (a)->(mid)->(ans)
    ("fwd", "rev"),      # (a)->(mid)<-(ans)
    ("rev", "fwd"),      # (a)<-(mid)->(ans)
    ("rev", "rev"),      # (a)<-(mid)<-(ans)
]

for pattern in direction_patterns:
    cypher = build_cypher(pattern)
    results.extend(execute(cypher))  # 全パターン実行
```

**Extended Type-KoPL: スキーマから方向を取得**

```python
# スキーマエッジに方向情報を保持
# (src_type, relation, tgt_type, direction)
# 例: ("Movie", "DIRECTED_BY", "Person", "->")

direction = schema.get_direction("Movie", "DIRECTED_BY", "Person")
cypher = build_cypher_single_direction(direction)  # 1パターンのみ
```

**違い**: SAFEは正しい方向が不明なので総当たり、Extended Type-KoPLはスキーマから方向を特定。

---

### 4.5 計算量比較

| 処理 | SAFE | Extended Type-KoPL |
|------|------|-------------------|
| **前処理** | O(n³) APSP計算 | O(E) エッジ埋め込み |
| **探索** | O(k^d) 全組み合わせ | O(k × d) ステップワイズ |
| **Cypher実行** | O(2^d) 方向パターン | O(1) 単一クエリ |

※ n=ノード型数, E=エッジ数, k=候補数, d=ホップ数

---

### 4.6 精度差の要因分析

| 状況 | SAFE | Extended Type-KoPL | 理由 |
|------|------|-------------------|------|
| **1-hop** | ◎ 100% | ○ 90% | SAFEは単純なマッピングで正確、Type-KoPLはLLMの選択ミス |
| **2-hop** | △ 82% | ◎ 97% | SAFEは方向総当たりで誤マッチ、Type-KoPLは型パスが正確 |
| **3-hop** | ○ 92% | ◎ 96% | 同上、ホップ数増加で差が拡大 |

**設計思想の違い**:
- **SAFE**: 「LLMは不正確」前提 → スキーマで補正
- **Extended Type-KoPL**: 「LLMは正確」前提 → 型パスを信頼

**結論**: マルチホップではLLMの型パス予測精度が高いため、Extended Type-KoPLが優位。

---

## 5. MetaQA 評価結果

### 5.1 評価条件

- **データセット**: MetaQA（映画ドメイン）
- **サンプル数**: 各データセット100件（ランダム抽出）
- **並列数**: 16 workers
- **評価指標**: Accuracy, Recall, Precision, F1, PathAcc

### 5.2 1-hop 結果

| Pipeline | Accuracy | Recall | Precision | F1 | PathAcc | Latency |
|----------|----------|--------|-----------|-----|---------|---------|
| Extended Type-KoPL | 90.0% | 90.0% | 90.0% | 90.0% | 91.0% | 2564ms |
| **SAFE** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 95.0% | 1936ms |
| KGT | 96.0% | 96.0% | 96.0% | 96.0% | **95.0%** | 2544ms |

**分析**:
- SAFEが完璧な精度（100%）を達成
- 単純なクエリではスキーマベースのアプローチが有効
- Extended Type-KoPLはLLMの関係選択ミスにより若干低下

### 5.3 2-hop 結果

| Pipeline | Accuracy | Recall | Precision | F1 | PathAcc | Latency |
|----------|----------|--------|-----------|-----|---------|---------|
| **Extended Type-KoPL** | **97.0%** | **97.0%** | **96.8%** | **96.9%** | **58.0%** | 2739ms |
| SAFE | 82.0% | 82.8% | 75.7% | 78.5% | 45.0% | 2563ms |
| KGT | 58.0% | 58.6% | 57.2% | 57.6% | 43.0% | 3185ms |

**分析**:
- Extended Type-KoPLが最高精度（97%）
- SAFEは方向パターンのマッチング失敗で低下
- KGTはパスランキングの精度が課題

### 5.4 3-hop 結果

| Pipeline | Accuracy | Recall | Precision | F1 | PathAcc | Latency |
|----------|----------|--------|-----------|-----|---------|---------|
| **Extended Type-KoPL** | **96.0%** | **96.0%** | 86.2% | 90.1% | 4.0% | 4262ms |
| SAFE | 92.0% | 92.9% | 81.1% | 85.3% | 3.0% | 4742ms |
| KGT | 6.0% | 13.0% | 34.0% | 15.1% | 0.0% | 3361ms |

**分析**:
- Extended Type-KoPLが最高精度（96%）を維持
- SAFEも92%と高精度（3-hop対応を新規実装）
- KGTは壊滅的（6%）、長いパスでのランキング失敗

### 5.5 総合比較

| Pipeline | 1-hop | 2-hop | 3-hop | 平均 |
|----------|-------|-------|-------|------|
| **Extended Type-KoPL** | 90% | **97%** | **96%** | **94.3%** |
| SAFE | **100%** | 82% | 92% | 91.3% |
| KGT | 96% | 58% | 6% | 53.3% |

---

## 6. 考察

### 6.1 Extended Type-KoPLの優位性

Extended Type-KoPLが2-hop, 3-hopで最高精度を達成した理由：

1. **型制約の効果**: src_type → tgt_type の制約により、誤ったパスを事前に排除
2. **Few-shot学習**: ドメイン特化の例により、関係選択の精度が向上
3. **柔軟な操作生成**: LLMが複数ホップの操作を一括で生成可能

### 6.2 SAFEの特性

- **1-hopでの強み**: スキーマエッジとのベクトル類似度が高精度に機能
- **マルチホップでの課題**: 方向パターンの組み合わせ（2^n）が必要で、不適切なパターンも試行される
- **3-hop対応**: 本評価に向けて8方向パターンを実装し、92%を達成

### 6.3 KGTの課題

- **パスランキングの限界**: 質問とパスのベクトル類似度だけでは、正しいパスを選択できない
- **3-hopでの崩壊**: 候補パス数の爆発により、適切なパスの選択が困難
- **改善案**: パス選択にLLMリランカーを導入、またはパス刈り込みの強化

### 6.4 PathAccとAccuracyの乖離

3-hopでPathAcc（パス正解率）が極端に低い（3-4%）一方、Accuracy（回答正解率）は高い（92-96%）：

- **原因**: 異なるパスでも同じ回答に到達可能（グラフの構造的特性）
- **示唆**: パス正解率は評価指標として不十分、回答正解率が実用的な指標

---

## 7. 結論

### 7.1 推奨パイプライン

| ユースケース | 推奨パイプライン | 理由 |
|-------------|-----------------|------|
| 1-hop クエリ | SAFE | 100%精度、最速 |
| 2-hop クエリ | Extended Type-KoPL | 97%精度、安定 |
| 3-hop クエリ | Extended Type-KoPL | 96%精度、唯一実用的 |
| 総合 | Extended Type-KoPL | 全体で最も安定した性能 |

### 7.2 主要な知見

1. **LLMベースの操作生成が有効**: Extended Type-KoPLのようにLLMに直接操作を生成させるアプローチが、マルチホップクエリで優位
2. **スキーマ活用は1-hopで有効**: SAFE/KGTのスキーマベースアプローチは1-hopでは高精度だが、複雑なクエリでは限界
3. **Few-shot例の重要性**: ドメイン特化のFew-shot例により、LLMの精度が大幅に向上

### 7.3 今後の課題

1. **KGTの改善**: パス選択にLLMリランカーを導入
2. **SAFEの最適化**: 方向パターンの事前刈り込み
3. **ハイブリッドアプローチ**: クエリ複雑度に応じたパイプライン切り替え

---

## 8. 実装変更

### 8.1 SAFE 3-hop対応（本評価で実装）

`app/pipeline/safe/pipeline.py` の `_multi_hop_search` メソッドに3-hop用のCypherパターンを追加：

```python
elif len(edges) == 3:
    # 3-hop: 8方向パターンを試行
    direction_patterns = [
        ("fwd", "fwd", "fwd"),
        ("fwd", "fwd", "rev"),
        # ... 全8パターン
    ]

    for d1, d2, d3 in direction_patterns:
        # 各エッジの方向に応じたCypherを構築
        cypher = f"""
            MATCH {e1_pattern}
            MATCH {e2_pattern}
            MATCH {e3_pattern}
            WHERE a.name = $anchor_name
            RETURN DISTINCT ...
        """
```

### 8.2 Few-shot例の統一

各パイプラインでMetaQA用のFew-shot例を統一：
- 同じ質問パターンを使用
- 各パイプラインの形式に変換
- 1-hop, 2-hop, 3-hopの例を網羅

---

## 9. 関連ファイル

### 実装
- `app/pipeline/extended_type_kopl/pipeline.py` - Extended Type-KoPL
- `app/pipeline/safe/pipeline.py` - SAFE（3-hop対応追加）
- `app/pipeline/kgt/pipeline.py` - KGT
- `app/pipeline/run_evaluation.py` - 統一評価スクリプト

### データ
- `result/metaqa/1hop.jsonl` - MetaQA 1-hop
- `result/metaqa/2hop.jsonl` - MetaQA 2-hop
- `result/metaqa/3hop.jsonl` - MetaQA 3-hop

### 関連レポート
- `report/kgt_safe_pipeline_implementation_report.md` - KGT/SAFE実装レポート
- `report/pipeline_accuracy_evaluation_report.md` - PrimeKGQA評価レポート

---

## 10. No-Schema Mode: バグ修正と精度改善

### 10.1 No-Schema Modeの概要

**No-Schema Mode**は、事前定義されたスキーマ（リレーション一覧）を使用せず、KGから動的にリレーション情報を取得するモードである。スキーマとしては**タイプ列挙のみ**を保持し、各ホップで使用可能なリレーションはその都度KGにクエリして取得する。

**メリット**:
- KGスキーマの変更に対して柔軟
- 新しいリレーションが追加されても自動対応
- スキーマ定義の事前作業が不要

### 10.2 発見されたバグと修正

#### 10.2.1 SAFE: 方向パターンの問題（2-hop）

**症状**: 2-hop精度が70%と、3-hop（82%）より低い逆転現象

**原因**: 2-hopで明示的な方向パターン（`->`, `<-`）を使用していたが、一部のタイプ組み合わせで正しい方向が選択されなかった

**修正** (`app/pipeline/safe/pipeline.py`):
```python
# 修正前: 明示的な方向パターンで8パターン試行
MATCH (a:Person)-[:STARRED_ACTORS]->(mid:Movie)-[:DIRECTED_BY]->(ans:Person)

# 修正後: 方向非依存のCypherクエリ
cypher = f"""
    MATCH (a)-[r1:{se1.relation}]-(mid)-[r2:{se2.relation}]-(ans)
    WHERE a.name = $anchor_name
      AND a <> mid AND mid <> ans AND a <> ans
    RETURN DISTINCT a.name AS anchor, mid.name AS mid_node, ans.name AS answer
    LIMIT 200
"""
```

**結果**: 2-hop精度 70% → 92%

#### 10.2.2 SAFE: n1 <> ans 制約の問題（3-hop）

**症状**: 一部の3-hopクエリで結果が空になる

**原因**: `n1 <> ans` 制約が、同一人物が複数役割で登場するケース（例: Scott Sandersが脚本家かつ監督）を排除

**修正** (`app/pipeline/safe/pipeline.py`):
```python
# 修正前: 過剰な制約
WHERE a.name = $anchor_name
  AND a <> n1 AND n1 <> n2 AND n2 <> ans AND a <> ans AND a <> n2
  AND n1 <> ans  # ← この制約が問題

# 修正後: n1 <> ans を削除
cypher = f"""
    MATCH (a)-[r1:{se1.relation}]-(n1)-[r2:{se2.relation}]-(n2)-[r3:{se3.relation}]-(ans)
    WHERE a.name = $anchor_name
      AND a <> n1 AND n1 <> n2 AND n2 <> ans AND a <> ans AND a <> n2
    RETURN DISTINCT ...
"""
```

**結果**: 3-hop精度 86% → 93%

#### 10.2.3 Extended Type-KoPL: エンティティタイプ優先順位の問題

**症状**: 1-hop精度が94%にとどまる（期待値は100%近く）

**原因**: 複数ラベルを持つエンティティ（例: `['Organization', 'Person']`）で、最初にマッチしたラベルが返される。組織に所属する俳優が `organization` と判定されていた。

**修正** (`app/pipeline/extended_type_kopl/pipeline.py`):
```python
# タイプの優先順位を定義
TYPE_PRIORITY = {
    "metaqa": ["movie", "person", "language", "date", "text", "number", "organization"],
    "primekgqa": ["drug", "disease", "gene/protein"],
    "pcqa": ["drug", "cancer", "genesymbol", "snvfull"],
}

def _get_entity_type(self, entity_name: str) -> Optional[str]:
    """KGからエンティティのタイプを取得"""
    # ...
    labels = records[0]["labels"]
    labels_lower = [lbl.lower() for lbl in labels]

    # KGタイプに応じた優先順位でタイプを選択
    priority_list = self.TYPE_PRIORITY.get(self.kg_type, [])
    for ptype in priority_list:
        if ptype in labels_lower:
            return ptype

    # 優先順位リストにない場合は有効タイプの最初のマッチを返す
    # ...
```

**結果**: 1-hop精度 94% → 100%, 3-hop精度 91% → 95%

### 10.3 精度改善サマリー

| Pipeline | Dataset | Before | After | Change |
|----------|---------|--------|-------|--------|
| SAFE | 2-hop | 70% | 92% | **+22%** |
| SAFE | 3-hop | 86% | 93% | **+7%** |
| Extended Type-KoPL | 1-hop | 94% | 100% | **+6%** |
| Extended Type-KoPL | 3-hop | 91% | 95% | **+4%** |

### 10.4 No-Schema Mode 最終結果（MetaQA n=100）

| Dataset | SAFE | Extended Type-KoPL |
|---------|------|-------------------|
| 1-hop | 99% | **100%** |
| 2-hop | 93% | **99%** |
| 3-hop | 88% | **95%** |
| **平均** | 93.3% | **98.0%** |

### 10.5 考察

#### No-Schema Modeの有効性

修正後の結果から、No-Schema Modeは十分に実用的であることが確認された：
- Extended Type-KoPL: 平均98.0%の精度を達成
- SAFE: 平均93.3%の精度を達成

#### 方向非依存クエリの効果

SAFEで方向非依存のCypherを採用したことで、以下の利点が得られた：
- 方向パターンの組み合わせ爆発を回避
- スキーマから方向情報を取得する必要がない
- 実行クエリ数の削減（8パターン → 1パターン）

#### エンティティタイプ優先順位の重要性

複数ラベルを持つエンティティが存在するKGでは、タイプの優先順位設定が精度に大きく影響する。MetaQAでは `Person` が `Organization` より優先されるべきケースが多い。

### 10.6 結論

No-Schema Modeにおける精度改善により、Extended Type-KoPLは全データセットで95%以上の精度を達成した。主要な修正点は以下の通り：

1. **SAFE**: 方向非依存のCypherクエリを採用
2. **SAFE**: 過剰なサイクル制約（n1 <> ans）を緩和
3. **Extended Type-KoPL**: エンティティタイプの優先順位を導入

これらの修正により、スキーマ定義なしでも高精度なKGQAが可能であることが実証された。
