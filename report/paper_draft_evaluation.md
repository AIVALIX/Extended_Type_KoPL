# ハイブリッドスキーマ誘導トラバーサルによるナレッジグラフ質問応答: データセット構築と比較評価

---

## 概要

ナレッジグラフ質問応答（KGQA）は、自然言語の質問を構造化されたグラフトラバーサルに変換する技術である。既存のKGQAベンチマークには正解の正確性に課題がある。例えばPrimeKGQA（Chandak et al., 2023のPrimeKGから派生した既存データセット）では、質問に対する正解集合がKGの構造と一致しないケースが存在し、パイプラインの正確な性能評価を妨げている。本研究では、KGの構造から直接正解を導出する**Answer-Firstアプローチ**により、正解の正確性を保証した新しいベンチマーク**PrimeKGQA-Struct**を構築する。PrimeKGQA-Structは1-hop、2-hop、2-anchor intersection、3-anchor intersectionの4段階のクエリ複雑度をサポートし、全23,970件のサンプルについてCypherクエリによりKG上の正解集合を検証済みである。3つのKGQAパイプライン---**Extended Type-KoPL（ETK）**、**SAFE**、**KGT**---をMetaQAとPrimeKGQA-Structの両方で評価し、スキーマ複雑度、クエリ構造、埋め込みベースのパス選択が回答精度に与える影響を分析する。

---

## 1. はじめに

（省略）

---

## 2. 関連研究

（省略）

---

## 3. PrimeKGQA-Structデータセット構築

### 3.1 既存PrimeKGQAベンチマークの課題

PrimeKGから派生した既存のQAベンチマーク（以下「既存PrimeKGQA」）には、KGQAパイプラインの正確な評価を妨げる複数の品質上の課題が存在する。

**課題1: 質問の一貫性の欠如.** 既存PrimeKGQAの質問文を人手で品質評価した結果、一貫性スコア（consistency score）は0.45/1.0にとどまった。すなわち、質問文の意図とKGから期待される回答が半数以上のケースで一致しないことを意味する。

**課題2: アノテータ間一致度の低さ.** 複数アノテータによる回答の妥当性評価において、Fleiss' Kappa係数は0.12〜0.17の範囲であった。これは「わずかな一致（slight agreement）」に分類され、何を正解とすべきかについてアノテータ間で合意が得られていないことを示す。

**課題3: 構造的に不適切なQAデータ.** 最も深刻な問題は、単一の質問に対して構造的に無関係な数百件のエンティティが「正解」として付与されているケースの存在である。例えば、ある遺伝子に関連する疾患を問う質問に対して、KG上の近傍探索では到達不可能なエンティティが正解集合に含まれており、パイプラインが正しいパスを選択し正しいトラバーサルを実行しても、正解判定において不当に低い精度となる。

これらの課題を受け、KGの構造から直接正解を導出する**Answer-Firstアプローチ**に基づく新しいベンチマーク**PrimeKGQA-Struct**を構築する。PrimeKGQA-Structでは、全ての正解がCypherクエリによりKG上で検証可能であり、質問テンプレートは意味的に一貫したリレーションパスに基づいて設計される。

### 3.2 ソースナレッジグラフ

PrimeKGQA-Structは、20のバイオメディカルデータベースを統合した精密医療ナレッジグラフ**PrimeKG**（Chandak et al., 2023）から構築する。質問応答に適した明確な意味論を持つリレーションに限定し、**5種類のエンティティタイプ**と**12種類のリレーションタイプ**のサブセットを使用する。

**エンティティタイプ:**
- `drug`（薬剤）, `disease`（疾患）, `gene/protein`（遺伝子/タンパク質）, `effect/phenotype`（効果/表現型）, `exposure`（環境因子）

**リレーションタイプ（12種）:**

| ソースタイプ | リレーション | ターゲットタイプ | エッジ数 | 意味 |
|---|---|---|---|---|
| drug | target | gene/protein | 16,380 | 薬の標的タンパク質 |
| drug | carrier | gene/protein | 864 | 薬のキャリアタンパク質 |
| drug | enzyme | gene/protein | 5,317 | 薬を代謝する酵素 |
| drug | transporter | gene/protein | 3,092 | 薬を輸送するタンパク質 |
| drug | indication | disease | 9,388 | 承認適応症 |
| drug | off-label use | disease | 2,568 | 適応外使用 |
| drug | contraindication | disease | 30,675 | 禁忌 |
| drug | side effect | effect/phenotype | 64,784 | 副作用 |
| gene/protein | ppi | gene/protein | 642,150 | タンパク質間相互作用 |
| disease | phenotype present | effect/phenotype | 150,317 | 疾患で見られる症状 |
| disease | phenotype absent | effect/phenotype | 1,193 | 疾患で見られない症状 |
| exposure | linked to | disease | 2,304 | 環境因子と疾患の関連 |

元のPrimeKGは10種類のエンティティタイプと30以上のリレーションタイプを含むが、質問文として定式化が困難なリレーション（例：`parent-child`階層、150万エッジ超の`expressed gene`、260万エッジ超の`synergistic interaction`）を除外し、意味的に解釈可能な質問応答ペアに焦点を当てた。

### 3.3 クエリパターンの体系化

PrimeKGQA-Structのクエリタイプは、KG上のグラフパターンとして体系的に分類できる。ノード数に応じたパターン分類を採用し、冗長なパターン（三角形等）を除外した意味的に有効なクエリパターンのみを選定する。

#### 3.3.1 2ノードパターン（N2）: 1-hopクエリ

最も基本的なパターンであり、アンカーノード $A$ から単一リレーションで回答ノード $X$ に到達する。

$$A \xrightarrow{r} X$$

1パターンのみ。例: *"What genes does Aspirin target?"*

$$q_{\text{1-hop}}: e_{\text{anchor}} \xrightarrow{r} \{e_{\text{answer}}\}$$

#### 3.3.2 3ノードパターン（N3）: 2-hopクエリ・2-anchor intersectionクエリ

3つのノードを含むパターンは理論上複数存在するが、三角形パターン（3ノードが互いに接続）は1-hopパターンの冗長な組み合わせに過ぎないため除外する。有効なパターンは以下の**6パターン**である:

**線形パターン（2-hop）:** アンカーから中間ノードを経由して回答に到達する。

$$A \xrightarrow{r_1} B \xrightarrow{r_2} X$$

例: *"What diseases are associated with genes targeted by Metformin?"*

$$q_{\text{2-hop}}: e_{\text{anchor}} \xrightarrow{r_1} e_{\text{mid}} \xrightarrow{r_2} \{e_{\text{answer}}\}$$

**星型パターン（2-anchor intersection）:** 2つのアンカーから共通のターゲットに接続するパターン。

$$A \xrightarrow{r_1} X \xleftarrow{r_2} B$$

例: *"What genes are targeted by both Aspirin and Ibuprofen?"*

$$q_{\text{2-int}}: \{e_{\text{answer}}\} = \{e \mid e_{\text{anchor}_1} \xrightarrow{r_1} e\} \cap \{e \mid e_{\text{anchor}_2} \xrightarrow{r_2} e\}$$

線形パターンと星型パターンの方向バリエーションにより計6パターンとなる。

#### 3.3.3 4ノードパターン（N4）: 3-anchor intersectionクエリ

4ノードパターンからは、以下の**2つの有効パターン**を採用する:

**N4_1: 3-anchor星型パターン.** 3つのアンカーが共通の回答ノードに接続する。

$$A \xrightarrow{r_1} X, \quad B \xrightarrow{r_2} X, \quad C \xrightarrow{r_3} X$$

例: *"What diseases are associated with BRCA1, TP53, and EGFR?"*

$$q_{\text{3-int}}: \{e_{\text{answer}}\} = \bigcap_{k=1}^{3} \{e \mid e_{\text{anchor}_k} \xrightarrow{r_k} e\}$$

**N4_2: 線形+分岐パターン.** アンカーから線形に辿った後に分岐する、または2-hopと1-hopの組み合わせ。

$$A \xrightarrow{r_1} B \xrightarrow{r_2} X \xleftarrow{r_3} C$$

本研究ではN4_1（3-anchor intersection）を中心に評価し、N4_2は今後の課題とする。

#### 3.3.4 クエリパターンのまとめ

| ノード数 | パターン名 | 構造 | 本研究での対応 | テンプレート数 |
|---|---|---|---|---|
| 2 (N2) | 1-hop | $A \to X$ | 1-hopクエリ | 10 |
| 3 (N3) | 線形 | $A \to B \to X$ | 2-hopクエリ | 7 |
| 3 (N3) | 星型 | $A \to X \gets B$ | 2-intersectionクエリ | 7 |
| 4 (N4_1) | 3-anchor星型 | $A,B,C \to X$ | 3-intersectionクエリ | 3 |
| 4 (N4_2) | 線形+分岐 | $A \to B \to X \gets C$ | 今後の課題 | -- |

### 3.4 Answer-Firstサンプル生成

**Answer-Firstアプローチ**を採用する。質問を生成してから回答を探索するのではなく、まずナレッジグラフから有効な（アンカー, 回答）ペアを特定し、その後自然言語質問を生成する。

**ステップ1: 候補ペア抽出.** 各クエリテンプレートに対して、Neo4jデータベースに対するCypherクエリを実行し、構造的制約を満たすすべての有効なアンカー-回答ペアを取得する。

1-hopの場合:
```cypher
MATCH (answer:tgt_type)<-[r:relation]-(anchor:src_type)
WITH anchor, collect(DISTINCT answer) AS answers
WHERE size(answers) >= 1 AND size(answers) <= 50
RETURN anchor, answers
ORDER BY rand()
LIMIT N
```

2-hopの場合、2ホップで到達可能な全回答を集約し、組み合わせ爆発を防ぐため総回答数でフィルタする:
```cypher
MATCH (a:src)-[r1:rel1]-(z:mid)-[r2:rel2]-(x:tgt)
WHERE a <> z AND z <> x AND a <> x
WITH a, collect(DISTINCT x) AS all_answers
WHERE size(all_answers) >= 1 AND size(all_answers) <= 100
RETURN a, all_answers
```

intersectionクエリの場合、共通の回答エンティティを持つアンカータプルを探索する:
```cypher
MATCH (a:type_a)-[ra:rel_a]->(x:int_type)<-[rb:rel_b]-(b:type_b)
WHERE a <> b
WITH a, b, collect(DISTINCT x) AS answers
WHERE size(answers) >= 1 AND size(answers) <= 50
RETURN a, b, answers
```

**ステップ2: エンティティ名フィルタリング.** 人間が読みにくい、あるいはLLMが処理しにくいエンティティ名を持つサンプルを除外するヒューリスティックフィルタを適用する:
- 30文字超の長い名前（化学IUPAC名等）
- データベース識別子（例: `DB00001`, `CHEMBL123456`）
- 化学式パターン
- 遺伝子/タンパク質名で `[A-Z][A-Z0-9]{1,5}` パターンに一致するもの（例: `BRCA1`, `TP53`）は明示的に**許可**

**ステップ3: 質問テンプレートのインスタンス化.** 各クエリテンプレートは2〜3の質問パターンを定義する。各サンプルについてテンプレートをランダムに1つ選択し、`{anchor}`プレースホルダにアンカーエンティティ名を代入する。

テンプレート例: `"What genes does {anchor} target?"` → `"What genes does Aspirin target?"`

**ステップ4: LLMパラフレーズ.** 言語的多様性を高めるため、テンプレート生成された各質問をLLM（GPT-4o-mini）で以下の制約のもとパラフレーズする:
- エンティティ名は正確に保持する
- ドメインに適切な動詞を使用する（例: `target`→"targets", `indication`→"treats"）
- グラフ用語（node, edge, hop）は使用しない
- 意味的な内容を保持する

パラフレーズは8ワーカースレッドで並列化し、セマフォで同時APIリクエスト数を16に制限する。

### 3.5 データセット統計

| クエリタイプ | グラフパターン | テンプレート数 | サンプル数 |
|---|---|---|---|
| 1-hop | N2: $A \to X$ | 10 | 8,331 |
| 2-hop | N3線形: $A \to B \to X$ | 7 | 10,414 |
| 2-intersection | N3星型: $A \to X \gets B$ | 7 | 4,618 |
| 3-intersection | N4_1: $A,B,C \to X$ | 3 | 607 |
| **合計** | | **27** | **23,970** |

各サンプルは以下を含む:
- 自然言語質問（パラフレーズ済み）
- アンカーエンティティ名・タイプ
- ゴールドリレーションパス
- ゴールド回答エンティティ集合（各回答の名前とタイプ）

### 3.6 既存ベンチマークとの比較

| 特性 | MetaQA | 既存PrimeKGQA | PrimeKGQA-Struct（本研究） |
|---|---|---|---|
| ドメイン | 映画 | バイオメディカル | バイオメディカル |
| エンティティタイプ数 | 7 | 10 | 5 |
| リレーションタイプ数 | 9 | 30+ | 12 |
| クエリタイプ | 1/2/3-hop | 1-hop | 1-hop, 2-hop, 2-int, 3-int |
| Intersectionクエリ | なし | なし | あり |
| 総サンプル数 | 600 | -- | 23,970 |
| 回答形式 | エンティティ集合 | エンティティ集合 | エンティティ集合 |
| 質問生成 | テンプレート | LLM生成 | テンプレート + LLMパラフレーズ |
| 回答検証 | KG構造一致 | 未検証 | Cypherクエリで全件検証 |
| 一貫性スコア | -- | 0.45 | 1.0（構造的保証） |
| エンティティ名の複雑さ | 映画/人名 | 薬剤/遺伝子/疾患名 | 薬剤/遺伝子/疾患名 |

PrimeKGQA-Structは既存ベンチマークを以下の点で拡張する: (1) Answer-Firstアプローチにより正解の構造的正確性を保証、(2) 集合演算を必要とするintersectionクエリの導入、(3) グラフパターンに基づく体系的なクエリ分類、(4) 大規模な評価セット（23,970サンプル）の提供。

---

## 4. KGQAパイプラインの説明

3つのKGQAパイプラインを評価する。いずれも質問理解、スキーマレベルのパス選択、インスタンスレベルの回答取得という共通アーキテクチャを共有するが、スキーマパス選択と回答ランキングの方法が異なる。

### 4.1 Extended Type-KoPL（ETK）

ETKは、LLMベースのクエリ分解と埋め込みベースのスキーマパス選択、およびオプションのLLMリランキングを組み合わせたハイブリッドパイプラインである。

**Phase 1: 擬似クエリ生成.** LLMが自然言語質問 $q$ をAtomic Type-KoPLプログラム $\mathcal{P}$ に変換する:

$$\mathcal{P} = \text{LLM}(q, e_{\text{anchor}}, \tau_e, \tau_{\text{target}}, \mathcal{T}, \text{examples})$$

プログラムは型付き操作の列を指定する: $\mathcal{P} = [(t_0, t_1, h_1), (t_1, t_2, h_2), \ldots]$。ここで $t_i$ はエンティティタイプ、$h_i$ はリレーションヒントである。intersectionクエリでは、異なるアンカーを持つ複数の独立した操作列が `final_operation = intersection` とともに生成される。

**Phase 2: ハイブリッドスキーマ探索.** プログラム中の連続する型ペア $(t_i, t_{i+1})$ に対して、スキーマグラフから有効なリレーションをすべて列挙する:

$$R_i = \{(r, d) \mid (t_i, r, t_{i+1}) \in \mathcal{E}_S, \; d \in \{\rightarrow, \leftarrow\}\}$$

$t_i$ と $t_{i+1}$ の間に直接リレーションが存在しない場合、スキーマグラフ上のBFSで中間タイプを特定し型パスを展開する。各ステップの候補リレーションの直積 $\mathcal{C} = R_1 \times R_2 \times \cdots \times R_L$ で候補パス集合を生成する。

**Phase 3: ベクトル剪定.** 各候補パス $p_j$ をリレーション-動詞マッピング（例: `target` → `"targets"`, `indication` → `"treats"`）を用いて自然言語テキスト表現に変換する:

$$\text{text}(p_j) = t_0 \;\texttt{-[}\text{verb}(r_1)\texttt{]}\;d_1\; t_1 \;\texttt{-[}\text{verb}(r_2)\texttt{]}\;d_2\; t_2 \;\cdots$$

最終ランキングスコアは、質問-パス間のグローバル類似度とホップごとのヒントマッチングを組み合わせる:

$$\text{score}(p_j) = \underbrace{\cos\bigl(\text{Emb}(q),\; \text{Emb}(\text{text}(p_j))\bigr)}_{\text{グローバル類似度}} + \underbrace{\frac{1}{L}\sum_{i=1}^{L} \cos\bigl(\text{Emb}(h_i),\; \text{Emb}(\text{noun}(r_{j,i}))\bigr)}_{\text{ホップごとのヒントブースト}}$$

ここで $h_i$ はPhase 1の $i$ 番目のリレーションヒント、$r_{j,i}$ はパス $p_j$ の $i$ 番目のリレーション、$\text{noun}(\cdot)$ は事前定義マッピングによるリレーション名の名詞形変換である。埋め込み関数 $\text{Emb}(\cdot)$ はOpenAI `text-embedding-3-small` を使用する。上位 $K$ パスが選択される（リランカーなし: $K=1$、リランカーあり: $K=10$）。

**Phase 3.5: LLMリランカー（オプション）.** 有効時、LLMが上位 $K$ 候補から最適パスを構造化出力で選択する。KoPLプログラムのコンテキスト（アンカー名、型チェーン、ヒント）が追加入力として提供される。

**Phase 4: サブグラフ取得.** 選択されたスキーマパスをCypherクエリとしてNeo4jデータベースに対して実行する。ロバスト性のため、リレーションは方向制約なし（`-[r]-`、双方向トラバーサル）で辿る。

**Phase 5: KoPL論理演算.** intersectionクエリでは、各アンカーから取得したエンティティ集合の共通部分を取る: $A_{\text{final}} = \bigcap_k A_k$。pathクエリでは、取得した全エンティティの和集合を返す。

### 4.2 SAFE（Semantic-Aware Factual Extraction）

SAFEは、埋め込み距離に基づく2段階の意味的マッチングフレームワークを実装する。

**Phase 1: 擬似クエリグラフ生成.** LLMが $q$ を擬似エッジの集合 $\{(e_{\text{src}}, \tau_{\text{src}}, r, e_{\text{tgt}}, \tau_{\text{tgt}})\}$ に変換する。既知エンティティはその名前を、未知ノードは "?" を使用する。

**Phase 2: 近似距離結合（ADJ）.** 各擬似エッジをL2埋め込み距離を用いて候補スキーマエッジにマッピングする:

$$d_{\text{schema}}(e_{\text{pseudo}}, e_{\text{schema}}) = \|\text{Emb}(e_{\text{pseudo}}) - \text{Emb}(e_{\text{schema}})\|_2$$

擬似エッジごとに上位 $k$ 個の候補スキーマエッジを選択する（$k=3$）。スキーマパスの連結性はAPSP（Floyd-Warshall）で検証し、クエリグラフ中の連続エッジがスキーマ上で距離 $\delta \leq 1$ 以内であることを保証する。

**Phase 3: ランク付き意味的サブグラフマッチング.** 優先度キューアルゴリズムがクエリグラフの各エッジを処理し、KG中のインスタンスレベルのサブグラフにマッチングする:

候補リレーション $r_G$（現在のマッチ済みノードに隣接）に対して:
$$d_r = \|\text{Emb}(r_Q) - \text{Emb}(r_G)\|_2$$

$r_G$ 経由で到達可能な候補ターゲットノード $u'_G$ に対して:
$$d_{\text{node}} = \min_{\tau \in \text{labels}(u'_G)} \|\text{Emb}(\tau_Q) - \text{Emb}(\tau)\|_2$$

分解エッジ距離は $d_e = d_r + d_{\text{node}}$ であり、全マッチ済みエッジにわたり蓄積される。最良 $k$ 件の閾値を超える部分マッチは剪定される。

候補インデックスは3つの類似度関数を使用する:
- **SimRel**: L2距離による上位 $k_r$ リレーション（$k_r = 10$）
- **SimTyp**: L2距離による上位 $k_t$ タイプ（$k_t = 8$）
- **SimEnt**: L2距離による上位 $k_e$ エンティティ（$k_e = 3$、アンカー解決用）

### 4.3 KGT（Knowledge Graph Traversal）

KGTは、構造的な最短パス推論を優先するシンプルなパイプラインである。

**Phase 1: 質問分析.** LLMが質問からヘッドエンティティ名、ヘッドエンティティタイプ、テールエンティティタイプを抽出する。

**Phase 2: スキーマベースパス探索.** BFSでヘッドエンティティタイプからスキーマグラフ上のすべての到達可能なターゲットタイプへの**最短パスのみ**を列挙する（最大深度3）。各パスは質問埋め込みとパステキスト埋め込み（Cypher記法形式）のコサイン類似度でスコアリングされる:

$$\text{score}(p) = \cos\bigl(\text{Emb}(q),\; \text{Emb}(\text{cypher}(p))\bigr)$$

**Phase 3: テンプレートベースCypher生成.** 最高スコアのスキーマパスを方向対応テンプレート（2-hopでは順方向/逆方向の4組み合わせ）でCypherクエリにインスタンス化する。

**Phase 4: 回答抽出.** Cypherの結果を回答エンティティ集合として抽出する。

### 4.4 パイプラインアプローチの比較

| 構成要素 | ETK | SAFE | KGT |
|---|---|---|---|
| クエリ分解 | Type-KoPLプログラム（型付き操作） | 擬似エッジグラフ | エンティティ+タイプ抽出 |
| スキーマパス探索 | ステップワイズBFS + 直積 | L2距離によるADJ | BFS最短パスのみ |
| パススコアリング | コサイン類似度 + ホップごとブースト | 蓄積L2距離 | コサイン類似度 |
| インスタンス取得 | 双方向Cypher | 優先度キューマッチング | 方向付きCypher |
| マルチホップ対応 | 最大3-hop（型パス展開） | 最大3-hop（マルチエッジ） | 最大3-hop（最短パス） |
| 集合演算 | Intersection/Union/Exclude | 単一サブグラフ | なし |
| 対応グラフパターン | N2, N3（線形・星型）, N4_1 | N2, N3（線形のみ） | N2, N3（線形のみ） |
| リランキング | オプションLLMリランカー | 暗黙的（優先度キュー） | なし |
| 埋め込みモデル | OpenAI text-embedding-3-small | OpenAI text-embedding-3-small | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini（構造化出力） | GPT-4o-mini（構造化出力） | GPT-4o-mini（構造化出力） |

---

## 5. 評価手法

### 5.1 評価メトリクス

予測エンティティ集合 $P$ とゴールド回答エンティティ集合 $G$ を比較する集合ベースの評価メトリクスを採用する。

**Hit Accuracy（Acc）:** ゴールド回答がすべて予測に含まれるかどうか。
$$\text{Acc} = \mathbb{1}[G \subseteq P]$$

**Recall:** 取得されたゴールド回答の割合。
$$\text{Recall} = \frac{|G \cap P|}{|G|}$$

**Precision:** 正解である予測の割合。
$$\text{Precision} = \frac{|G \cap P|}{|P|}$$

**F1スコア:**
$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Path Accuracy:** 予測リレーションパスがゴールドリレーションパスと完全一致するか（順序考慮、正規化後）。

すべてのエンティティ名比較はcase-insensitiveで行う。リレーション名は小文字化、スペース/ハイフンのアンダースコア置換、既知エイリアスの解決（例: `"associated_with"` → `"associated_disease"`）により正規化する。

### 5.2 実験設定

**ナレッジグラフ:**
- **MetaQA:** 7エンティティタイプ、9リレーションタイプ（映画ドメイン）。Neo4j（`neo4j_metaqa`）に格納。
- **PrimeKGQA-Struct:** 5エンティティタイプ、12リレーションタイプ（バイオメディカルドメイン）。Neo4j（`neo4j_primekgqa`）に格納。

**評価プロトコル:**
- 各（パイプライン, データセット）ペアについて、データセットプールから $n$ 件の質問をランダムサンプリングする
- 各質問はパイプラインを通じて独立に処理される
- 処理は `ProcessPoolExecutor` を用いて並列化する
- 結果はJSONLファイルとして保存され、サンプルごとの予測、ゴールド回答、予測リレーション、レイテンシ、エラー情報を含む

**モデル:**
- LLM: GPT-4o-mini（`gpt-4o-mini`）を全LLM呼び出し（エンティティ抽出、クエリ生成、リランキング）に使用
- 埋め込み: OpenAI `text-embedding-3-small` を全埋め込み計算に使用
- Temperature: エンティティ抽出とパス選択で決定的出力を得るため0に設定

**実行環境:**
- パイプラインはDockerコンテナ（`python-primekgqa-experiment`）内で実行
- Neo4jインスタンスは別個のDockerコンテナとして稼働
- LLMおよび埋め込み呼び出しはOpenAI API経由

### 5.3 評価データセット

**MetaQA:**

| クエリタイプ | サンプル数 | 説明 |
|---|---|---|
| 1-hop | 200 | 単一リレーション走査（例: 映画の監督） |
| 2-hop | 200 | 2段階走査（例: ある俳優が出演した映画の監督） |
| 3-hop | 200 | 3段階走査（例: Xの脚本家が書いた映画に出演した俳優） |

**PrimeKGQA-Struct:**

| クエリタイプ | グラフパターン | 総プール | 評価サンプル | 説明 |
|---|---|---|---|---|
| 1-hop | N2 | 8,331 | 100（ランダム） | 単一リレーション（例: 薬の標的） |
| 2-hop | N3線形 | 10,414 | 100（ランダム） | 2リレーション（例: 薬→遺伝子→疾患） |
| 2-intersection | N3星型 | 4,618 | 100（ランダム） | 2アンカーintersection（例: 共通標的） |
| 3-intersection | N4_1 | 607 | 100（ランダム） | 3アンカーintersection（例: 共通パスウェイ） |

PrimeKGQA-Structでは、再現性のためランダムシードを固定し、クエリタイプごとに $n = 100$ 件の質問をサンプリングする。大規模プール（総計23,970件）により、重複なく繰り返しサンプリングが可能である。

---

## 6. 結果

### 6.1 MetaQA結果

| パイプライン | 1-hop Acc | 2-hop Acc | 3-hop Acc |
|---|---|---|---|
| ETK | 91% | **95%** | **96%** |
| SAFE | **100%** | 90% | -- |
| KGT | 90% | 58% | 7% |

| パイプライン | 1-hop F1 | 2-hop F1 | 3-hop F1 |
|---|---|---|---|
| ETK | 87.0% | **92.8%** | **89.6%** |
| SAFE | **100%** | 81.2% | -- |
| KGT | 93.0% | 51.9% | 13.4% |

### 6.2 PrimeKGQA-Struct結果

| パイプライン | 1-hop Acc | 2-hop Acc | 2-int Acc | 3-int Acc |
|---|---|---|---|---|
| ETK | **92%** | **65%** | **35%** | TBD |
| SAFE | TBD | TBD | N/A | N/A |
| KGT | TBD | TBD | TBD | TBD |

*注: SAFEはintersectionクエリ（N3星型、N4_1）に非対応。全結果は追って完成予定。*

---

## 7. 考察

### 7.1 クエリ複雑度の影響

両データセットにおいて、クエリ複雑度の増加に伴い精度が低下する。MetaQAではKGTが90%（1-hop）から7%（3-hop）に急落する一方、ETKは3-hopでも96%を維持する。これは、KGTの最短パスのみのスキーマ探索がマルチホップ推論には不十分であり、スキーマグラフ上の最短パスが必ずしも最適パスではないことを示唆している。

### 7.2 スキーマパス選択のボトルネック

パス精度と回答精度の乖離は、スキーマパス選択の誤りが主要な失敗モードであることを明らかにする。MetaQA 2-hopでETKは95%の回答精度を達成するがパス精度は59%にとどまり、誤ったパスでも双方向Cypherトラバーサルにより正しい回答が回復できることを示している。

### 7.3 埋め込みベース vs 構造的パス選択

SAFEのL2距離ベースアプローチはMetaQA 1-hopで完全な精度を達成するが、2-hopでは90%に低下する。ETKのハイブリッドアプローチ（コサイン類似度 + ホップごとヒントブースト）はより安定したマルチホップ性能を提供する。これは、グローバルな質問-パス類似度とローカルなホップごとヒントマッチングの組み合わせが、純粋な距離ベースマッチングよりも複雑なクエリに対してロバストであることを示唆している。

### 7.4 Intersectionクエリとグラフパターン対応

PrimeKGQA-Structのintersectionクエリ（N3星型、N4_1パターン）はMetaQAには存在しない能力をテストする。ETKのKoPL論理演算（集合intersection）はN3星型（2-intersection）で35%、N4_1（3-intersection）ではTBDを達成する。これらのマルチアンカーパターンはKGQAシステムにとって挑戦的なフロンティアである。

現在の3パイプラインはいずれも**線形パス**（N2、N3線形）を主な対象として設計されている。N3星型パターンに対してはETKのみがKoPLの`intersection`操作で対応しているが、N4_2（線形+分岐）のような複合パターンへの拡張は今後の課題である。N4_2パターンを扱うためには、パス定義を分岐条件（例: $A \to B \to X$ かつ $C \to X$）として表現する拡張が必要であり、単純なトラバーサルの枠組みを超えた手法の開発が求められる。

### 7.5 データセット品質の影響

PrimeKGQA-Structと既存PrimeKGQAの品質差はKGQA評価における重要な教訓を示す。既存PrimeKGQAでは一貫性スコア0.45、Fleiss' Kappa 0.12〜0.17という低品質のため、パイプラインが正しいパスを選択しても不当に低い精度が報告される可能性がある。Answer-Firstアプローチにより構造的正確性を保証したPrimeKGQA-Structは、パイプラインの真の性能をより正確に反映し、手法間の公平な比較を可能にする。

---

## 8. 結論

（省略）

---

## 参考文献

- Chandak, P., et al. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10(1), 67.
- Zhang, Y., et al. (2022). MetaQA: Question Answering over Knowledge Graphs with Multi-hop Reasoning.
- （KGT, SAFE の参考文献を追加予定）
