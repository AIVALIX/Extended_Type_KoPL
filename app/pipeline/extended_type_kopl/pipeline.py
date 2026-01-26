"""
Extended Type-KoPL Pipeline

ハイブリッドアプローチ:
1. 擬似クエリ生成 (Pseudo Query Generation) - LLMでType-KoPLプログラム生成
2. ハイブリッド探索 (Hybrid Schema Search) - Global BFS + Step-wise BFS
3. ベクトル剪定 (Vector-based Pruning) - Top-K選択
4. データ取得 (Data Retrieval) - Cypher実行
5. KoPL論理演算 (Logical Operation) - Intersection/Union/Exclude
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from core.config import BASEMODEL, get_settings
from database.search import GraphPathFinder
from pipeline.extended_type_kopl.reranker import create_reranker, BaseReranker


# =============================================================================
# Data Classes
# =============================================================================

class OperationType(str, Enum):
    """KoPL操作タイプ"""
    RELATE = "relate"
    INTERSECTION = "intersection"
    UNION = "union"
    EXCLUDE = "exclude"


@dataclass
class TypeRelation:
    """Type間のリレーション"""
    src_type: str
    tgt_type: str
    intermediate_type: Optional[str] = None  # 中間Type（2-hop用）
    intermediate_type2: Optional[str] = None  # 2番目の中間Type（3-hop用）
    relation_hints: List[str] = field(default_factory=list)  # 推測されるリレーション名


@dataclass
class KoPLOperation:
    """KoPL操作"""
    op_type: OperationType
    relations: List[TypeRelation] = field(default_factory=list)
    children: List["KoPLOperation"] = field(default_factory=list)
    anchor_name: Optional[str] = None  # アンカーエンティティ名


# リレーション名を自然言語に変換するマッピング
RELATION_TO_NATURAL_LANGUAGE = {
    # MetaQA
    "DIRECTED_BY": "directed by",
    "STARRED_ACTORS": "starred by actors",
    "WRITTEN_BY": "written by",
    "IN_LANGUAGE": "in language",
    "HAS_GENRE": "has genre",
    "HAS_TAGS": "has tags",
    "RELEASE_YEAR": "released in year",
    "HAS_IMDB_RATING": "has rating",
    "HAS_IMDB_VOTES": "has votes",
    # PrimeKGQA
    "target": "targets gene",
    "indication": "treats disease",
    "contraindication": "contraindicated for",
    "off_label_use": "off-label use for",
    "side_effect": "causes side effect",
    "associated_with": "associated with gene",
    "associated_disease": "associated with disease",
    "phenotype_present": "shows phenotype",
    "phenotype_absent": "lacks phenotype",
    "ppi": "interacts with protein",
    "carrier": "carried by",
    "enzyme": "metabolized by enzyme",
    "transporter": "transported by",
    "expression_present": "expressed in",
    "expression_absent": "not expressed in",
    "interacts_with": "interacts with",
    "interacted_by": "interacted by",
    "parent_child": "parent of",
    "linked_to": "linked to",
    "linked_exposure": "linked to exposure",
    "treated_by": "treated by",
    "targeted_by": "targeted by",
    "caused_by_drug": "caused by drug",
    "disease_with_phenotype": "disease with phenotype",
    "disease_without_phenotype": "disease without phenotype",
    "synergistic_interaction": "synergistic with",
    "metabolized_by": "metabolized by",
    "transported_by": "transported by",
    "carrier_for": "carrier for",
    "absent_gene": "absent gene",
    "expressed_gene": "expresses gene",
}


@dataclass
class SchemaPath:
    """スキーマパス"""
    types: List[str]  # [src_type, ..., tgt_type]
    relations: List[str]  # [rel1, rel2, ...]
    source: str  # "global" or "stepwise"
    score: float = 0.0
    directions: List[str] = field(default_factory=list)  # ["->", "<-", ...] for each relation

    def to_text(self) -> str:
        """構造化テキスト（デバッグ用）"""
        parts = []
        for i, t in enumerate(self.types):
            parts.append(t)
            if i < len(self.relations):
                direction = self.directions[i] if i < len(self.directions) else "->"
                parts.append(f"-[{self.relations[i]}]{direction}")
        return " ".join(parts)

    def to_natural_language(self) -> str:
        """自然言語テキスト（ベクトル剪定用）"""
        if not self.relations:
            return ""

        # リレーションを自然言語に変換
        nl_relations = []
        for rel in self.relations:
            nl = RELATION_TO_NATURAL_LANGUAGE.get(rel, rel.lower().replace("_", " "))
            nl_relations.append(nl)

        # テンプレートに当てはめる
        if len(nl_relations) == 1:
            return f"find entities {nl_relations[0]}"
        elif len(nl_relations) == 2:
            return f"find entities {nl_relations[0]}, then {nl_relations[1]}"
        elif len(nl_relations) == 3:
            # 3-hop: "share X" パターンの検出
            if nl_relations[0] == nl_relations[1]:
                return f"find entities that share {nl_relations[0]}, then {nl_relations[2]}"
            else:
                return f"find entities {nl_relations[0]}, then {nl_relations[1]}, then {nl_relations[2]}"
        else:
            return ", then ".join(nl_relations)


@dataclass
class EntitySet:
    """エンティティ集合（KoPL演算用）"""
    entities: Set[str]
    source_path: Optional[SchemaPath] = None
    anchor_name: Optional[str] = None  # どのアンカーから取得したか


@dataclass
class ExtendedTypeKoPLResult:
    """Extended Type-KoPL結果"""
    question: str
    kopl_program: Optional[KoPLOperation]
    candidate_paths: List[SchemaPath]
    selected_paths: List[SchemaPath]
    entity_sets: List[EntitySet]
    answer_entities: List[str]
    processing_log: List[str] = field(default_factory=list)


# =============================================================================
# Pydantic Models for LLM
# =============================================================================

class AtomicOperationSchema(BaseModel):
    """Atomic操作（LLM出力用）- 1ホップ分の操作"""
    operation: str = Field(default="relate", description="Operation type: relate")
    src_type: str = Field(description="Source node type for this hop")
    tgt_type: str = Field(description="Target node type for this hop")
    relation: str = Field(description="Relation name for this hop")
    anchor_name: Optional[str] = Field(default=None, description="Anchor entity name (only for the first operation)")


class AtomicKoPLProgramSchema(BaseModel):
    """Atomic Type-KoPLプログラム（LLM出力用）"""
    operations: List[AtomicOperationSchema] = Field(
        description="List of atomic operations, one per hop. For 3-hop queries, provide exactly 3 operations."
    )
    final_operation: str = Field(
        default="relate",
        description="Final operation to combine results: intersection, union, or relate"
    )


# 後方互換性のため残す
class TypeRelationSchema(BaseModel):
    """Type間のリレーション（LLM出力用）- 旧形式"""
    src_type: str = Field(description="Source node type")
    tgt_type: str = Field(description="Target node type")
    intermediate_type: Optional[str] = Field(default=None, description="Intermediate type for 2-hop")
    intermediate_type2: Optional[str] = Field(default=None, description="Second intermediate type for 3-hop")
    relation_hints: List[str] = Field(default=[], description="Predicted relation names in order (e.g., ['rel1', 'rel2'])")


class KoPLOperationSchema(BaseModel):
    """KoPL操作（LLM出力用）- 旧形式"""
    operation: str = Field(description="Operation type: relate, intersection, union, exclude")
    relations: List[TypeRelationSchema] = Field(default=[], description="Type relations for relate operation")
    anchor_name: Optional[str] = Field(default=None, description="Anchor entity name if known")


class TypeKoPLProgramSchema(BaseModel):
    """Type-KoPLプログラム（LLM出力用）- 旧形式"""
    operations: List[KoPLOperationSchema] = Field(
        default=[],
        description="List of KoPL operations"
    )
    final_operation: str = Field(
        default="relate",
        description="Final operation to combine results: intersection, union, or relate"
    )


# =============================================================================
# Schema Graph
# =============================================================================

class SchemaGraph:
    """スキーマグラフ（APSP探索用）"""

    def __init__(self):
        self.types: Set[str] = set()
        self.edges: List[Tuple[str, str, str, str]] = []  # (src, rel, tgt, direction)
        self.adjacency: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # type -> [(neighbor, rel, direction), ...]
        self.embeddings = None
        self.edge_embeddings: Dict[str, np.ndarray] = {}
        # APSP用
        self.apsp_dist: Dict[str, Dict[str, int]] = {}
        self.type_to_idx: Dict[str, int] = {}
        self.idx_to_type: Dict[int, str] = {}

    def add_edge(self, src_type: str, relation: str, tgt_type: str, direction: str = "->"):
        self.edges.append((src_type, relation, tgt_type, direction))
        self.types.add(src_type)
        self.types.add(tgt_type)
        # 有向グラフとして順方向のみ追加（方向情報も保持）
        self.adjacency[src_type].append((tgt_type, relation, direction))

    def compute_apsp(self):
        """Floyd-Warshallで全点対最短経路を計算"""
        type_list = sorted(self.types)
        n = len(type_list)

        # インデックスマッピング
        self.type_to_idx = {t: i for i, t in enumerate(type_list)}
        self.idx_to_type = {i: t for i, t in enumerate(type_list)}

        # 距離行列初期化
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]

        for i in range(n):
            dist[i][i] = 0

        # エッジの距離を1に設定
        for src, rel, tgt, direction in self.edges:
            i, j = self.type_to_idx[src], self.type_to_idx[tgt]
            dist[i][j] = 1
            dist[j][i] = 1  # 無向グラフ

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        # 辞書形式に変換
        for i in range(n):
            src = self.idx_to_type[i]
            self.apsp_dist[src] = {}
            for j in range(n):
                tgt = self.idx_to_type[j]
                self.apsp_dist[src][tgt] = dist[i][j] if dist[i][j] != INF else -1

    def get_shortest_distance(self, src_type: str, tgt_type: str) -> int:
        """2つのタイプ間の最短距離を取得"""
        if src_type not in self.apsp_dist or tgt_type not in self.apsp_dist.get(src_type, {}):
            return -1
        return self.apsp_dist[src_type][tgt_type]

    def find_shortest_paths(self, start_type: str, end_type: str) -> List[SchemaPath]:
        """start_type から end_type への最短パスのみを返す"""
        if start_type not in self.types or end_type not in self.types:
            return []

        shortest_dist = self.get_shortest_distance(start_type, end_type)
        if shortest_dist <= 0:
            return []

        # BFSで最短距離のパスのみを収集
        queue = [(start_type, [start_type], [], [])]  # (current, type_path, rel_path, dir_path)
        shortest_paths = []
        visited_at_depth = defaultdict(set)  # depth -> visited states

        while queue:
            current, type_path, rel_path, dir_path = queue.pop(0)
            current_depth = len(type_path) - 1

            # 最短距離を超えたら終了
            if current_depth > shortest_dist:
                continue

            # ゴール到達（最短距離で）
            if current == end_type and current_depth == shortest_dist:
                shortest_paths.append(SchemaPath(
                    types=type_path,
                    relations=rel_path,
                    source="apsp",
                    directions=dir_path,
                ))
                continue

            # まだゴールに到達していない場合、探索継続
            if current_depth < shortest_dist:
                for next_type, relation, direction in self.adjacency[current]:
                    # サイクル防止（ただし自己参照は1回許可）
                    if next_type not in type_path:
                        state = (next_type, tuple(type_path + [next_type]), tuple(rel_path + [relation]))
                        if state not in visited_at_depth[current_depth + 1]:
                            visited_at_depth[current_depth + 1].add(state)
                            queue.append((next_type, type_path + [next_type], rel_path + [relation], dir_path + [direction]))
                    elif next_type == current and type_path.count(next_type) < 2:
                        # 自己参照（PPI等）
                        state = (next_type, tuple(type_path + [next_type]), tuple(rel_path + [relation]))
                        if state not in visited_at_depth[current_depth + 1]:
                            visited_at_depth[current_depth + 1].add(state)
                            queue.append((next_type, type_path + [next_type], rel_path + [relation], dir_path + [direction]))

        return shortest_paths

    def compute_embeddings(self, embeddings):
        """エッジの埋め込みを計算"""
        self.embeddings = embeddings
        texts = [f"{s} {r} {t}" for s, r, t, _ in self.edges]
        if texts:
            vectors = embeddings.embed_documents(texts)
            for i, (s, r, t, _) in enumerate(self.edges):
                key = f"{s}|{r}|{t}"
                self.edge_embeddings[key] = np.array(vectors[i])

    def global_bfs(self, start_type: str, end_type: str, max_depth: int = 3) -> List[SchemaPath]:
        """Global BFS: 全パスを探索（KGT方式）"""
        if start_type not in self.types or end_type not in self.types:
            return []

        # BFSで全パスを探索
        queue = [(start_type, [start_type], [], [])]  # (current, type_path, rel_path, dir_path)
        all_paths = []
        visited_states = set()

        while queue:
            current, type_path, rel_path, dir_path = queue.pop(0)

            if len(type_path) > max_depth + 1:
                continue

            # 状態キーにrelation pathも含める（異なるrelationで同じtype pathを許容）
            state = (current, tuple(type_path), tuple(rel_path))
            if state in visited_states:
                continue
            visited_states.add(state)

            if current == end_type and len(type_path) > 1:
                all_paths.append(SchemaPath(
                    types=type_path,
                    relations=rel_path,
                    source="global",
                    directions=dir_path,
                ))
                continue  # ゴールに到達したら探索を続けない

            for next_type, relation, direction in self.adjacency[current]:
                # サイクル防止（ただし自己参照リレーション(PPI等)は1回だけ許可）
                if next_type not in type_path:
                    queue.append((
                        next_type,
                        type_path + [next_type],
                        rel_path + [relation],
                        dir_path + [direction],
                    ))
                elif next_type == current and type_path.count(next_type) < 2:
                    # 自己参照（例: gene/protein -> gene/protein via ppi）を許可
                    # ただし同じタイプが2回以上出現するのは防止
                    queue.append((
                        next_type,
                        type_path + [next_type],
                        rel_path + [relation],
                        dir_path + [direction],
                    ))

        return all_paths

    def find_stepwise_shortest_paths(
        self,
        start_type: str,
        intermediate_type: str,
        end_type: str,
    ) -> List[SchemaPath]:
        """Step-wise APSP: 中間Typeを経由する最短パスを探索"""
        # Step 1: start -> intermediate の最短パス
        paths_1 = self.find_shortest_paths(start_type, intermediate_type)

        # Step 2: intermediate -> end の最短パス
        paths_2 = self.find_shortest_paths(intermediate_type, end_type)

        # Join: パスを結合
        combined_paths = []
        for p1 in paths_1:
            for p2 in paths_2:
                # 中間Typeで結合（重複を除去）
                combined_types = p1.types + p2.types[1:]
                combined_rels = p1.relations + p2.relations
                combined_dirs = p1.directions + p2.directions
                combined_paths.append(SchemaPath(
                    types=combined_types,
                    relations=combined_rels,
                    source="stepwise",
                    directions=combined_dirs,
                ))

        return combined_paths


def build_schema_graph(kg_type: str = "primekgqa") -> SchemaGraph:
    """スキーマグラフを構築

    Args:
        kg_type: "primekgqa" or "metaqa"
    """
    if kg_type == "metaqa":
        from dataset_construction.schema_metaqa import SCHEMA_GRAPH
    else:
        from dataset_construction.schema_v2 import SCHEMA_GRAPH

    schema = SchemaGraph()
    for src, rel, tgt, direction in SCHEMA_GRAPH:
        schema.add_edge(src, rel, tgt, direction)

    # APSPを計算
    schema.compute_apsp()

    return schema


# =============================================================================
# Extended Type-KoPL Pipeline
# =============================================================================

class ExtendedTypeKoPLPipeline:
    """Extended Type-KoPL パイプライン"""

    # KGタイプごとのエンティティタイプリスト
    ENTITY_TYPES = {
        "primekgqa": [
            "drug", "disease", "gene/protein", "exposure", "biological_process",
            "molecular_function", "cellular_component", "pathway", "anatomy", "effect/phenotype"
        ],
        "metaqa": [
            "movie", "person", "organization", "text", "date", "language", "number"
        ],
    }

    def __init__(
        self,
        model: str = BASEMODEL,
        embedding_model: str = "text-embedding-3-small",
        top_k_paths: int = 3,
        max_depth: int = 3,
        kg_type: str = "primekgqa",
        reranker_type: str = "none",  # "none", "llm", "hybrid"
        reranker_input_k: int = 10,  # Rerankerに渡す候補数
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings

        self.llm = init_chat_model(model, model_provider="openai", temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.kg_type = kg_type
        self.schema = build_schema_graph(kg_type)
        self.schema.compute_embeddings(self.embeddings)

        self.finder = GraphPathFinder(kg_type=kg_type)

        self.top_k_paths = top_k_paths
        self.max_depth = max_depth
        self.max_candidate_paths = 50  # 候補パスの最大数を制限

        # Reranker設定
        self.reranker_type = reranker_type
        self.reranker_input_k = reranker_input_k
        self.reranker = create_reranker(reranker_type, model=model, kg_type=kg_type)

    def _extract_relation_hints(self, kopl_program: KoPLOperation) -> Optional[List[str]]:
        """KoPLプログラムからrelation_hintsを抽出"""
        operations = kopl_program.children if kopl_program.children else [kopl_program]
        for op in operations:
            for rel in op.relations:
                if rel.relation_hints:
                    return rel.relation_hints
        return None

    def _get_entity_type(self, entity_name: str) -> Optional[str]:
        """KGからエンティティのタイプを取得"""
        if not entity_name:
            return None
        try:
            cypher = """
            MATCH (n)
            WHERE n.name = $name
            RETURN labels(n) AS labels
            LIMIT 1
            """
            records = self.finder.graph.run(cypher, name=entity_name).data()
            if records and records[0]["labels"]:
                labels = records[0]["labels"]
                # KGタイプに応じたエンティティタイプリストを使用
                valid_types = self.ENTITY_TYPES.get(self.kg_type, [])
                for lbl in labels:
                    if lbl.lower() in valid_types:
                        return lbl.lower()
                return labels[0].lower() if labels else None
        except Exception as e:
            print(f"Entity type lookup error: {e}")
        return None

    def run(self, question: str, entity_name: Optional[str] = None) -> ExtendedTypeKoPLResult:
        """パイプライン実行"""
        log = []
        log.append(f"Question: {question}")

        # エンティティタイプを取得
        entity_type = self._get_entity_type(entity_name) if entity_name else None

        # Phase 1: 擬似クエリ生成
        log.append("Phase 1: Pseudo Query Generation")
        kopl_program = self._generate_type_kopl(question, entity_name, entity_type)
        relation_hints = None
        if kopl_program:
            log.append(f"  Generated KoPL program with {len(kopl_program.relations)} relations")
            if kopl_program.children:
                log.append(f"  Final operation: {kopl_program.op_type}")
            # relation_hints を抽出
            relation_hints = self._extract_relation_hints(kopl_program)
            if relation_hints:
                log.append(f"  Relation hints: {relation_hints}")
        else:
            log.append("  Failed to generate KoPL program")
            return ExtendedTypeKoPLResult(
                question=question,
                kopl_program=None,
                candidate_paths=[],
                selected_paths=[],
                entity_sets=[],
                answer_entities=[],
                processing_log=log,
            )

        # Phase 2: ハイブリッド探索
        log.append("Phase 2: Hybrid Schema Search")
        candidate_paths = self._hybrid_schema_search(kopl_program)
        log.append(f"  Found {len(candidate_paths)} candidate paths")
        for p in candidate_paths[:5]:
            log.append(f"    [{p.source}] {p.to_text()}")

        if not candidate_paths:
            return ExtendedTypeKoPLResult(
                question=question,
                kopl_program=kopl_program,
                candidate_paths=[],
                selected_paths=[],
                entity_sets=[],
                answer_entities=[],
                processing_log=log,
            )

        # Phase 3: ベクトル剪定（リレーションヒント対応）
        log.append("Phase 3: Vector-based Pruning (with relation hints)" if relation_hints else "Phase 3: Vector-based Pruning")
        # Reranker使用時はより多くの候補をベクトル剪定で残す
        prune_k = self.reranker_input_k if self.reranker_type != "none" else self.top_k_paths
        pruned_paths = self._vector_pruning(question, candidate_paths, top_k=prune_k, relation_hints=relation_hints)
        log.append(f"  Pruned to {len(pruned_paths)} paths")
        for p in pruned_paths[:5]:  # ログは最大5つまで
            log.append(f"    {p.to_text()} (score: {p.score:.4f})")
        if len(pruned_paths) > 5:
            log.append(f"    ... and {len(pruned_paths) - 5} more")

        # Phase 3.5: LLM Reranker (optional)
        if self.reranker_type != "none":
            log.append(f"Phase 3.5: LLM Reranker ({self.reranker_type}, input={len(pruned_paths)})")
            selected_paths = self.reranker.rerank(question, pruned_paths, top_k=1)
            log.append(f"  Reranked to {len(selected_paths)} path(s)")
            for p in selected_paths:
                log.append(f"    {p.to_text()}")
        else:
            # Rerankerなしの場合はtop1のみ使用
            selected_paths = pruned_paths[:1]

        # Phase 4: データ取得
        log.append("Phase 4: Data Retrieval")
        entity_sets = self._retrieve_entities(kopl_program, selected_paths)
        log.append(f"  Retrieved {len(entity_sets)} entity sets")
        for es in entity_sets:
            log.append(f"    Set with {len(es.entities)} entities")

        # Phase 5: KoPL論理演算
        log.append("Phase 5: KoPL Logical Operation")
        answer_entities = self._apply_kopl_operations(kopl_program, entity_sets)
        log.append(f"  Final answer: {len(answer_entities)} entities")

        return ExtendedTypeKoPLResult(
            question=question,
            kopl_program=kopl_program,
            candidate_paths=candidate_paths,
            selected_paths=selected_paths,
            entity_sets=entity_sets,
            answer_entities=answer_entities,
            processing_log=log,
        )

    def _get_available_relations(self) -> str:
        """KGタイプに応じた利用可能なリレーション一覧を返す"""
        # リレーション情報は提供しない（タイプ情報のみ）
        return ""

    def _get_prompt_examples(self) -> str:
        """KGタイプに応じたプロンプト例を返す（Atomic形式）"""
        if self.kg_type == "metaqa":
            return """Examples (Atomic format - one operation per hop):

Question: "What movies did Tom Hanks star in?" (Tom Hanks is a person)
→ 1-hop query
{{
  "operations": [
    {{"src_type": "Person", "tgt_type": "Movie", "relation": "STARRED_ACTORS", "anchor_name": "Tom Hanks"}}
  ],
  "final_operation": "relate"
}}

Question: "Who directed the movies that Tom Hanks starred in?" (Tom Hanks is a person)
→ 2-hop query: Person's movies, then directors
{{
  "operations": [
    {{"src_type": "Person", "tgt_type": "Movie", "relation": "STARRED_ACTORS", "anchor_name": "Tom Hanks"}},
    {{"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY"}}
  ],
  "final_operation": "relate"
}}

Question: "What year was Titanic released?" (Titanic is a movie)
→ 1-hop query
{{
  "operations": [
    {{"src_type": "Movie", "tgt_type": "Date", "relation": "RELEASE_YEAR", "anchor_name": "Titanic"}}
  ],
  "final_operation": "relate"
}}

Question: "What languages are the films that share directors with Titanic in?" (Titanic is a movie)
→ 3-hop query: Titanic's director → director's other movies → those movies' languages
{{
  "operations": [
    {{"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY", "anchor_name": "Titanic"}},
    {{"src_type": "Person", "tgt_type": "Movie", "relation": "DIRECTED_BY"}},
    {{"src_type": "Movie", "tgt_type": "Language", "relation": "IN_LANGUAGE"}}
  ],
  "final_operation": "relate"
}}

Question: "Who starred in movies directed by the director of Titanic?" (Titanic is a movie)
→ 3-hop query: Titanic's director → director's movies → actors in those movies
{{
  "operations": [
    {{"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY", "anchor_name": "Titanic"}},
    {{"src_type": "Person", "tgt_type": "Movie", "relation": "DIRECTED_BY"}},
    {{"src_type": "Movie", "tgt_type": "Person", "relation": "STARRED_ACTORS"}}
  ],
  "final_operation": "relate"
}}

Question: "who are the directors of movies whose actors also appear in Candleshoe" (Candleshoe is a movie)
→ 3-hop query: Candleshoe's actors → actors' other movies → those movies' directors
{{
  "operations": [
    {{"src_type": "Movie", "tgt_type": "Person", "relation": "STARRED_ACTORS", "anchor_name": "Candleshoe"}},
    {{"src_type": "Person", "tgt_type": "Movie", "relation": "STARRED_ACTORS"}},
    {{"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY"}}
  ],
  "final_operation": "relate"
}}

Question: "what are the release years of films written by writers of Four Daughters" (Four Daughters is a movie)
→ 3-hop query: Four Daughters' writers → writers' other movies → release years
{{
  "operations": [
    {{"src_type": "Movie", "tgt_type": "Person", "relation": "WRITTEN_BY", "anchor_name": "Four Daughters"}},
    {{"src_type": "Person", "tgt_type": "Movie", "relation": "WRITTEN_BY"}},
    {{"src_type": "Movie", "tgt_type": "Date", "relation": "RELEASE_YEAR"}}
  ],
  "final_operation": "relate"
}}"""
        else:
            return """Examples:

Question: "What diseases are associated with BRCA1?" (BRCA1 is a gene)
→ Anchor type: gene/protein, Target: disease
operations: [{{operation: "relate", relations: [{{src_type: "gene/protein", tgt_type: "disease", relation_hints: ["associated_with"]}}], anchor_name: "BRCA1"}}]

Question: "Which diseases are linked to genes targeted by Tacrolimus?" (Tacrolimus is a drug)
→ Anchor type: drug, Target: disease, Via: gene/protein
operations: [{{operation: "relate", relations: [{{src_type: "drug", tgt_type: "disease", intermediate_type: "gene/protein", relation_hints: ["target", "associated_with"]}}], anchor_name: "Tacrolimus"}}]

Question: "Which drugs target genes linked to heart failure?" (heart failure is a disease)
→ Anchor type: disease, Target: drug, Via: gene/protein
operations: [{{operation: "relate", relations: [{{src_type: "disease", tgt_type: "drug", intermediate_type: "gene/protein", relation_hints: ["associated_with", "target"]}}], anchor_name: "heart failure"}}]

Question: "Which genes are targeted by both DrugA and DrugB?"
→ INTERSECTION query with two anchors:
operations: [
  {{operation: "relate", relations: [{{src_type: "drug", tgt_type: "gene/protein", relation_hints: ["target"]}}], anchor_name: "DrugA"}},
  {{operation: "relate", relations: [{{src_type: "drug", tgt_type: "gene/protein", relation_hints: ["target"]}}], anchor_name: "DrugB"}}
]
final_operation: "intersection\""""

    def _generate_type_kopl(
        self,
        question: str,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None
    ) -> Optional[KoPLOperation]:
        """Phase 1: LLMでType-KoPLプログラムを生成（Atomic形式）"""

        type_list = ", ".join(sorted(self.schema.types))
        available_relations = self._get_available_relations()

        entity_info = ""
        if entity_name:
            entity_info = f"Known entity: {entity_name}"
            if entity_type:
                entity_info += f" (type: {entity_type})"

        examples = self._get_prompt_examples()

        prompt = f"""Convert this question into an Atomic Type-KoPL program.

Question: {question}
{entity_info}

Available node types: {type_list}

{available_relations}

CRITICAL RULES:
1. Each operation represents ONE HOP in the path
2. For N-hop queries, provide exactly N operations
3. The first operation MUST have anchor_name (the known entity)
4. Each operation specifies: src_type, tgt_type, relation
5. Operations are connected: op[i].tgt_type == op[i+1].src_type

IMPORTANT:
- 1-hop query: 1 operation
- 2-hop query: 2 operations
- 3-hop query: 3 operations (most common for "share X" patterns)

{examples}

Return a JSON object with operations array and final_operation."""

        llm_with_output = self.llm.with_structured_output(AtomicKoPLProgramSchema)

        try:
            result = llm_with_output.invoke(prompt)

            # 有効なタイプのセット（名寄せ用にlower -> 正式名のマッピングを作成）
            valid_types = self.schema.types
            type_normalizer = {t.lower(): t for t in valid_types}

            def normalize_type(t: Optional[str]) -> Optional[str]:
                """タイプ名を正規化（case-insensitive マッチング）"""
                if not t:
                    return None
                t_clean = t.strip().rstrip("}],")
                # 完全一致
                if t_clean in valid_types:
                    return t_clean
                # case-insensitive マッチング
                t_lower = t_clean.lower()
                if t_lower in type_normalizer:
                    return type_normalizer[t_lower]
                return None

            # Atomic操作からrelation_hintsを抽出
            relation_hints = []
            anchor_name = None
            src_type = None
            tgt_type = None

            for i, op in enumerate(result.operations):
                # タイプ名を正規化
                op_src = normalize_type(op.src_type)
                op_tgt = normalize_type(op.tgt_type)
                op_rel = op.relation.strip() if op.relation else None

                if not op_src or not op_tgt or not op_rel:
                    continue

                # 最初の操作からanchorとsrc_typeを取得
                if i == 0:
                    anchor_name = op.anchor_name
                    src_type = op_src

                # 最後の操作からtgt_typeを取得
                tgt_type = op_tgt

                # relationを収集
                relation_hints.append(op_rel)

            if not relation_hints:
                raise ValueError("No valid operations generated")

            # intermediate_typeを決定
            intermediate_type = None
            intermediate_type2 = None
            if len(result.operations) >= 2:
                intermediate_type = normalize_type(result.operations[0].tgt_type)
            if len(result.operations) >= 3:
                intermediate_type2 = normalize_type(result.operations[1].tgt_type)

            # 単一のKoPLOperationとしてまとめる
            return KoPLOperation(
                op_type=OperationType.RELATE,
                relations=[TypeRelation(
                    src_type=src_type or "",
                    tgt_type=tgt_type or "",
                    intermediate_type=intermediate_type,
                    intermediate_type2=intermediate_type2,
                    relation_hints=relation_hints,
                )],
                anchor_name=anchor_name,
            )

        except Exception as e:
            print(f"Error generating Type-KoPL: {e}")
            # フォールバック
            if entity_name:
                return KoPLOperation(
                    op_type=OperationType.RELATE,
                    relations=[TypeRelation(src_type="", tgt_type="")],
                    anchor_name=entity_name,
                )
            return None

    def _hybrid_schema_search(self, kopl_program: KoPLOperation) -> List[SchemaPath]:
        """Phase 2: ハイブリッド探索（APSPベース）

        各操作のsrc_type → tgt_type に対して:
        - intermediate_type がない場合: Global APSP（最短パス）
        - intermediate_type がある場合: Step-wise APSP + Global APSP を両方取得
        - intermediate_type2 がある場合: 3-hop Step-wise APSP
        """

        all_paths = []

        # 子操作がある場合は各子操作に対して探索
        operations = kopl_program.children if kopl_program.children else [kopl_program]

        for op in operations:
            for rel in op.relations:
                if not rel.src_type or not rel.tgt_type:
                    continue

                if rel.intermediate_type2:
                    # 3-hop: src -> inter1 -> inter2 -> tgt
                    # Step 1: src -> inter1
                    paths_1 = self.schema.find_shortest_paths(rel.src_type, rel.intermediate_type)
                    # Step 2: inter1 -> inter2
                    paths_2 = self.schema.find_shortest_paths(rel.intermediate_type, rel.intermediate_type2)
                    # Step 3: inter2 -> tgt
                    paths_3 = self.schema.find_shortest_paths(rel.intermediate_type2, rel.tgt_type)

                    # 3つのパスを結合
                    for p1 in paths_1:
                        for p2 in paths_2:
                            for p3 in paths_3:
                                combined_types = p1.types + p2.types[1:] + p3.types[1:]
                                combined_rels = p1.relations + p2.relations + p3.relations
                                combined_dirs = p1.directions + p2.directions + p3.directions
                                all_paths.append(SchemaPath(
                                    types=combined_types,
                                    relations=combined_rels,
                                    source="stepwise-3hop",
                                    directions=combined_dirs,
                                ))

                elif rel.intermediate_type:
                    # Route A: Step-wise APSP（中間Typeを経由する最短パス）
                    stepwise_paths = self.schema.find_stepwise_shortest_paths(
                        rel.src_type,
                        rel.intermediate_type,
                        rel.tgt_type,
                    )
                    all_paths.extend(stepwise_paths)

                    # Route B: Global APSP（直接の最短パス）
                    global_paths = self.schema.find_shortest_paths(
                        rel.src_type,
                        rel.tgt_type,
                    )
                    all_paths.extend(global_paths)
                else:
                    # Global APSP のみ（最短パス）
                    global_paths = self.schema.find_shortest_paths(
                        rel.src_type,
                        rel.tgt_type,
                    )
                    all_paths.extend(global_paths)

        # 重複除去
        seen = set()
        unique_paths = []
        for p in all_paths:
            key = (tuple(p.types), tuple(p.relations))
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        return unique_paths

    def _vector_pruning(
        self,
        question: str,
        candidate_paths: List[SchemaPath],
        top_k: Optional[int] = None,
        relation_hints: Optional[List[str]] = None,
    ) -> List[SchemaPath]:
        """Phase 3: ベクトル類似度でTop-K選択（リレーションヒント対応）

        Args:
            question: 質問文
            candidate_paths: 候補パス
            top_k: 選択するパス数
            relation_hints: LLMが予測したリレーション名のリスト
        """

        if not candidate_paths:
            return []

        if top_k is None:
            top_k = self.top_k_paths

        # リレーションヒントがある場合、ヒントベースのスコアリングを使用
        if relation_hints:
            # ヒントのリレーションをテキスト化
            hint_text = " ".join(relation_hints)
            hint_vec = np.array(self.embeddings.embed_query(hint_text))

            # パスのリレーションをテキスト化してベクトル化
            path_rel_texts = [" ".join(path.relations) for path in candidate_paths]
            path_rel_vecs = self.embeddings.embed_documents(path_rel_texts)

            # 各パスのスコアを計算（リレーションヒントとの類似度）
            for i, path in enumerate(candidate_paths):
                path_vec = np.array(path_rel_vecs[i])
                # コサイン類似度
                similarity = np.dot(hint_vec, path_vec) / (
                    np.linalg.norm(hint_vec) * np.linalg.norm(path_vec) + 1e-8
                )
                path.score = similarity
        else:
            # フォールバック: 従来の質問ベースのベクトル剪定
            question_vec = np.array(self.embeddings.embed_query(question))

            # パステキストをバッチでベクトル化（効率化）
            # 自然言語表現を使用してベクトル剪定の精度を向上
            path_texts = [path.to_natural_language() for path in candidate_paths]
            path_vecs = self.embeddings.embed_documents(path_texts)

            # 各パスのスコアを計算
            for i, path in enumerate(candidate_paths):
                path_vec = np.array(path_vecs[i])
                # コサイン類似度
                similarity = np.dot(question_vec, path_vec) / (
                    np.linalg.norm(question_vec) * np.linalg.norm(path_vec) + 1e-8
                )
                path.score = similarity

        # スコア順にソートしてTop-K
        candidate_paths.sort(key=lambda p: p.score, reverse=True)
        return candidate_paths[:top_k]

    def _retrieve_entities(
        self,
        kopl_program: KoPLOperation,
        selected_paths: List[SchemaPath]
    ) -> List[EntitySet]:
        """Phase 4: Cypherでエンティティ取得

        APSPにより選択されたパスはすでに適切な長さなので、
        各アンカーに対して全選択パスを実行する。
        """

        graph = self.finder.graph
        entity_sets = []

        operations = kopl_program.children if kopl_program.children else [kopl_program]

        for op in operations:
            anchor_name = op.anchor_name

            if not anchor_name:
                continue

            # 各パスに対してCypher実行
            for path in selected_paths:
                entities = self._execute_cypher_for_path(graph, path, anchor_name)
                if entities:
                    entity_sets.append(EntitySet(
                        entities=entities,
                        source_path=path,
                        anchor_name=anchor_name,
                    ))

        return entity_sets

    def _execute_cypher_for_path(
        self,
        graph,
        path: SchemaPath,
        anchor_name: str
    ) -> Set[str]:
        """パスに対してCypherを実行（方向を考慮）"""

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        def get_direction(idx: int) -> str:
            """パスのidx番目のエッジの方向を取得"""
            if path.directions and idx < len(path.directions):
                return path.directions[idx]
            return "->"  # デフォルトは順方向

        if len(path.types) < 2:
            return set()

        # パスの長さに応じてCypherを構築（方向を考慮）
        if len(path.types) == 2:
            # 1-hop
            direction = get_direction(0)
            if direction == "<-":
                # Reverse: 実際のエッジは (tgt)-[r]->(src) なので、アンカーを右側に
                cypher = f"""
                MATCH (b:{get_label(path.types[1])})-[r:{path.relations[0]}]->(a:{get_label(path.types[0])})
                WHERE a.name = $anchor_name
                RETURN DISTINCT b.name AS answer
                """
            else:
                # Forward: (src)-[r]->(tgt)
                cypher = f"""
                MATCH (a:{get_label(path.types[0])})-[r:{path.relations[0]}]->(b:{get_label(path.types[1])})
                WHERE a.name = $anchor_name
                RETURN DISTINCT b.name AS answer
                """
        elif len(path.types) == 3:
            # 2-hop: 各エッジの方向を個別にチェック
            dir1 = get_direction(0)
            dir2 = get_direction(1)

            # パターンを構築
            # Forward: (a)-[r]->(b)
            # Reverse: (b)-[r]->(a) ... つまり逆向きにマッチ
            if dir1 == "<-" and dir2 == "<-":
                # Both reverse: (b)->(mid)->(a)
                cypher = f"""
                MATCH (b:{get_label(path.types[2])})-[r2:{path.relations[1]}]->(mid:{get_label(path.types[1])})-[r1:{path.relations[0]}]->(a:{get_label(path.types[0])})
                WHERE a.name = $anchor_name AND a <> mid AND mid <> b AND a <> b
                RETURN DISTINCT b.name AS answer
                """
            elif dir1 == "<-" and dir2 == "->":
                # First reverse, second forward: (mid)->(a), (mid)->(b)
                cypher = f"""
                MATCH (mid:{get_label(path.types[1])})-[r1:{path.relations[0]}]->(a:{get_label(path.types[0])})
                MATCH (mid)-[r2:{path.relations[1]}]->(b:{get_label(path.types[2])})
                WHERE a.name = $anchor_name AND a <> mid AND mid <> b AND a <> b
                RETURN DISTINCT b.name AS answer
                """
            elif dir1 == "->" and dir2 == "<-":
                # First forward, second reverse: (a)->(mid), (b)->(mid)
                cypher = f"""
                MATCH (a:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(mid:{get_label(path.types[1])})
                MATCH (b:{get_label(path.types[2])})-[r2:{path.relations[1]}]->(mid)
                WHERE a.name = $anchor_name AND a <> mid AND mid <> b AND a <> b
                RETURN DISTINCT b.name AS answer
                """
            else:
                # Both forward: (a)->(mid)->(b)
                cypher = f"""
                MATCH (a:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(mid:{get_label(path.types[1])})-[r2:{path.relations[1]}]->(b:{get_label(path.types[2])})
                WHERE a.name = $anchor_name AND a <> mid AND mid <> b AND a <> b
                RETURN DISTINCT b.name AS answer
                """
        elif len(path.types) == 4:
            # 3-hop: 各エッジの方向を個別にチェック
            dir1 = get_direction(0)
            dir2 = get_direction(1)
            dir3 = get_direction(2)

            # パターンを動的に構築
            # 基本形: (a)-[r1]->(m1)-[r2]->(m2)-[r3]->(b)
            # 方向に応じて各セグメントを調整

            def build_segment(from_node: str, to_node: str, rel: str, direction: str, from_type: str, to_type: str) -> str:
                if direction == "<-":
                    return f"({to_node}:{get_label(to_type)})-[:{rel}]->({from_node}:{get_label(from_type)})"
                else:
                    return f"({from_node}:{get_label(from_type)})-[:{rel}]->({to_node}:{get_label(to_type)})"

            # 各方向の組み合わせに応じてCypherを構築
            # 簡略化: MATCHを複数使用して各セグメントを接続
            if dir1 == "->" and dir2 == "<-" and dir3 == "->":
                # Most common for "share X" queries: (a)->(m1)<-(m2)->(b)
                cypher = f"""
                MATCH (a:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(m1:{get_label(path.types[1])})
                MATCH (m2:{get_label(path.types[2])})-[r2:{path.relations[1]}]->(m1)
                MATCH (m2)-[r3:{path.relations[2]}]->(b:{get_label(path.types[3])})
                WHERE a.name = $anchor_name AND a <> m1 AND m1 <> m2 AND m2 <> b
                RETURN DISTINCT b.name AS answer
                """
            elif dir1 == "<-" and dir2 == "->" and dir3 == "->":
                # (m1)->(a), (m1)->(m2)->(b)
                cypher = f"""
                MATCH (m1:{get_label(path.types[1])})-[r1:{path.relations[0]}]->(a:{get_label(path.types[0])})
                MATCH (m1)-[r2:{path.relations[1]}]->(m2:{get_label(path.types[2])})
                MATCH (m2)-[r3:{path.relations[2]}]->(b:{get_label(path.types[3])})
                WHERE a.name = $anchor_name AND a <> m1 AND m1 <> m2 AND m2 <> b
                RETURN DISTINCT b.name AS answer
                """
            elif dir1 == "->" and dir2 == "->" and dir3 == "->":
                # All forward: (a)->(m1)->(m2)->(b)
                cypher = f"""
                MATCH (a:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(m1:{get_label(path.types[1])})-[r2:{path.relations[1]}]->(m2:{get_label(path.types[2])})-[r3:{path.relations[2]}]->(b:{get_label(path.types[3])})
                WHERE a.name = $anchor_name AND a <> m1 AND m1 <> m2 AND m2 <> b
                RETURN DISTINCT b.name AS answer
                """
            else:
                # Other combinations - generic approach with multiple MATCH
                cypher = f"""
                MATCH (a:{get_label(path.types[0])})
                WHERE a.name = $anchor_name
                MATCH (m1:{get_label(path.types[1])})
                MATCH (m2:{get_label(path.types[2])})
                MATCH (b:{get_label(path.types[3])})
                WHERE {'(m1)-[:' + path.relations[0] + ']->(a)' if dir1 == '<-' else '(a)-[:' + path.relations[0] + ']->(m1)'}
                  AND {'(m2)-[:' + path.relations[1] + ']->(m1)' if dir2 == '<-' else '(m1)-[:' + path.relations[1] + ']->(m2)'}
                  AND {'(b)-[:' + path.relations[2] + ']->(m2)' if dir3 == '<-' else '(m2)-[:' + path.relations[2] + ']->(b)'}
                  AND a <> m1 AND m1 <> m2 AND m2 <> b
                RETURN DISTINCT b.name AS answer
                """
        else:
            # 4-hop以上は複雑になるのでスキップ
            return set()

        try:
            records = graph.run(cypher, anchor_name=anchor_name).data()
            return {r["answer"] for r in records if r["answer"]}
        except Exception as e:
            print(f"Cypher error: {e}")
            return set()

    def _apply_kopl_operations(
        self,
        kopl_program: KoPLOperation,
        entity_sets: List[EntitySet]
    ) -> List[str]:
        """Phase 5: KoPL論理演算を適用"""

        if not entity_sets:
            return []

        # アンカーごとにエンティティをグループ化（同一アンカー内はunion）
        anchor_groups: Dict[str, Set[str]] = defaultdict(set)
        for es in entity_sets:
            anchor_key = es.anchor_name or "default"
            anchor_groups[anchor_key].update(es.entities)

        # 全エンティティを集める
        all_entities = set()
        for entities in anchor_groups.values():
            all_entities.update(entities)

        # 最終操作に応じて集合演算
        if kopl_program.op_type == OperationType.INTERSECTION:
            # Intersection: 各アンカーグループに共通するもの
            groups = list(anchor_groups.values())
            if groups:
                result = groups[0].copy()
                for g in groups[1:]:
                    result = result & g
                return list(result)

        elif kopl_program.op_type == OperationType.UNION:
            # Union: いずれかのグループに含まれるもの
            return list(all_entities)

        elif kopl_program.op_type == OperationType.EXCLUDE:
            # Exclude: 最初のグループから他を除外
            groups = list(anchor_groups.values())
            if len(groups) >= 2:
                result = groups[0].copy()
                for g in groups[1:]:
                    result = result - g
                return list(result)

        # デフォルト（RELATE）: 全て返す
        return list(all_entities)


# =============================================================================
# Main
# =============================================================================

def main():
    """テスト実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Extended Type-KoPL Pipeline Test")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    pipeline = ExtendedTypeKoPLPipeline()
    result = pipeline.run(args.question, args.entity)

    print("\n" + "=" * 60)
    print("Extended Type-KoPL Pipeline Result")
    print("=" * 60)

    for log_line in result.processing_log:
        print(log_line)

    print("\n--- Answers ---")
    print(f"  {result.answer_entities[:10]}")


if __name__ == "__main__":
    main()
