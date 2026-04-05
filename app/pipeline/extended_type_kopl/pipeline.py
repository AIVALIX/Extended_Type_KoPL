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

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from core.config import BASEMODEL, LLM_API_BASE, get_settings
from database.search import GraphPathFinder
from pipeline.extended_type_kopl.kg_config import (
    ETKKGConfig,
    build_etk_kg_config,
    ALL_RELATION_NL,
    auto_generate_relation_nl,
)
from pipeline.extended_type_kopl.few_shot_pool import FewShotPool
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
    """Type間のリレーション（1ホップ分）"""

    src_type: str
    tgt_type: str
    relation_hint: Optional[str] = None  # 推測されるリレーション名（1つ）


@dataclass
class KoPLOperation:
    """KoPL操作"""

    op_type: OperationType
    relations: List[TypeRelation] = field(default_factory=list)
    children: List["KoPLOperation"] = field(default_factory=list)
    anchor_name: Optional[str] = None  # アンカーエンティティ名
    filters: Optional[List[PropertyFilterSchema]] = None  # Property filters for answer entities


# リレーション名を自然言語に変換するマッピング
# 各リレーションは (noun_form, verb_phrase) のタプルで保持
# verb_phraseは質問文のパターンにマッチしやすい形式
RELATION_TO_NATURAL_LANGUAGE = {
    # MetaQA - (noun, verb_phrase)
    "DIRECTED_BY": ("director", "directed"),
    "STARRED_ACTORS": ("actor", "starred"),
    "WRITTEN_BY": ("writer", "written"),
    "IN_LANGUAGE": ("language", "in language"),
    "HAS_GENRE": ("genre", "genre"),
    "HAS_TAGS": ("tags", "tags"),
    "RELEASE_YEAR": ("release year", "released"),
    "HAS_IMDB_RATING": ("rating", "rated"),
    "HAS_IMDB_VOTES": ("votes", "votes"),
    # PrimeKGQA - (noun, verb_phrase)
    "target": ("target gene", "targets"),
    "indication": ("indication", "treats"),
    "contraindication": ("contraindication", "contraindicated"),
    "off_label_use": ("off-label use", "off-label"),
    "side_effect": ("side effect", "causes"),
    "associated_disease": ("associated disease", "associated"),
    "phenotype_present": ("phenotype", "shows phenotype"),
    "phenotype_absent": ("absent phenotype", "lacks phenotype"),
    "ppi": ("protein interaction", "interacts"),
    "carrier": ("carrier", "carried"),
    "enzyme": ("enzyme", "metabolized"),
    "transporter": ("transporter", "transported"),
    "expression_present": ("expression", "expressed"),
    "expression_absent": ("absent expression", "not expressed"),
    "interacts_with": ("interaction", "interacts"),
    "parent_child": ("parent", "parent of"),
    "linked_to": ("link", "linked"),
    "linked_exposure": ("exposure", "exposed"),
    "synergistic_interaction": ("synergy", "synergistic"),
    "absent_gene": ("absent gene", "absent"),
    "expressed_gene": ("expressed gene", "expresses"),
    # PcQA - Pan-cancer Knowledge Graph
    "TREATMENT": ("treatment", "treats"),
    "SENSITIVITY_TO": ("sensitivity", "sensitive to"),
    "RESISTANCE_TO": ("resistance", "resistant to"),
    "INHIBITION_TO": ("inhibition", "inhibits"),
    "ACTIVATION_TO": ("activation", "activates"),
    "HAS_VAR": ("variant", "has variant"),
    "ORIGINATED_FROM": ("origin", "originates from"),
    "DRIVING_TO": ("driver", "associated with"),
    "HAS_GENE": ("gene", "has gene"),
    "DEVELOP_TO": ("development", "develops to"),
    "INCLUDE_A": ("includes", "includes"),
    "IS_A": ("alias", "is also known as"),
    "CAUSE_TO": ("cause", "causes"),
    "POSITIVE_REGULATED": ("positive regulation", "positively regulates"),
    "NEGATIVE_REGULATED": ("negative regulation", "negatively regulates"),
    "SYNTHETIC_LETHALITY": ("synthetic lethality", "synthetic lethal with"),
    "HAS_3GENE": ("three genes", "has three genes"),
    "INDUCE_TO": ("induction", "induces"),
}

# Type alias mapping for normalization (LLM output -> schema type)
TYPE_ALIAS_MAP = {
    # Common LLM outputs that need mapping
    "genetic mutations": "snvfull",
    "mutation": "snvfull",
    "mutations": "snvfull",
    "gene mutation": "snvfull",
    "variant": "snvfull",
    "variants": "snvfull",
    "snv": "snvfull",
    "gene": "genesymbol",
    "genes": "genesymbol",
    "gene symbol": "genesymbol",
    "cancer type": "cancer",
    "cancer types": "cancer",
    "cancers": "cancer",
    "drugs": "drug",
    "medication": "drug",
    "medications": "drug",
    "fusion gene": "fusion",
    "gene fusion": "fusion",
    "fusions": "fusion",
    "clinical trial": "clinicaltrial",
    "trial": "clinicaltrial",
    "trials": "clinicaltrial",
    "disease": "geneticdisease",
    "genetic disease": "geneticdisease",
    "cell line": "cancercell",
    "cell lines": "cancercell",
    "cells": "cancercell",
    # WebQSP (Freebase) type aliases (lowercased for TYPE_ALIAS_MAP lookup)
    "film_director": "person",
    "film_actor": "person",
    "deceased_person": "person",
    "politician": "person",
    "us_president": "person",
    "celebrity": "person",
    "athlete": "person",
    "musical_artist": "person",
    "american_football_player": "person",
    "author": "person",
    "city_town_village": "location",
    "college_university": "organization",
}


def get_nl_forms(relation: str) -> Tuple[str, str]:
    """リレーション名から(noun, verb)のタプルを取得

    Lookup order:
    1. Manually curated ``RELATION_TO_NATURAL_LANGUAGE`` dict
    2. Auto-generated from the relation name (handles Freebase dot-notation,
       underscore, CamelCase, etc.)
    """
    entry = RELATION_TO_NATURAL_LANGUAGE.get(relation)
    if entry:
        if isinstance(entry, tuple):
            return entry
        # 後方互換: 文字列の場合は同じ値をnounとverbに使用
        return (entry, entry)
    # Auto-generate for unknown / new KG relations
    return auto_generate_relation_nl(relation)


@dataclass
class SchemaPath:
    """スキーマパス"""

    types: List[str]  # [src_type, ..., tgt_type]
    relations: List[str]  # [rel1, rel2, ...]
    source: str  # "global" or "stepwise"
    score: float = 0.0
    directions: List[str] = field(
        default_factory=list
    )  # ["->", "<-", ...] for each relation

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
        """自然言語テキスト（ベクトル剪定用）

        Type情報を含めた構造的表現を使用:
        - "movie -[STARRED_ACTORS]-> person -[DIRECTED_BY]-> person"
        Uses RELATION_TO_NATURAL_LANGUAGE mapping for better semantic matching.
        """
        if not self.relations or not self.types:
            return ""

        # Type情報を含めたパス表現を構築
        parts = []
        for i, t in enumerate(self.types):
            parts.append(t)
            if i < len(self.relations):
                direction = self.directions[i] if i < len(self.directions) else "->"
                # Use natural language form if available
                noun, verb = get_nl_forms(self.relations[i])
                rel_name = verb if verb else self.relations[i].lower().replace("_", " ")
                parts.append(f"-[{rel_name}]{direction}")
        return " ".join(parts)


@dataclass
class EntitySet:
    """エンティティ集合（KoPL演算用）"""

    entities: Set[str]
    source_path: Optional[SchemaPath] = None
    anchor_name: Optional[str] = None  # どのアンカーから取得したか


@dataclass
class SubgraphEntity:
    """プロパティ付きエンティティ（PcQA用）"""

    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str, str]] = field(
        default_factory=list
    )  # (rel_type, direction, target_name)
    relation_properties: Dict[str, Any] = field(
        default_factory=dict
    )  # リレーションのプロパティ（fda_approved等）

    def to_kgt_format(self, anchor_name: str) -> str:
        """KGT形式のサブグラフ表現を生成

        Example: (cabozantinib {fda_approved: YES, nmpa_approved: NO})-[:treatment]->(cancer)
        Note: In this KG, approval properties are on nodes, not relations.
        """
        # ノードのプロパティを含める（fda_approved, nmpa_approved等）
        node_props_str = ""
        important_props = [
            "fda_approved",
            "nmpa_approved",
            "phase",
            "status",
            "evidence_level",
        ]
        props_to_show = {
            k: v for k, v in self.properties.items() if k in important_props
        }

        # リレーションプロパティも確認（一部のKGではリレーションに格納）
        if self.relation_properties:
            props_to_show.update(self.relation_properties)

        if props_to_show:
            props_list = [f"{k}: {v}" for k, v in props_to_show.items()]
            node_props_str = " {" + ", ".join(props_list) + "}"

        if self.relations:
            rel_type, direction, _ = self.relations[0]
            if direction == "<-":
                return f"({self.name}{node_props_str})-[:{rel_type}]->({anchor_name})"
            else:
                return f"({anchor_name})-[:{rel_type}]->({self.name}{node_props_str})"
        return f"({self.name}{node_props_str})"


@dataclass
class ExtendedTypeKoPLResult:
    """Extended Type-KoPL結果"""

    question: str
    kopl_program: Optional[KoPLOperation]
    candidate_paths: List[SchemaPath]
    selected_paths: List[SchemaPath]
    entity_sets: List[EntitySet]
    answer_entities: List[str]
    natural_answer: str = ""  # 自然言語回答（PcQA評価用）
    processing_log: List[str] = field(default_factory=list)


# =============================================================================
# Pydantic Models for LLM
# =============================================================================


class ExtractedEntitySchema(BaseModel):
    """KGT-style entity extraction result"""

    entity_name: str = Field(
        description="The main entity mentioned in the question (head entity)"
    )
    entity_type: str = Field(description="The type/label of the entity")
    target_type: str = Field(
        description="The type of entity being asked about (tail entity type)"
    )
    attribute: Optional[str] = Field(
        default=None, description="Specific attribute being queried, if any"
    )


class PropertyFilterSchema(BaseModel):
    """プロパティフィルタ（LLM出力用）"""

    node_type: str = Field(description="Node type to filter (e.g. Drug)")
    property_name: str = Field(description="Property name to filter on (e.g. fda_approved)")
    operator: str = Field(default="=", description="Comparison operator: =, !=, >, <, >=, <=, CONTAINS")
    value: str = Field(description="Value to compare against (e.g. YES)")


class AtomicOperationSchema(BaseModel):
    """Atomic操作（LLM出力用）- 1ホップ分の操作"""

    operation: str = Field(default="relate", description="Operation type: relate")
    src_type: str = Field(description="Source node type for this hop")
    tgt_type: str = Field(description="Target node type for this hop")
    relation: str = Field(description="Relation name for this hop")
    anchor_name: Optional[str] = Field(
        default=None, description="Anchor entity name (only for the first operation)"
    )


class AtomicKoPLProgramSchema(BaseModel):
    """Atomic Type-KoPLプログラム（LLM出力用）"""

    operations: List[AtomicOperationSchema] = Field(
        description="List of atomic operations, one per hop. For 3-hop queries, provide exactly 3 operations."
    )
    final_operation: str = Field(
        default="relate",
        description="Final operation to combine results: intersection, union, or relate",
    )
    filters: Optional[List[PropertyFilterSchema]] = Field(
        default=None,
        description="Property filters to apply on result nodes (e.g. fda_approved = YES)",
    )


# =============================================================================
# Schema Graph
# =============================================================================


class SchemaGraph:
    """スキーマグラフ（APSP探索用）"""

    def __init__(self):
        self.types: Set[str] = set()
        self.edges: List[Tuple[str, str, str, str]] = []  # (src, rel, tgt, direction)
        self.adjacency: Dict[str, List[Tuple[str, str, str]]] = defaultdict(
            list
        )  # type -> [(neighbor, rel, direction), ...]
        self.embeddings = None
        self.edge_embeddings: Dict[str, np.ndarray] = {}
        self.edge_freq: Dict[Tuple[str, str, str], int] = {}  # (src, rel, tgt) -> frequency
        # APSP用
        self.apsp_dist: Dict[str, Dict[str, int]] = {}
        self.type_to_idx: Dict[str, int] = {}
        self.idx_to_type: Dict[int, str] = {}

    def add_edge(
        self, src_type: str, relation: str, tgt_type: str, direction: str = "->"
    ):
        self.edges.append((src_type, relation, tgt_type, direction))
        self.types.add(src_type)
        self.types.add(tgt_type)
        # 順方向エッジを追加
        self.adjacency[src_type].append((tgt_type, relation, "->"))
        # 逆方向エッジも追加（逆引きトラバーサル用）
        # 例: drug -[target]-> gene を追加したら、gene <-[target]- drug も追加
        if src_type != tgt_type:  # 自己参照エッジ（ppi等）は逆方向追加不要
            self.adjacency[tgt_type].append((src_type, relation, "<-"))

    def compute_apsp(self):
        """Floyd-Warshallで全点対最短経路を計算"""
        type_list = sorted(self.types)
        n = len(type_list)

        # インデックスマッピング
        self.type_to_idx = {t: i for i, t in enumerate(type_list)}
        self.idx_to_type = {i: t for i, t in enumerate(type_list)}

        # 距離行列初期化
        INF = float("inf")
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
        if src_type not in self.apsp_dist or tgt_type not in self.apsp_dist.get(
            src_type, {}
        ):
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
        queue = [
            (start_type, [start_type], [], [])
        ]  # (current, type_path, rel_path, dir_path)
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
                shortest_paths.append(
                    SchemaPath(
                        types=type_path,
                        relations=rel_path,
                        source="apsp",
                        directions=dir_path,
                    )
                )
                continue

            # まだゴールに到達していない場合、探索継続
            if current_depth < shortest_dist:
                for next_type, relation, direction in self.adjacency[current]:
                    # サイクル防止（ただし自己参照は1回許可）
                    if next_type not in type_path:
                        state = (
                            next_type,
                            tuple(type_path + [next_type]),
                            tuple(rel_path + [relation]),
                        )
                        if state not in visited_at_depth[current_depth + 1]:
                            visited_at_depth[current_depth + 1].add(state)
                            queue.append(
                                (
                                    next_type,
                                    type_path + [next_type],
                                    rel_path + [relation],
                                    dir_path + [direction],
                                )
                            )
                    elif next_type == current and type_path.count(next_type) < 2:
                        # 自己参照（PPI等）
                        state = (
                            next_type,
                            tuple(type_path + [next_type]),
                            tuple(rel_path + [relation]),
                        )
                        if state not in visited_at_depth[current_depth + 1]:
                            visited_at_depth[current_depth + 1].add(state)
                            queue.append(
                                (
                                    next_type,
                                    type_path + [next_type],
                                    rel_path + [relation],
                                    dir_path + [direction],
                                )
                            )

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


def build_schema_graph(
    kg_type: str = "primekgqa", types_only: bool = False
) -> SchemaGraph:
    """スキーマグラフを構築

    Args:
        kg_type: "primekgqa", "metaqa", "pcqa", or other (e.g. "webqsp")
        types_only: Trueの場合、タイプ情報のみを含むスキーマを構築（リレーション情報なし）

    For KG types without a pre-defined schema module (e.g. "webqsp"), an empty
    SchemaGraph is returned.  The pipeline will build a question-specific local
    schema at runtime via ``_build_local_schema``.
    """
    SCHEMA_GRAPH = None
    entity_types: list = []

    if kg_type == "metaqa":
        from dataset_construction.schema_metaqa import SCHEMA_GRAPH as _SG, ENTITY_TYPES
        SCHEMA_GRAPH = _SG
        entity_types = ENTITY_TYPES
    elif kg_type == "pcqa":
        from dataset_construction.schema_pcqa import SCHEMA_GRAPH as _SG, ENTITY_TYPES
        SCHEMA_GRAPH = _SG
        entity_types = ENTITY_TYPES
    elif kg_type == "primekgqa":
        from dataset_construction.schema_v3 import SCHEMA_GRAPH as _SG, NODE_TYPES
        SCHEMA_GRAPH = _SG
        entity_types = NODE_TYPES
    # else: unknown KG (e.g. webqsp) – leave SCHEMA_GRAPH=None, empty schema

    schema = SchemaGraph()

    if SCHEMA_GRAPH is None:
        # No pre-defined schema available; fetch types dynamically from Neo4j.
        from database.search import GraphPathFinder
        finder = GraphPathFinder(kg_type=kg_type)
        labels = finder.graph.run(
            "CALL db.labels() YIELD label RETURN label"
        ).data()
        for row in labels:
            schema.types.add(row["label"])
    elif types_only:
        # タイプ情報のみを追加（エッジなし）
        for t in entity_types:
            schema.types.add(t)
    else:
        # 通常通りエッジを追加
        for item in SCHEMA_GRAPH:
            src, rel, tgt = item[0], item[1], item[2]
            # 4番目の要素が文字列なら方向、整数ならエッジ数（無視）
            if len(item) > 3 and isinstance(item[3], str):
                direction = item[3]
            else:
                direction = "->"
            schema.add_edge(src, rel, tgt, direction)

    # APSPを計算（types_onlyの場合は空のグラフだが、タイプ情報は保持）
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
            "drug",
            "disease",
            "gene/protein",
            "exposure",
            "biological_process",
            "molecular_function",
            "cellular_component",
            "pathway",
            "anatomy",
            "effect/phenotype",
        ],
        "metaqa": [
            "movie",
            "person",
            "organization",
            "text",
            "date",
            "language",
            "number",
        ],
        "pcqa": [
            "cancer",
            "cancercell",
            "canceralias",
            "drug",
            "drugalias",
            "genesymbol",
            "snvfull",
            "fusion",
            "geneticdisease",
            "clinicaltrial",
        ],
    }

    def __init__(
        self,
        model: str = BASEMODEL,
        embedding_model: str = "text-embedding-3-small",
        top_k_paths: int = 3,
        max_depth: int = 3,
        kg_type: str = "primekgqa",
        kgc: Optional[ETKKGConfig] = None,  # KG config (overrides kg_type)
        reranker_type: str = "none",  # "none", "llm", "hybrid"
        reranker_input_k: int = 10,  # Rerankerに渡す候補数
        use_schema_relations: bool = True,  # Falseの場合、KGから動的にリレーションを取得
        few_shot_k: int = 3,
        few_shot_pool_path: Optional[str] = None,
        n_kopl_candidates: int = 1,
        max_correction_rounds: int = 0,
        beam_width: int = 1,
        use_llm_cypher: bool = False,
        schema_distill: bool = False,
        cypher_informed_rerank: bool = False,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings

        # LLM: ローカルLLM (LiteLLM proxy) or OpenAI
        api_base = LLM_API_BASE or None
        llm_kwargs = {"model_provider": "openai", "temperature": 0, "max_tokens": 8192}
        if api_base:
            llm_kwargs["base_url"] = api_base
            llm_kwargs["api_key"] = "sk-local"  # LiteLLMはAPIキー不要
        self.llm = init_chat_model(model, **llm_kwargs)

        # Embeddings: 常にOpenAI (ローカルLLM使用時も)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Build KG config from kg_type if not provided directly
        self.kgc = kgc or build_etk_kg_config(kg_type)
        self.kg_type = self.kgc.kg_type  # backward compat
        self.use_schema_relations = use_schema_relations

        # スキーマ構築（use_schema_relations=Falseの場合、タイプ情報のみ）
        self.schema = build_schema_graph(kg_type, types_only=not use_schema_relations)
        if use_schema_relations:
            self.schema.compute_embeddings(self.embeddings)

        self.finder = GraphPathFinder(kg_type=kg_type)

        # KGからリレーション情報をキャッシュ（use_schema_relations=Falseの場合）
        self._relation_cache: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self._node_label_cache: Dict[str, Set[str]] = {}

        self.top_k_paths = top_k_paths
        self.max_depth = max_depth
        self.max_candidate_paths = 50  # 候補パスの最大数を制限

        # Reranker設定
        self.reranker_type = reranker_type
        self.reranker_input_k = reranker_input_k
        self.reranker = create_reranker(reranker_type, model=model, kg_type=kg_type)

        # PcQA: CancerCell複合名（run()で設定、per-sample）
        self._compound_names: List[str] = []

        # LLM Cypher generation flag
        self.use_llm_cypher = use_llm_cypher
        self.cypher_informed_rerank = cypher_informed_rerank

        # Multi-candidate KoPL generation
        self.n_kopl_candidates = n_kopl_candidates
        self.max_correction_rounds = max_correction_rounds
        self.schema_distill = schema_distill

        # Retrieval-based few-shot pool (MMR selection)
        self.few_shot_k = few_shot_k
        self.few_shot_pool: Optional[FewShotPool] = None
        if few_shot_pool_path:
            self.few_shot_pool = FewShotPool(kg_type=kg_type, embedding_model=embedding_model)
            self.few_shot_pool.load_pool(few_shot_pool_path)

    def _extract_relation_hints(
        self, kopl_program: KoPLOperation
    ) -> Optional[List[str]]:
        """KoPLプログラムからrelation_hintsを抽出（各atomic operationのrelation_hintを収集）"""
        operations = kopl_program.children if kopl_program.children else [kopl_program]
        hints = []
        for op in operations:
            for rel in op.relations:
                if rel.relation_hint:
                    hints.append(rel.relation_hint)
        return hints if hints else None

    # タイプの優先順位（複数ラベルがある場合に使用）
    TYPE_PRIORITY = {
        "metaqa": [
            "movie",
            "person",
            "language",
            "date",
            "text",
            "number",
            "organization",
        ],
        "primekgqa": ["drug", "disease", "gene/protein"],
        "pcqa": ["drug", "cancer", "genesymbol", "snvfull"],
    }

    def _get_entity_type(self, entity_name: str) -> Optional[str]:
        """KGからエンティティのタイプを取得

        Note: An entity name may appear as multiple nodes with different labels
        (e.g., Progesterone exists as both 'drug' and 'exposure').
        We collect ALL labels across all matching nodes and apply priority.
        """
        if not entity_name:
            return None
        try:
            cypher = """
            MATCH (n)
            WHERE n.name = $name
            RETURN labels(n) AS labels
            """
            records = self.finder.graph.run(cypher, name=entity_name).data()
            if not records:
                # Fallback: case-insensitive match
                cypher = """
                MATCH (n)
                WHERE toLower(n.name) = toLower($name)
                RETURN labels(n) AS labels
                """
                records = self.finder.graph.run(cypher, name=entity_name).data()
            if not records:
                return None

            # Collect all labels across all matching nodes
            all_labels_lower = set()
            for record in records:
                if record["labels"]:
                    for lbl in record["labels"]:
                        all_labels_lower.add(lbl.lower())

            # KGタイプに応じた優先順位でタイプを選択
            priority_list = self.TYPE_PRIORITY.get(self.kg_type, [])
            for ptype in priority_list:
                if ptype in all_labels_lower:
                    return ptype

            # 優先順位リストにない場合は有効タイプの最初のマッチを返す
            valid_types = self.ENTITY_TYPES.get(self.kg_type, [])
            for lbl in all_labels_lower:
                if lbl in valid_types:
                    return lbl
            return list(all_labels_lower)[0] if all_labels_lower else None
        except Exception as e:
            print(f"Entity type lookup error: {e}")
        return None

    def _extract_entity_kgt_style(
        self, question: str
    ) -> Optional[Tuple[str, str, str]]:
        """KGT-style entity extraction using LLM

        Official KGT approach:
        1. LLM extracts (entity_name, entity_type, target_type) from question
        2. Validate entity_name against database
        3. Return validated entity info

        Returns:
            Tuple of (entity_name, entity_type, target_type) or None
        """
        type_list = ", ".join(sorted(self.schema.types))

        # KGT-style prompt for entity extraction
        prompt = f"""Extract the main entity from this question.

Question: {question}

Available entity types: {type_list}

Instructions:
1. Identify the HEAD ENTITY - the specific entity mentioned in the question (e.g., drug name, gene name, cancer type)
2. Identify its TYPE from the available types
3. Identify the TARGET TYPE - what type of entity the question is asking about

{self.kgc.entity_extraction_examples}

Return the extracted entity information."""

        try:
            llm_with_output = self.llm.with_structured_output(ExtractedEntitySchema)
            result = llm_with_output.invoke(prompt)

            if not result or not result.entity_name:
                return None

            # Normalize types using alias map and schema types
            def normalize_schema_type(t: str) -> str:
                """Normalize type name to valid schema type"""
                t_lower = t.lower().strip()
                # Try alias map first
                if t_lower in TYPE_ALIAS_MAP:
                    t_lower = TYPE_ALIAS_MAP[t_lower]
                # Try case-insensitive match against schema types
                type_normalizer = {st.lower(): st for st in self.schema.types}
                return type_normalizer.get(t_lower, t_lower)

            # Validate entity against database
            validated_name = self._validate_entity_in_db(result.entity_name)
            normalized_target = normalize_schema_type(result.target_type)

            if validated_name:
                # Get actual type from DB
                actual_type = self._get_entity_type(validated_name)
                return (
                    validated_name,
                    actual_type or normalize_schema_type(result.entity_type),
                    normalized_target,
                )
            else:
                # Try case-insensitive search
                validated_name = self._fuzzy_entity_search(result.entity_name)
                if validated_name:
                    actual_type = self._get_entity_type(validated_name)
                    return (
                        validated_name,
                        actual_type or normalize_schema_type(result.entity_type),
                        normalized_target,
                    )

            # Return LLM result even if not validated (might still work)
            return (
                result.entity_name,
                normalize_schema_type(result.entity_type),
                normalized_target,
            )

        except Exception as e:
            print(f"KGT entity extraction error: {e}")
            return None

    def _extract_target_type(self, question: str) -> Optional[str]:
        """Extract the target entity type from a question (lightweight).

        Used when entity_name is already provided but target_type is unknown.
        Returns a normalized schema type or None.
        """
        type_list = ", ".join(sorted(self.schema.types))

        # PcQA-specific examples to handle tricky question patterns
        pcqa_examples = ""
        if self.kg_type == "pcqa":
            pcqa_examples = """
IMPORTANT: The target type is the type of ANSWER the question seeks, NOT the type mentioned in the question.
- "How does EGFR gene mutation affect the efficacy of X?" -> Drug (asking about drugs, not genes)
- "How to treat cancer carrying TP53?" -> Drug (asking about treatment drugs)
- "Which drugs are cancers with NTRK3-p.G623R resistant to?" -> Drug (asking about drugs)
- "What is the relationship between BRAF mutation and the therapeutic effect of X?" -> Drug (asking about drugs for a gene)
- "What drugs inhibit FGFR3?" -> Drug
- "What cancers can be treated by X?" -> Cancer
- "What genes does drug X inhibit?" -> Genesymbol
- "What genetic mutations are present in ovarian cancer?" -> SnvFull
- "What type of cancer can be driven by X?" -> Cancer
- "What fusion genes are in melanoma?" -> Fusion
- "What genetic diseases are caused by X?" -> GeneticDisease
Key patterns:
- "affect the efficacy of", "treat", "therapeutic effect", "resistant to", "sensitive to" -> Drug
- "what drugs", "targeted drugs", "targeted therapies" -> Drug
- "mutations need to be tested/detected for [drug]" -> Drug (NOT SnvFull! The answer is drugs that target the gene)
- "relationship between [gene] and [drug]" -> Drug
- "what type of cancer", "what cancers" -> Cancer
- "what genes", "which genes" -> Genesymbol
- "what mutations are present in [cancer]" -> SnvFull (only when asking what mutations exist IN a cancer)
"""

        prompt = f"""What type of entity is this question asking about?

Question: {question}

Available entity types: {type_list}
{pcqa_examples}
Return ONLY the target entity type name from the list above. For example:
- "What drugs treat X?" -> Drug
- "What cancers are related to X?" -> Cancer
- "What genes are associated with X?" -> Genesymbol

Target type:"""

        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip().split("\n")[0].strip()
            # Normalize
            type_normalizer = {t.lower(): t for t in self.schema.types}
            # Also handle common aliases
            raw_lower = raw.lower().strip()
            if raw_lower in TYPE_ALIAS_MAP:
                raw_lower = TYPE_ALIAS_MAP[raw_lower]
            return type_normalizer.get(raw_lower)
        except Exception as e:
            logger.warning(f"Target type extraction failed: {e}")
            return None

    def _validate_entity_in_db(self, entity_name: str) -> Optional[str]:
        """Validate entity name exists in database

        Checks both 'name' and 'name_en' fields (for PCQA)
        Also resolves aliases via IS_A relationship
        """
        try:
            # For PCQA, try alias resolution FIRST (DrugAlias->Drug, CancerAlias->Cancer)
            if self.kg_type == "pcqa":
                cypher = """
                MATCH (alias)-[:IS_A]->(main)
                WHERE alias.name = $name
                RETURN main.name AS name
                LIMIT 1
                """
                records = self.finder.graph.run(cypher, name=entity_name).data()
                if records:
                    return records[0]["name"]

            # Try exact match on name
            cypher = """
            MATCH (n)
            WHERE n.name = $name
            RETURN n.name AS name
            LIMIT 1
            """
            records = self.finder.graph.run(cypher, name=entity_name).data()
            if records:
                return records[0]["name"]

            # Try name_en field (for PCQA)
            cypher = """
            MATCH (n)
            WHERE toLower(n.name_en) = toLower($name)
            RETURN n.name AS name
            LIMIT 1
            """
            records = self.finder.graph.run(cypher, name=entity_name).data()
            if records:
                return records[0]["name"]

            return None
        except Exception as e:
            print(f"Entity validation error: {e}")
            return None

    def _fuzzy_entity_search(self, entity_name: str) -> Optional[str]:
        """Fuzzy search for entity name (case-insensitive, partial match)

        For PCQA, also tries to resolve aliases via IS_A relationship (prioritized)
        """
        try:
            # Case-insensitive alias resolution FIRST (for PCQA)
            if self.kg_type == "pcqa":
                cypher = """
                MATCH (alias)-[:IS_A]->(main)
                WHERE toLower(alias.name) = toLower($name)
                RETURN main.name AS name
                LIMIT 1
                """
                records = self.finder.graph.run(cypher, name=entity_name).data()
                if records:
                    return records[0]["name"]

            # Case-insensitive exact match
            cypher = """
            MATCH (n)
            WHERE toLower(n.name) = toLower($name)
            RETURN n.name AS name
            LIMIT 1
            """
            records = self.finder.graph.run(cypher, name=entity_name).data()
            if records:
                return records[0]["name"]

            # Contains match (for partial names, prefer shorter/more exact)
            cypher = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($name)
            RETURN n.name AS name
            ORDER BY length(n.name)
            LIMIT 1
            """
            records = self.finder.graph.run(cypher, name=entity_name).data()
            if records:
                return records[0]["name"]

            return None
        except Exception as e:
            print(f"Fuzzy search error: {e}")
            return None

    def _try_direct_paths(
        self, entity_name: str, entity_type: Optional[str], target_type: str
    ) -> Set[str]:
        """Fallback: Try all direct 1-hop paths from entity to target type

        This is useful when the multi-hop path (e.g., Gene->Cancer->Drug)
        doesn't exist but a direct path (e.g., Gene->Drug via INHIBITION_TO) does.
        """
        try:
            graph = self.finder.graph

            # Normalize types for Cypher
            src_type = entity_type.capitalize() if entity_type else None
            tgt_type = target_type.capitalize() if target_type else None

            if not src_type or not tgt_type:
                return set()

            def _esc(t: str) -> str:
                if "/" in t or "." in t or " " in t:
                    return f"`{t}`"
                return t

            # Try all direct relationships from entity to target type
            cypher = f"""
            MATCH (src:{_esc(src_type)})-[r]-(tgt:{_esc(tgt_type)})
            WHERE toLower(src.name) = toLower($name)
            RETURN DISTINCT tgt.name AS answer, type(r) AS rel
            LIMIT 20
            """
            records = graph.run(cypher, name=entity_name).data()

            if records:
                # Return all found entities
                return {r["answer"] for r in records if r["answer"]}

            # Also try with alias types for PCQA
            if self.kg_type == "pcqa":
                alias_types = {
                    "Cancer": "Cancer|CancerAlias",
                    "Drug": "Drug|DrugAlias",
                }
                tgt_label = alias_types.get(tgt_type, tgt_type)

                cypher = f"""
                MATCH (src:{_esc(src_type)})-[r]-(tgt:{tgt_label})
                WHERE toLower(src.name) = toLower($name)
                RETURN DISTINCT tgt.name AS answer, type(r) AS rel
                LIMIT 20
                """
                records = graph.run(cypher, name=entity_name).data()
                if records:
                    return {r["answer"] for r in records if r["answer"]}

            return set()
        except Exception as e:
            print(f"Direct path search error: {e}")
            return set()

    def _cancercell_2hop_search(
        self,
        entity_name: str,
        entity_type: str,
        selected_paths: List[SchemaPath],
    ) -> List[SubgraphEntity]:
        """PCQA: Search through CancerCell as intermediate hub node.

        For Genesymbol/Fusion: find CancerCells whose name CONTAINS the gene symbol,
        then traverse to Drug/Cancer targets.
        For SnvFull: find CancerCells connected via HAS_VAR, then traverse to targets.
        """
        try:
            graph = self.finder.graph
            etype_norm = entity_type.lower()

            # Determine search name and CancerCell match clause
            search_name = self._compound_search_term or entity_name
            if etype_norm == "snvfull":
                cc_match = "MATCH (cc:CancerCell)-[:HAS_VAR]->(snv:SnvFull) WHERE toLower(snv.name) = toLower($name)"
            else:
                cc_match = "MATCH (cc:CancerCell) WHERE toLower(cc.name) CONTAINS toLower($name)"

            # Determine target type from selected paths or use Drug as default
            target_labels = set()
            for p in selected_paths:
                if p.types:
                    # The last type in the path is usually the target
                    last_type = p.types[-1]
                    if last_type.lower() not in ("cancercell", etype_norm):
                        target_labels.add(last_type)
                    # Also check first type (path may be reversed)
                    first_type = p.types[0]
                    if first_type.lower() not in ("cancercell", etype_norm):
                        target_labels.add(first_type)
            if not target_labels:
                target_labels = {"Drug"}

            entities = []
            seen_names = set()

            for tgt_label in target_labels:
                # Expand alias types
                if tgt_label == "Cancer":
                    tgt_cypher = "Cancer|CancerAlias"
                elif tgt_label == "Drug":
                    tgt_cypher = "Drug|DrugAlias"
                else:
                    tgt_cypher = tgt_label

                cypher = f"""
                {cc_match}
                MATCH (cc)-[r]-(tgt:{tgt_cypher})
                WHERE cc <> tgt
                RETURN DISTINCT tgt AS entity, labels(tgt) AS labels, type(r) AS rel_type
                LIMIT 30
                """
                records = graph.run(cypher, name=search_name).data()

                for record in records:
                    entity_node = record.get("entity")
                    if not entity_node:
                        continue
                    props = dict(entity_node) if hasattr(entity_node, "__iter__") else {}
                    name = props.get("name", "")
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)
                    labels = record.get("labels", [])
                    entity_type_label = labels[0] if labels else "Unknown"
                    filtered_props = {}
                    for key in ["name", "fda_approved", "nmpa_approved"]:
                        if key in props and props[key] is not None:
                            filtered_props[key] = props[key]
                    entities.append(
                        SubgraphEntity(
                            name=name,
                            entity_type=entity_type_label,
                            properties=filtered_props,
                            relations=[(record.get("rel_type", ""), "via CancerCell ->", entity_name)],
                        )
                    )

            return entities

        except Exception as e:
            print(f"CancerCell 2-hop search error: {e}")
            return []

    def _try_direct_paths_with_properties(
        self, entity_name: str, entity_type: Optional[str], target_type: str
    ) -> List[SubgraphEntity]:
        """PcQA用: 直接パスでプロパティ付きエンティティを取得

        Fallback for when multi-hop path retrieval fails.
        Returns SubgraphEntity with properties (fda_approved, nmpa_approved, etc.)
        """
        try:
            graph = self.finder.graph

            # Normalize types to match schema (case-insensitive lookup)
            type_normalizer = {t.lower(): t for t in self.schema.types}
            src_type = (
                type_normalizer.get(entity_type.lower(), entity_type.capitalize())
                if entity_type
                else None
            )
            tgt_type = (
                type_normalizer.get(target_type.lower(), target_type.capitalize())
                if target_type
                else None
            )

            if not src_type or not tgt_type:
                return []

            # PcQA: Expand to include alias types
            ALIAS_TYPES = {
                "Cancer": "Cancer|CancerAlias",
                "Drug": "Drug|DrugAlias",
            }
            tgt_label = (
                ALIAS_TYPES.get(tgt_type, tgt_type)
                if self.kg_type == "pcqa"
                else tgt_type
            )
            src_label = (
                ALIAS_TYPES.get(src_type, src_type)
                if self.kg_type == "pcqa"
                else src_type
            )

            def _esc(t: str) -> str:
                if "/" in t or "." in t or " " in t:
                    return f"`{t}`"
                return t

            # Try all direct relationships from entity to target type
            # Return full node properties
            cypher = f"""
            MATCH (src:{_esc(src_label)})-[r]-(tgt:{_esc(tgt_label)})
            WHERE toLower(src.name) = toLower($name)
            RETURN DISTINCT tgt AS entity, labels(tgt) AS labels, type(r) AS rel_type
            LIMIT 30
            """
            records = graph.run(cypher, name=entity_name).data()

            entities = []
            seen_names = set()

            for record in records:
                entity_node = record.get("entity")
                if not entity_node:
                    continue

                # Extract properties from node
                props = dict(entity_node) if hasattr(entity_node, "__iter__") else {}
                name = props.get("name", "")

                if not name or name in seen_names:
                    continue
                seen_names.add(name)

                labels = record.get("labels", [])
                entity_type_label = labels[0] if labels else "Unknown"

                # Filter to important properties
                filtered_props = {}
                important_keys = ["name", "fda_approved", "nmpa_approved"]
                for key in important_keys:
                    if key in props and props[key] is not None:
                        filtered_props[key] = props[key]

                entities.append(
                    SubgraphEntity(
                        name=name,
                        entity_type=entity_type_label,
                        properties=filtered_props,
                        relations=[(record.get("rel_type", ""), "->", entity_name)],
                    )
                )

            return entities

        except Exception as e:
            print(f"Direct path with properties error: {e}")
            return []

    def _try_any_neighbor_with_properties(
        self, entity_name: str, entity_type: str
    ) -> List[SubgraphEntity]:
        """PcQA fallback: search ALL 1-hop neighbors regardless of target type.

        Used when target_type extraction was wrong and no results were found.
        """
        try:
            graph = self.finder.graph
            cypher = """
            MATCH (src)-[r]-(tgt)
            WHERE toLower(src.name) = toLower($name) AND src <> tgt
            RETURN DISTINCT tgt AS entity, labels(tgt) AS labels, type(r) AS rel_type
            LIMIT 30
            """
            records = graph.run(cypher, name=entity_name).data()

            entities = []
            seen_names = set()
            for record in records:
                entity_node = record.get("entity")
                if not entity_node:
                    continue
                props = dict(entity_node) if hasattr(entity_node, "__iter__") else {}
                name = props.get("name", "")
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                labels = record.get("labels", [])
                entity_type_label = labels[0] if labels else "Unknown"
                filtered_props = {
                    k: v
                    for k, v in props.items()
                    if k in ("name", "name_en", "fda_approved", "nmpa_approved")
                    and v is not None
                }
                entities.append(
                    SubgraphEntity(
                        name=name,
                        entity_type=entity_type_label,
                        properties=filtered_props,
                        relations=[
                            (record.get("rel_type", ""), "->", entity_name)
                        ],
                    )
                )
            return entities
        except Exception as e:
            print(f"Any neighbor search error: {e}")
            return []

    def _retrieve_subgraph_with_properties(
        self,
        anchor_name: str,
        anchor_type: str,
        target_type: str,
        path: SchemaPath,
    ) -> List[SubgraphEntity]:
        """PcQA用: プロパティ付きサブグラフを取得（KGT方式）

        Returns entities with their properties AND relation properties (fda_approved, nmpa_approved, etc.)
        This follows the official KGT approach where relation properties are used for filtering.
        """
        graph = self.finder.graph
        entities = []

        # PCQA重要プロパティ（エンティティ用）- fda_approved, nmpa_approved含む
        IMPORTANT_ENTITY_PROPS = [
            "name",
            "name_en",
            "fda_approved",
            "nmpa_approved",
            "cancer_type",
            "drug_class",
            "target_gene",
            "mutation_type",
            "phase",
            "status",
            "gender",
            "location",
            "evidence_level",
            "class_type",
        ]

        # リレーションの重要プロパティ（フィルタリング用）
        IMPORTANT_REL_PROPS = [
            "fda_approved",
            "nmpa_approved",
            "score",
            "evidence_level",
            "phase",
            "status",
            "class_type",
        ]

        # PCQA alias mapping: also search for alias types
        ALIAS_TYPES = {
            "Cancer": ["Cancer", "CancerAlias"],
            "Drug": ["Drug", "DrugAlias"],
        }

        def get_label(t: str) -> str:
            if "/" in t or "." in t or " " in t:
                return f"`{t}`"
            # For PCQA, expand Cancer/Drug to also include aliases
            if t in ALIAS_TYPES:
                labels = ALIAS_TYPES[t]
                return "|".join(labels)  # Returns "Cancer|CancerAlias"
            return t

        def get_rel(r: str) -> str:
            """リレーション名をCypher用にエスケープ（スペース、ハイフン、ドット等）"""
            if " " in r or "-" in r or "/" in r or "." in r:
                return f"`{r}`"
            return r

        try:
            # PcQA compound: use CONTAINS for CancerCell gene matching
            if self._compound_search_term:
                where_anchor = "toLower(a.name) CONTAINS toLower($search_term)"
                params = {"search_term": self._compound_search_term}
            else:
                where_anchor = "toLower(a.name) = toLower($anchor_name)"
                params = {"anchor_name": anchor_name}

            # Auto-reverse path if anchor_type doesn't match path.types[0]
            # e.g., entity is Drug but path starts with Cancer -> reverse so Drug is anchor
            path_types = list(path.types)
            path_rels = list(path.relations)
            if anchor_type and path_types:
                anchor_norm = anchor_type.lower()
                first_norm = path_types[0].lower()
                last_norm = path_types[-1].lower()
                # Also check alias types (Cancer/CancerAlias, Drug/DrugAlias)
                alias_map = {"canceralias": "cancer", "drugalias": "drug"}
                anchor_check = alias_map.get(anchor_norm, anchor_norm)
                first_check = alias_map.get(first_norm, first_norm)
                last_check = alias_map.get(last_norm, last_norm)
                if anchor_check != first_check and anchor_check == last_check:
                    path_types = list(reversed(path_types))
                    path_rels = list(reversed(path_rels))

            # パスに沿ってサブグラフを取得（リレーションプロパティ含む、方向を無視）
            if len(path_types) == 2:
                # 1-hop: 方向を無視
                cypher = f"""
                MATCH (a:{get_label(path_types[0])})-[r:{get_rel(path_rels[0])}]-(b:{get_label(path_types[1])})
                WHERE {where_anchor} AND a <> b
                RETURN b AS entity, labels(b) AS labels, type(r) AS rel_type, properties(r) AS rel_props
                """
            elif len(path_types) == 3:
                # 2-hop: 方向を無視
                cypher = f"""
                MATCH (a:{get_label(path_types[0])})-[r1:{get_rel(path_rels[0])}]-(mid:{get_label(path_types[1])})-[r2:{get_rel(path_rels[1])}]-(b:{get_label(path_types[2])})
                WHERE {where_anchor} AND a <> mid AND mid <> b AND a <> b
                RETURN b AS entity, labels(b) AS labels, type(r2) AS rel_type, properties(r2) AS rel_props, mid AS intermediate
                """
            else:
                return []

            # Note: Property filters are applied post-retrieval in Phase 5.5
            # to ensure fallback paths (4.5a/b/c) are also filtered.

            records = graph.run(cypher, **params).data()

            seen_names = set()
            for record in records:
                entity_node = record.get("entity")
                if not entity_node:
                    continue

                # Node から properties を取得
                props = dict(entity_node) if hasattr(entity_node, "__iter__") else {}
                name = props.get("name", "")

                if not name or name in seen_names:
                    continue
                seen_names.add(name)

                # エンティティの重要プロパティのみ抽出
                filtered_props = {}
                for key in IMPORTANT_ENTITY_PROPS:
                    if key in props and props[key] is not None:
                        filtered_props[key] = props[key]

                # リレーションのプロパティを抽出（KGT方式の肝）
                rel_props = record.get("rel_props", {}) or {}
                filtered_rel_props = {}
                for key in IMPORTANT_REL_PROPS:
                    if key in rel_props and rel_props[key] is not None:
                        filtered_rel_props[key] = rel_props[key]

                labels = record.get("labels", [])
                entity_type = labels[0] if labels else "Unknown"
                direction = path.directions[0] if path.directions else "->"

                entities.append(
                    SubgraphEntity(
                        name=name,
                        entity_type=entity_type,
                        properties=filtered_props,
                        relations=[
                            (record.get("rel_type", ""), direction, anchor_name)
                        ],
                        relation_properties=filtered_rel_props,
                    )
                )

            return entities

        except Exception as e:
            print(f"Subgraph retrieval error: {e}")
            return []

    def _generate_answer_from_subgraph_llm(
        self,
        question: str,
        anchor_name: str,
        subgraph_entities: List[SubgraphEntity],
    ) -> str:
        """PcQA用: LLMでサブグラフから自然言語回答を生成（KGT方式）

        Official KGT approach: use LLM to generate answer from subgraph WITH properties.
        The LLM uses relation properties (fda_approved, nmpa_approved, etc.) to filter results.
        """
        if not subgraph_entities:
            return f"Output: There are no results found for this query."

        # KGT形式のサブグラフ表現を生成
        subgraph_lines = []
        for entity in subgraph_entities[:30]:  # 最大30件
            kgt_format = entity.to_kgt_format(anchor_name)
            subgraph_lines.append(f"{kgt_format} {entity.name}")

        subgraph_text = "\n".join(subgraph_lines)

        prompt = f"""You are a reasoning robot, and you need to perform the following two steps step by step:
1. Output a corresponding natural language sentence for each relationship chain.
2. Answer my question using natural language from step 1.
Note: The output format is: Output: One sentence in natural language.

IMPORTANT RULES:
- Property values YES/true/True = approved, NO/false/False = not approved
- Filter by fda_approved/nmpa_approved when the question asks about approval status
- If no entities match the criteria, state that none exist

=== EXAMPLES ===

Example 1 - Cancer association:
Subgraph:
(MET)-[:DRIVING_TO]->(low-grade glioma)
(MET)-[:DRIVING_TO]->(renal clear cell carcinoma)
Question: Which types of cancer are associated with MET?
Step 1: MET drives low-grade glioma. MET drives renal clear cell carcinoma.
Output: MET is associated with low-grade glioma and renal clear cell carcinoma.

Example 2 - Drug treatment:
Subgraph:
(diethylstilbestrol)-[:TREATMENT]->(breast cancer)
(diethylstilbestrol)-[:TREATMENT]->(prostate cancer)
Question: What types of cancer can be treated with diethylstilbestrol?
Step 1: Diethylstilbestrol is a treatment for breast cancer. Diethylstilbestrol is a treatment for prostate cancer.
Output: Diethylstilbestrol can treat breast cancer and prostate cancer.

Example 3 - Drug inhibition:
Subgraph:
(TERT)-[:INHIBITION_TO]->(doxorubicin {{fda_approved: YES}})
Question: What drugs can treat cancers with TERT mutations?
Step 1: TERT is inhibited by doxorubicin (FDA approved).
Output: Cancers with TERT mutations can be inhibited by doxorubicin.

Example 4 - Gene activation:
Subgraph:
(codeine)-[:ACTIVATION_TO]->(OPRD1)
(codeine)-[:ACTIVATION_TO]->(OPRK1)
(codeine)-[:ACTIVATION_TO]->(OPRM1)
Question: Which genes can be activated by codeine?
Step 1: Codeine activates OPRD1. Codeine activates OPRK1. Codeine activates OPRM1.
Output: The following genes can be activated by codeine: OPRD1, OPRK1, and OPRM1.

Example 5 - NMPA-approved drugs (with filtering):
Subgraph:
(DDR2)-[:INHIBITION_TO]->(nilotinib {{fda_approved: YES, nmpa_approved: YES}})
(DDR2)-[:INHIBITION_TO]->(dasatinib {{fda_approved: YES, nmpa_approved: YES}})
(DDR2)-[:INHIBITION_TO]->(sitravatinib {{fda_approved: NO, nmpa_approved: NO}})
Question: What are the NMPA-approved drugs for cancers with DDR2 mutations?
Step 1: DDR2 is inhibited by nilotinib (NMPA approved). DDR2 is inhibited by dasatinib (NMPA approved). DDR2 is inhibited by sitravatinib (not NMPA approved).
Output: The NMPA-approved drugs for DDR2 are nilotinib and dasatinib.

Example 6 - No matching results:
Subgraph:
(TNKS)-[:INHIBITION_TO]->(drug1 {{fda_approved: NO, nmpa_approved: NO}})
Question: What are the NMPA-approved drugs for cancers with TNKS mutations?
Step 1: TNKS is inhibited by drug1 (not NMPA approved).
Output: There are no NMPA-approved drugs that can treat cancers with TNKS mutations.

Example 7 - Genetic mutations:
Subgraph:
(astrocytoma)-[:HAS_VAR]->(EGFR-p.L861R)
(astrocytoma)-[:HAS_VAR]->(EGFR-p.G719A)
Question: What genetic mutations are present in astrocytoma?
Step 1: Astrocytoma has the variant EGFR-p.L861R. Astrocytoma has the variant EGFR-p.G719A.
Output: Astrocytoma can be caused by the following genetic mutations: EGFR-p.L861R and EGFR-p.G719A.

=== YOUR TASK ===

Subgraph:
{subgraph_text}

Question: {question}

Step 1:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            # Extract the "Output: ..." line from the CoT response
            # The LLM generates "Step 1: ... \nOutput: ..."
            if "Output:" in answer:
                # Take everything after the last "Output:"
                output_idx = answer.rfind("Output:")
                answer = answer[output_idx:]
            else:
                answer = f"Output: {answer}"

            return answer

        except Exception as e:
            print(f"LLM answer generation error: {e}")
            # フォールバック: エンティティ名のリストを返す
            entity_names = [e.name for e in subgraph_entities]
            return f"Output: {', '.join(entity_names[:10])}."

    def run(
        self, question: str, entity_name: Optional[str] = None
    ) -> ExtendedTypeKoPLResult:
        """パイプライン実行"""
        self._current_question = question
        log = []
        log.append(f"Question: {question}")

        # Phase 0: KGT-style Entity Extraction (if entity_name not provided)
        target_type = None
        if not entity_name:
            log.append("Phase 0: KGT-style Entity Extraction")
            extracted = self._extract_entity_kgt_style(question)
            if extracted:
                entity_name, entity_type_extracted, target_type = extracted
                log.append(
                    f"  Extracted: entity='{entity_name}', type='{entity_type_extracted}', target='{target_type}'"
                )
            else:
                log.append("  Failed to extract entity from question")
        else:
            # entity_nameが渡された場合もDB上の正式名に解決する（大文字小文字の不一致対応）
            validated = self._validate_entity_in_db(entity_name)
            if not validated:
                validated = self._fuzzy_entity_search(entity_name)
            if validated and validated != entity_name:
                log.append(f"  Entity resolved: '{entity_name}' -> '{validated}'")
                entity_name = validated

            # PcQA: Also extract target_type even when entity_name is provided
            if self.kg_type == "pcqa":
                target_type = self._extract_target_type(question)
                if target_type:
                    log.append(f"  Target type extracted: '{target_type}'")

        # エンティティタイプを取得
        entity_type = self._get_entity_type(entity_name) if entity_name else None

        # Property filters from KoPL (set during Phase 1, applied in Phase 4)
        self._active_filters: Optional[List[PropertyFilterSchema]] = None

        # PcQA: CancerCell複合名マッチング
        self._compound_names: List[str] = []
        self._compound_search_term: Optional[str] = None
        if (
            self.kg_type == "pcqa"
            and entity_type
            and entity_type.lower() in ("genesymbol", "fusion")
            and entity_name
        ):
            gene_name = entity_name  # 元の遺伝子名を保持
            from pipeline.common.pcqa import (
                resolve_pcqa_compound_entity,
                CANCERCELL_KEYWORDS,
            )

            compound = resolve_pcqa_compound_entity(
                self.llm, self.finder, question, entity_name, entity_type
            )
            if compound:
                compound_name, compound_type, compound_names = compound
                log.append(
                    f"  PcQA CancerCell resolved: {entity_name} -> {compound_name} ({compound_type})"
                )
                log.append(f"  Compound names: {len(compound_names)} variants")
                entity_name = compound_name
                entity_type = compound_type
                self._compound_names = compound_names
                self._compound_search_term = gene_name
            elif any(kw in question.lower() for kw in CANCERCELL_KEYWORDS):
                # Compound resolution失敗でもキーワードからCancerCellと推定
                log.append(
                    f"  PcQA CancerCell fallback: using CONTAINS for {gene_name}"
                )
                entity_type = "CancerCell"
                self._compound_search_term = gene_name

        # Dynamic local schema for large / unknown KGs
        original_schema = None
        self._using_local_schema = False
        if entity_name and self._needs_local_schema():
            local_schema = self._build_local_schema(entity_name, max_hops=2, question=question)
            if local_schema:
                original_schema = self.schema
                self.schema = local_schema
                self._using_local_schema = True
                log.append(
                    f"  Using local schema: {len(local_schema.edges)} edges "
                    f"around '{entity_name}'"
                )
            else:
                log.append(
                    f"  Warning: could not build local schema for '{entity_name}'"
                )

        try:
            # Phase 1: 擬似クエリ生成
            log.append("Phase 1: Pseudo Query Generation")
            kopl_program = self._generate_type_kopl(
                question, entity_name, entity_type, target_type
            )
            relation_hints = None
            if kopl_program:
                log.append(
                    f"  Generated KoPL program with {len(kopl_program.relations)} relations"
                )
                if kopl_program.children:
                    log.append(f"  Final operation: {kopl_program.op_type}")
                # Store filters for Phase 4 Cypher execution
                self._active_filters = kopl_program.filters
                if self._active_filters:
                    for f in self._active_filters:
                        log.append(f"  Filter: {f.node_type}.{f.property_name} {f.operator} {f.value}")
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

            # Phase 1.5: Disabled (bak版にはなし)
            # kopl_program = self._verify_and_correct_type_path(kopl_program)
            log.append("Phase 1.5: Skipped (bak parity)")

            # Phase 2: ハイブリッド探索
            log.append("Phase 2: Hybrid Schema Search")
            candidate_paths = self._hybrid_schema_search(kopl_program)
            log.append(f"  Found {len(candidate_paths)} candidate paths")
            for p in candidate_paths[:5]:
                log.append(f"    [{p.source}] {p.to_text()}")

            # Phase 1→2 Correction Loop: retry with diagnostic feedback
            correction_round = 0
            while not candidate_paths and correction_round < self.max_correction_rounds:
                correction_round += 1
                log.append(f"Phase 1→2 Correction Round {correction_round}/{self.max_correction_rounds}")
                diagnosis = self._diagnose_phase2_failure(kopl_program)
                log.append(f"  Diagnosis: {diagnosis[:300]}")
                corrected = self._generate_corrected_type_kopl(
                    question, entity_name, entity_type, target_type,
                    kopl_program, diagnosis,
                )
                if not corrected:
                    log.append("  Correction failed (LLM returned None)")
                    break
                log.append(f"  Corrected KoPL: {len(corrected.relations)} relations")
                corrected = self._verify_and_correct_type_path(corrected)
                candidate_paths = self._hybrid_schema_search(corrected)
                log.append(f"  Re-search found {len(candidate_paths)} candidate paths")
                if candidate_paths:
                    kopl_program = corrected
                    relation_hints = self._extract_relation_hints(kopl_program)
                    self._active_filters = kopl_program.filters

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

            # Phase 3: 質問文でベクトル剪定
            # Rerankerあり: Top-10 → LLM RerankerでTop-1
            # Rerankerなし: Top-1を使用
            use_reranker = self.reranker_type != "none"
            prune_k = self.reranker_input_k if use_reranker else 1

            log.append(f"Phase 3: Vector-based Pruning (top_k={prune_k})")
            pruned_paths = self._vector_pruning(
                question, candidate_paths, top_k=prune_k, relation_hints=relation_hints
            )
            log.append(f"  Pruned to {len(pruned_paths)} paths")
            for p in pruned_paths[:5]:  # ログは最大5つまで
                log.append(f"    {p.to_text()} (score: {p.score:.4f})")
            if len(pruned_paths) > 5:
                log.append(f"    ... and {len(pruned_paths) - 5} more")

            # Phase 3.5: パス選択
            trial_samples = None  # Cypher trial結果（multi_path_k > 1 時のみ）

            if self.cypher_informed_rerank and entity_name and use_reranker:
                # Cypher-Informed Reranking: trial実行 → 結果付きでReranker
                log.append("Phase 3.5a: Cypher Trial (cypher_informed_rerank)")
                trial_results = self._run_cypher_trials(pruned_paths, entity_name, limit=5)
                log.append(f"  Cypher trial: {len(pruned_paths)} -> {len(trial_results)} with results")
                for path, samples in trial_results[:5]:
                    log.append(f"    {path.to_text()} -> {samples[:3]}")

                if trial_results:
                    verified_paths = [p for p, _ in trial_results]
                    trial_samples = {id(p): s for p, s in trial_results}
                    log.append(f"Phase 3.5b: Cypher-Informed Reranker (input={len(verified_paths)})")
                    selected_paths = self.reranker.rerank(
                        question,
                        verified_paths,
                        top_k=1,
                        relation_hints=relation_hints,
                        kopl_program=kopl_program,
                        trial_samples=trial_samples,
                    )
                else:
                    log.append("  Cypher trial: all empty, fallback to vector top-1")
                    selected_paths = pruned_paths[:1]
                log.append(f"  Reranked to {len(selected_paths)} path(s)")
                for p in selected_paths:
                    log.append(f"    {p.to_text()}")

            elif use_reranker:
                # 従来のLLM Reranker (multi_path_k=1)
                log.append(
                    f"Phase 3.5: LLM Reranker ({self.reranker_type}, input={len(pruned_paths)})"
                )
                selected_paths = self.reranker.rerank(
                    question,
                    pruned_paths,
                    top_k=1,
                    relation_hints=relation_hints,
                    kopl_program=kopl_program,
                )
                log.append(f"  Reranked to {len(selected_paths)} path(s)")
                for p in selected_paths:
                    log.append(f"    {p.to_text()}")
            elif (
                kopl_program.op_type == OperationType.INTERSECTION and kopl_program.children
            ):
                # Intersectionクエリ: 各子操作のsrc_typeに対応する最良パスを選択
                selected_paths = []
                needed_src_types = set()
                for child in kopl_program.children:
                    for rel in child.relations:
                        if rel.src_type:
                            needed_src_types.add(rel.src_type)

                for src_type in needed_src_types:
                    # このsrc_typeに対応する最高スコアのパスを選択
                    for p in pruned_paths:
                        if p.types and p.types[0] == src_type:
                            selected_paths.append(p)
                            break

                log.append(
                    f"  Selected {len(selected_paths)} paths for intersection (src_types: {needed_src_types})"
                )
                for p in selected_paths:
                    log.append(f"    {p.to_text()}")
            else:
                # Rerankerなし: Top-3をそのまま使用
                selected_paths = pruned_paths[:3]

            # Phase 4: データ取得
            # PcQA: プロパティ付きサブグラフを取得してLLMで回答生成
            # その他: エンティティ名のみ取得
            if self.kg_type == "pcqa":
                log.append("Phase 4: Subgraph Retrieval with Properties (PcQA)")
                subgraph_entities: List[SubgraphEntity] = []

                # 選択されたパスでサブグラフを取得
                if selected_paths and entity_name and entity_type:
                    for path in selected_paths:
                        entities = self._retrieve_subgraph_with_properties(
                            anchor_name=entity_name,
                            anchor_type=entity_type,
                            target_type=target_type or path.types[-1] if path.types else "",
                            path=path,
                        )
                        subgraph_entities.extend(entities)

                log.append(f"  Retrieved {len(subgraph_entities)} entities with properties")
                for e in subgraph_entities[:5]:
                    props_str = ", ".join(
                        f"{k}={v}" for k, v in list(e.properties.items())[:3]
                    )
                    log.append(f"    {e.name} ({e.entity_type}): {props_str}")

                # Fallback 1: Try other candidate paths if selected path returned nothing
                if (
                    not subgraph_entities
                    and candidate_paths
                    and entity_name
                    and entity_type
                ):
                    log.append("Phase 4.5a: Fallback - Try Other Candidate Paths")
                    # Try all candidate paths that weren't selected
                    other_paths = [p for p in candidate_paths if p not in selected_paths]
                    for path in other_paths[:3]:  # Try up to 3 more paths
                        entities = self._retrieve_subgraph_with_properties(
                            anchor_name=entity_name,
                            anchor_type=entity_type,
                            target_type=target_type or path.types[-1] if path.types else "",
                            path=path,
                        )
                        if entities:
                            log.append(
                                f"  Found {len(entities)} entities via path: {path.to_text()}"
                            )
                            subgraph_entities.extend(entities)
                            break  # Stop once we find results

                # Fallback 2: Direct path with properties (any relationship)
                if not subgraph_entities and entity_name and target_type:
                    log.append("Phase 4.5b: Fallback - Direct Path Search with Properties")
                    subgraph_entities = self._try_direct_paths_with_properties(
                        entity_name, entity_type, target_type
                    )
                    if subgraph_entities:
                        log.append(
                            f"  Found {len(subgraph_entities)} entities with properties"
                        )
                        for e in subgraph_entities[:3]:
                            props_str = ", ".join(
                                f"{k}={v}" for k, v in list(e.properties.items())[:3]
                            )
                            log.append(f"    {e.name}: {props_str}")

                # Fallback 2b: Try ALL neighbor types when target_type was wrong
                if not subgraph_entities and entity_name and entity_type and self.kg_type == "pcqa":
                    log.append("Phase 4.5b2: Fallback - Any Neighbor Type Search")
                    subgraph_entities = self._try_any_neighbor_with_properties(
                        entity_name, entity_type
                    )
                    if subgraph_entities:
                        log.append(
                            f"  Found {len(subgraph_entities)} entities (any neighbor)"
                        )
                        for e in subgraph_entities[:3]:
                            log.append(f"    {e.name} ({e.entity_type})")

                # Fallback 3: PCQA CancerCell-mediated 2-hop search
                # When entity is Genesymbol/SnvFull/Fusion, go through CancerCell hub
                if not subgraph_entities and self.kg_type == "pcqa" and entity_name:
                    etype_norm = (entity_type or "").lower()
                    if etype_norm in ("genesymbol", "snvfull", "fusion", "cancercell"):
                        log.append("Phase 4.5c: Fallback - CancerCell 2-hop Search")
                        subgraph_entities = self._cancercell_2hop_search(
                            entity_name, entity_type or "", selected_paths
                        )
                        if subgraph_entities:
                            log.append(
                                f"  Found {len(subgraph_entities)} entities via CancerCell hub"
                            )
                            for e in subgraph_entities[:3]:
                                log.append(f"    {e.name} ({e.entity_type})")

                # Phase 5: KoPL論理演算（PcQAでは単純にエンティティ名を抽出）
                log.append("Phase 5: Entity Extraction")
                answer_entities = [e.name for e in subgraph_entities]
                entity_sets = (
                    [
                        EntitySet(
                            entities=set(answer_entities),
                            source_path=selected_paths[0] if selected_paths else None,
                            anchor_name=entity_name,
                        )
                    ]
                    if answer_entities
                    else []
                )
                log.append(f"  Extracted {len(answer_entities)} entity names")

                # Phase 6: LLMで自然言語回答生成（PcQA専用）
                log.append("Phase 6: LLM-based Answer Generation (PcQA)")
                natural_answer = self._generate_answer_from_subgraph_llm(
                    question=question,
                    anchor_name=entity_name or "",
                    subgraph_entities=subgraph_entities,
                )
                log.append(f"  Generated: {natural_answer[:100]}...")

            else:
                # PrimeKGQA / MetaQA / WebQSP: 従来のロジック
                log.append("Phase 4: Data Retrieval")
                entity_sets = self._retrieve_entities(kopl_program, selected_paths)
                log.append(f"  Retrieved {len(entity_sets)} entity sets")
                for es in entity_sets:
                    log.append(f"    Set with {len(es.entities)} entities")

                # Phase 4→1 Correction: if Cypher returned 0 entities, retry with feedback
                p4_correction = 0
                while (
                    not entity_sets
                    and p4_correction < self.max_correction_rounds
                    and selected_paths
                ):
                    p4_correction += 1
                    log.append(f"Phase 4→1 Correction Round {p4_correction}")
                    # Build feedback: which paths were tried and returned nothing
                    path_descs = [p.to_text() for p in selected_paths[:3]]
                    feedback = (
                        f"Cypher execution returned 0 results for anchor '{entity_name}' "
                        f"with paths: {'; '.join(path_descs)}. "
                        f"The entity may have a different name in the KG, or the relation path is wrong."
                    )
                    diagnosis = self._diagnose_phase2_failure(kopl_program)
                    full_diag = f"{feedback}\nSchema info:\n{diagnosis}"
                    log.append(f"  Feedback: {feedback[:200]}")
                    corrected = self._generate_corrected_type_kopl(
                        question, entity_name, entity_type, target_type,
                        kopl_program, full_diag,
                    )
                    if not corrected:
                        log.append("  Correction failed")
                        break
                    corrected = self._verify_and_correct_type_path(corrected)
                    new_paths = self._hybrid_schema_search(corrected)
                    if not new_paths:
                        log.append("  Corrected KoPL still yields 0 paths")
                        break
                    pruned = self._vector_pruning(
                        question, new_paths, top_k=3, relation_hints=self._extract_relation_hints(corrected)
                    )
                    entity_sets = self._retrieve_entities(corrected, pruned)
                    log.append(f"  Re-retrieval: {len(entity_sets)} entity sets")
                    if entity_sets:
                        kopl_program = corrected
                        selected_paths = pruned
                        relation_hints = self._extract_relation_hints(corrected)

                # Phase 5: KoPL論理演算
                log.append("Phase 5: KoPL Logical Operation")
                answer_entities = self._apply_kopl_operations(kopl_program, entity_sets)
                log.append(f"  Final answer: {len(answer_entities)} entities")

                natural_answer = ""

            # Phase 5.5: Apply KoPL filters to answer entities (post-retrieval)
            # This catches cases where fallbacks (4.5a/b/c) bypassed Cypher-level filtering
            if answer_entities and self._active_filters:
                graph = self.finder.graph
                filtered = set()
                for f in self._active_filters:
                    prop = f.property_name
                    val = f.value
                    for ent in (answer_entities if isinstance(answer_entities, (set, list)) else []):
                        try:
                            cypher_filter = f"MATCH (n) WHERE toLower(n.name) = toLower($name) AND n.{prop} = $val RETURN n.name AS name LIMIT 1"
                            result = graph.run(cypher_filter, name=ent, val=val).data()
                            if result:
                                filtered.add(result[0]["name"])
                        except Exception as e:
                            print(f"[Filter] Error for {ent}: {e}")
                log.append(f"Phase 5.5: KoPL filter applied ({len(answer_entities)} -> {len(filtered)})")
                answer_entities = filtered if isinstance(answer_entities, set) else list(filtered)

            return ExtendedTypeKoPLResult(
                question=question,
                kopl_program=kopl_program,
                candidate_paths=candidate_paths,
                selected_paths=selected_paths,
                entity_sets=entity_sets,
                answer_entities=answer_entities,
                natural_answer=natural_answer,
                processing_log=log,
            )
        finally:
            # Restore global schema if it was replaced by a local one
            if original_schema is not None:
                self.schema = original_schema
            self._using_local_schema = False

    # ─── Local schema construction for large-scale KGs ────────────
    _LOCAL_SCHEMA_THRESHOLD = 200  # edge count above which we switch to local schema

    def _build_local_schema(
        self, entity_name: str, max_hops: int = 2, max_edges: int = 100,
        question: str = None,
    ) -> Optional[SchemaGraph]:
        """Build a question-specific SchemaGraph from the 2-hop neighbourhood
        of *entity_name* in the actual KG.

        This avoids pre-loading all 6000+ relations for large KGs like WebQSP.
        Returns ``None`` when no edges are found (entity not in KG, etc.).

        When the neighbourhood produces more than *max_edges* schema patterns,
        only the most frequent ones are kept.  1-hop edges (directly connected
        to the anchor entity) are always retained regardless of the limit.
        """
        if self.kg_type == "webqsp":
            local_edges = self.finder.get_local_schema_edges_cvt_collapsed(entity_name, max_hops=max_hops)
        else:
            local_edges = self.finder.get_local_schema_edges(entity_name, max_hops=max_hops)
        if not local_edges:
            return None

        # Separate 1-hop (always keep) from 2-hop (prune by frequency)
        one_hop = [(src, rel, tgt, d) for src, rel, tgt, d, freq, is_1 in local_edges if is_1]
        two_hop = [(src, rel, tgt, d, freq) for src, rel, tgt, d, freq, is_1 in local_edges if not is_1]

        # Deduplicate 1-hop edges (same pattern may appear with different freqs)
        seen: set = set()
        kept: list = []
        for src, rel, tgt, d in one_hop:
            key = (src, rel, tgt)
            if key not in seen:
                seen.add(key)
                kept.append((src, rel, tgt, d))

        # Collect ALL unique edges (1-hop + 2-hop) before applying any cap
        # This ensures question-dependent distillation can access all candidates
        for src, rel, tgt, d, freq in two_hop:
            key = (src, rel, tgt)
            if key not in seen:
                seen.add(key)
                kept.append((src, rel, tgt, d))

        # Question-dependent schema distillation: split budget between
        # frequency pool and relevance pool so that semantically relevant
        # but low-frequency relations are not dropped.
        if question and len(kept) > max_edges:
            freq_budget = max_edges // 2
            freq_pool = kept[:freq_budget]  # already sorted by freq
            remaining = kept[freq_budget:]

            # Embed question and remaining relation NL forms
            question_vec = np.array(self.embeddings.embed_query(question))
            rel_texts = []
            for src, rel, tgt, d in remaining:
                # For CVT-collapsed relations, use last segment
                base_rel = rel.split("..")[-1] if ".." in rel else rel
                nl_form, _ = auto_generate_relation_nl(base_rel)
                rel_texts.append(nl_form)

            if rel_texts:
                rel_vecs = self.embeddings.embed_documents(rel_texts)
                scores = []
                for idx, vec in enumerate(rel_vecs):
                    vec = np.array(vec)
                    cos = float(
                        np.dot(question_vec, vec)
                        / (np.linalg.norm(question_vec) * np.linalg.norm(vec) + 1e-9)
                    )
                    scores.append((idx, cos))
                scores.sort(key=lambda x: -x[1])

                relevance_budget = max_edges - freq_budget
                relevance_pool = [remaining[idx] for idx, _ in scores[:relevance_budget]]
            else:
                relevance_pool = []

            # Merge freq_pool + relevance_pool, deduplicated
            seen_keys = set((s, r, t) for s, r, t, d in freq_pool)
            final = list(freq_pool)
            for edge in relevance_pool:
                key = (edge[0], edge[1], edge[2])
                if key not in seen_keys:
                    seen_keys.add(key)
                    final.append(edge)
            kept = final[:max_edges]

            logger.info(
                "Schema distillation: %d freq + %d relevance -> %d kept",
                len(freq_pool), len(relevance_pool), len(kept),
            )
        else:
            # No question or small schema: cap by frequency (original behavior)
            if len(kept) > max_edges:
                kept = kept[:max_edges]

        logger.info(
            "Local schema for '%s': %d total -> %d kept (%d 1-hop, %d 2-hop)",
            entity_name, len(local_edges), len(kept),
            sum(1 for _, _, _, _, _, h in local_edges if h),
            sum(1 for _, _, _, _, _, h in local_edges if not h),
        )

        local_schema = SchemaGraph()
        # Build frequency map from original edges for schema distillation
        freq_map: Dict[Tuple[str, str, str], int] = {}
        for src, rel, tgt, d, freq, is_1 in local_edges:
            key = (src, rel, tgt)
            freq_map[key] = max(freq_map.get(key, 0), freq)
        for src, rel, tgt, direction in kept:
            local_schema.add_edge(src, rel, tgt, direction)
            local_schema.edge_freq[(src, rel, tgt)] = freq_map.get((src, rel, tgt), 0)
        local_schema.compute_apsp()
        return local_schema

    def _needs_local_schema(self) -> bool:
        """Return True when the global schema is too large or empty (unknown KG).

        When use_schema_relations=False (--no-schema), the user explicitly
        opted out of schema info, so we should NOT build a local schema.
        Local schema is only for KGs without a pre-defined schema (e.g. WebQSP).
        """
        if not self.use_schema_relations:
            return False
        return (
            len(self.schema.edges) > self._LOCAL_SCHEMA_THRESHOLD
            or len(self.schema.edges) == 0
        )

    def _get_available_relations(self) -> str:
        """スキーマグラフから利用可能なリレーション一覧を自動生成する"""
        if not hasattr(self, 'schema') or not self.schema:
            return self.kgc.available_relations_text
        # Respect KGConfig's deliberate choice to omit relation list
        # (PrimeKGQA/MetaQA rely on LLM's parametric knowledge + examples).
        # For local/dynamic schemas (e.g. WebQSP), always generate.
        using_local = getattr(self, '_using_local_schema', False)
        if not self.kgc.available_relations_text and not using_local:
            # Schema distillation: inject full relation list for small schemas
            if self.schema_distill and len(self.schema.edges) > 0:
                return self._format_schema_relations_with_nl()
            return ""

        local_schema = using_local
        max_relations = 50 if local_schema else 0  # 0 = unlimited

        # Schema distillation for local schemas: sort by question relevance
        if self.schema_distill and local_schema and hasattr(self, '_current_question'):
            return self._format_local_schema_distilled()

        # スキーマグラフのエッジ情報から動的に生成（順方向のみ）
        if local_schema:
            lines = ["Available relations for this question (from KG neighbourhood):"]
        else:
            lines = ["Available relations (src_type)-[RELATION]->(tgt_type):"]

        seen = set()
        count = 0
        for src_type in sorted(self.schema.adjacency):
            for tgt_type, rel, direction in self.schema.adjacency[src_type]:
                if direction != "->":
                    continue
                key = (src_type, rel, tgt_type)
                if key not in seen:
                    seen.add(key)
                    count += 1
                    if max_relations and count > max_relations:
                        continue
                    # For CVT-collapsed relations (a..b), show the last segment's NL form
                    # so the LLM sees "spouse" instead of "people.person.spouse_s..people.marriage.spouse"
                    if ".." in rel:
                        # Use the second (target-side) relation for display
                        display_rel = rel.split("..")[-1]
                        nl_form, _ = auto_generate_relation_nl(display_rel)
                        lines.append(f"  ({src_type})-[{nl_form}]->({tgt_type})")
                    else:
                        lines.append(f"  ({src_type})-[{rel}]->({tgt_type})")

        if not seen:
            return self.kgc.available_relations_text

        if max_relations and count > max_relations:
            lines.append(f"  ... and {count - max_relations} more relations")

        result = "\n".join(lines)

        # Append KG-specific semantic notes
        if self.kg_type == "pcqa":
            result += "\nNOTE: INHIBITION_TO means a drug inhibits/targets a gene. Use this when asking about drugs for a gene's mutations."

        return result

    def _format_schema_relations_with_nl(self) -> str:
        """Small-schema distillation: format full relation list with NL forms.

        For KGs like PrimeKGQA/MetaQA where the schema is small (12-25 edges),
        inject the complete relation list so the LLM knows exactly which
        type-relation-type triples exist.
        """
        lines = ["Available relations (src_type)-[relation / human-readable]->(tgt_type):"]
        seen = set()
        for src, rel, tgt, direction in self.schema.edges:
            key = (src, rel, tgt)
            if key in seen:
                continue
            seen.add(key)
            # NL form: try kgc mapping first, then auto-generate
            nl_forms = self.kgc.relation_to_nl.get(rel) if self.kgc.relation_to_nl else None
            if nl_forms:
                noun = nl_forms[0] if isinstance(nl_forms, tuple) else nl_forms
            else:
                noun, _ = auto_generate_relation_nl(rel)
            lines.append(f"  ({src})-[{rel} / {noun}]->({tgt})")
        return "\n".join(lines)

    def _format_local_schema_distilled(self) -> str:
        """Local-schema distillation: sort relations by question relevance, add NL + freq.

        For large KGs (WebQSP) where the local schema has many edges,
        rank relations by embedding similarity to the question so the
        most relevant ones appear first in the prompt.
        """
        # Collect forward edges with metadata
        edges = []
        seen = set()
        for src_type in self.schema.adjacency:
            for tgt_type, rel, direction in self.schema.adjacency[src_type]:
                if direction != "->":
                    continue
                key = (src_type, rel, tgt_type)
                if key in seen:
                    continue
                seen.add(key)
                # NL form
                if ".." in rel:
                    display_rel = rel.split("..")[-1]
                    nl_form, _ = auto_generate_relation_nl(display_rel)
                else:
                    nl_form, _ = auto_generate_relation_nl(rel)
                freq = self.schema.edge_freq.get(key, 0)
                edges.append((src_type, rel, tgt_type, nl_form, freq))

        if not edges:
            return ""

        # Sort by embedding similarity to question
        question = getattr(self, '_current_question', '')
        if question and edges:
            try:
                q_vec = np.array(self.embeddings.embed_query(question))
                rel_texts = [f"{nl} from {s} to {t}" for s, _, t, nl, _ in edges]
                rel_vecs = self.embeddings.embed_documents(rel_texts)
                scores = []
                for i, vec in enumerate(rel_vecs):
                    vec = np.array(vec)
                    cos = float(np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-9))
                    scores.append(cos)
                # Sort by descending similarity
                ranked = sorted(range(len(edges)), key=lambda i: scores[i], reverse=True)
                edges = [edges[i] for i in ranked]
            except Exception as e:
                logger.warning("Schema distillation embedding sort failed: %s", e)

        lines = ["Available relations for this question (ranked by relevance):"]
        max_show = 50
        for i, (src, rel, tgt, nl, freq) in enumerate(edges[:max_show]):
            freq_hint = f"  [freq={freq}]" if freq > 0 else ""
            lines.append(f"  ({src})-[{nl} / {rel}]->({tgt}){freq_hint}")
        if len(edges) > max_show:
            lines.append(f"  ... and {len(edges) - max_show} more relations")
        return "\n".join(lines)

    def _build_kopl_prompt(
        self,
        question: str,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> str:
        """Type-KoPL生成用のプロンプトを構築する（bak版準拠）"""
        type_list = ", ".join(sorted(self.schema.types))
        available_relations = self._get_available_relations()

        entity_info = ""
        if entity_name:
            entity_info = f"Known entity: {entity_name}"
            if entity_type:
                entity_info += f" (type: {entity_type})"
            if target_type:
                entity_info += (
                    f"\nTarget type (what the question asks for): {target_type}"
                )

        # Few-shot例の選択
        if self.few_shot_pool is not None:
            domain_labels = {
                "primekgqa": "PrimeKGQA (Biomedical domain)",
                "metaqa": "MetaQA (Movie domain)",
                "pcqa": "PcQA (Pan-cancer QA domain)",
            }
            examples = self.few_shot_pool.format_examples_prompt(
                query=question,
                k=self.few_shot_k,
                domain_label=domain_labels.get(self.kg_type, ""),
            )
        else:
            examples = self._get_prompt_examples()

        # フィルタ可能プロパティ情報を追加
        filter_info = ""
        if self.kgc.filterable_properties:
            filter_lines = []
            for node_type, props in self.kgc.filterable_properties.items():
                for prop_name, prop_info in props.items():
                    vals = prop_info.get("values")
                    if vals:
                        filter_lines.append(
                            f"  {node_type}.{prop_name} (values: {', '.join(vals)})"
                        )
                    else:
                        filter_lines.append(f"  {node_type}.{prop_name} (string)")
            filter_info = f"""
Filterable properties (use filters array when the question mentions these):
{chr(10).join(filter_lines)}
"""

        return f"""Convert this question into an Atomic Type-KoPL program.

Question: {question}
{entity_info}

Available node types: {type_list}

{available_relations}
{filter_info}
CRITICAL RULES:
1. Each operation represents ONE HOP in the path. For N-hop queries, provide exactly N operations.
2. Each operation specifies: src_type, tgt_type, relation, anchor_name (if applicable)
3. For PATH queries: operations are connected (op[i].tgt_type == op[i+1].src_type)
4. For INTERSECTION queries: each operation has its own anchor_name, final_operation="intersection"
5. Break down the question step by step. Each operation = one hop. Do NOT combine multiple steps into one semantic description.
6. Build path from anchor to answer. The relation nearest to [anchor] in the question is the FIRST hop, not the last.

{examples}

Return a JSON object with operations array, final_operation, and optionally filters array."""

    def _get_prompt_examples(self) -> str:
        """プロンプト例を返す（FewShotPool優先、なければKGConfigから静的例）"""
        if self.few_shot_pool and hasattr(self, '_current_question'):
            dynamic = self.few_shot_pool.format_examples_prompt(
                self._current_question,
                k=self.few_shot_k,
            )
            if dynamic:
                return dynamic
        return self.kgc.kopl_examples

    def _generate_type_kopl(
        self,
        question: str,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> Optional[KoPLOperation]:
        """Phase 1: LLMでType-KoPLプログラムを生成（Atomic形式）

        n_kopl_candidates > 1 の場合、temperature > 0 で複数候補を生成し、
        スキーマ整合スコアが最も高い候補を選択する。
        """

        prompt = self._build_kopl_prompt(question, entity_name, entity_type, target_type)

        n = self.n_kopl_candidates
        if n <= 1:
            # 単一候補: bak版と同じ直接呼び出し + パース
            llm_with_output = self.llm.with_structured_output(AtomicKoPLProgramSchema)
            try:
                result = llm_with_output.invoke(prompt)

                # bak版と同一のパースロジック
                valid_types = self.schema.types
                type_normalizer = {t.lower(): t for t in valid_types}

                def normalize_type(t: Optional[str]) -> Optional[str]:
                    if not t:
                        return None
                    t_clean = t.strip().rstrip("}],")
                    if t_clean in valid_types:
                        return t_clean
                    t_lower = t_clean.lower()
                    if t_lower in type_normalizer:
                        return type_normalizer[t_lower]
                    return None

                is_intersection = result.final_operation.lower() == "intersection"
                anchors_in_ops = [
                    op.anchor_name for op in result.operations if op.anchor_name
                ]

                if is_intersection and len(anchors_in_ops) >= 2:
                    children = []
                    for op in result.operations:
                        if not op.anchor_name:
                            continue
                        op_src = normalize_type(op.src_type)
                        op_tgt = normalize_type(op.tgt_type)
                        op_rel = op.relation.strip() if op.relation else None
                        if not op_src or not op_tgt or not op_rel:
                            continue
                        child_op = KoPLOperation(
                            op_type=OperationType.RELATE,
                            relations=[
                                TypeRelation(src_type=op_src, tgt_type=op_tgt, relation_hint=op_rel)
                            ],
                            anchor_name=op.anchor_name,
                        )
                        children.append(child_op)
                    if children:
                        return KoPLOperation(
                            op_type=OperationType.INTERSECTION,
                            children=children,
                            filters=result.filters,
                        )

                # PATH クエリ
                atomic_relations = []
                anchor = entity_name
                for i, op in enumerate(result.operations):
                    op_src = normalize_type(op.src_type)
                    op_tgt = normalize_type(op.tgt_type)
                    op_rel = op.relation.strip() if op.relation else None
                    if not op_src or not op_tgt or not op_rel:
                        continue
                    if i == 0 and not anchor:
                        anchor = op.anchor_name
                    atomic_relations.append(
                        TypeRelation(src_type=op_src, tgt_type=op_tgt, relation_hint=op_rel)
                    )

                if not atomic_relations:
                    raise ValueError("No valid operations generated")

                return KoPLOperation(
                    op_type=OperationType.RELATE,
                    relations=atomic_relations,
                    anchor_name=anchor,
                    filters=result.filters,
                )

            except Exception as e:
                print(f"Error generating Type-KoPL: {e}")
                if entity_name:
                    return KoPLOperation(
                        op_type=OperationType.RELATE,
                        relations=[
                            TypeRelation(src_type="", tgt_type="", relation_hint=None)
                        ],
                        anchor_name=entity_name,
                    )
                return None

        if n > 1:
            # 複数候補生成: temperature > 0 で N 個生成し、スキーマ整合スコアで選択
            candidates: List[Tuple[KoPLOperation, float]] = []
            for i in range(n):
                kopl = self._invoke_and_parse_kopl(
                    prompt, entity_name, temperature=0.7
                )
                if kopl:
                    score = self._score_schema_compatibility(kopl)
                    candidates.append((kopl, score))

            if not candidates:
                return None

            # スコア降順でソートし、最良を返す
            candidates.sort(key=lambda x: x[1], reverse=True)
            best, best_score = candidates[0]
        if len(candidates) > 1:
            scores_str = ", ".join(f"{s:.2f}" for _, s in candidates)
            logger.info(
                "KoPL multi-candidate: %d generated, scores=[%s], best=%.2f",
                len(candidates), scores_str, best_score,
            )
        return best

    def _score_schema_compatibility(self, kopl: KoPLOperation) -> float:
        """KoPLプログラムのスキーマ整合スコアを計算

        各型ペアにスキーマ上のエッジが存在する割合を返す (0.0 - 1.0)。
        """
        operations = kopl.children if kopl.children else [kopl]
        total_pairs = 0
        valid_pairs = 0

        for op in operations:
            for rel in op.relations:
                if not rel.src_type or not rel.tgt_type:
                    continue
                total_pairs += 1
                available = self._get_relations_between_types(
                    rel.src_type, rel.tgt_type
                )
                if available:
                    valid_pairs += 1

        if total_pairs == 0:
            return 0.0
        return valid_pairs / total_pairs

    # ─── Phase 1→2 Correction (Feature B) ────────────

    def _diagnose_phase2_failure(self, kopl: KoPLOperation) -> str:
        """Analyze why Phase 2 returned 0 candidate paths.

        For each type pair in the KoPL, check if edges exist and list
        what types each side actually connects to.
        """
        operations = kopl.children if kopl.children else [kopl]
        parts = []
        for op in operations:
            for rel in op.relations:
                src, tgt = rel.src_type, rel.tgt_type
                if not src or not tgt:
                    continue
                available = self._get_relations_between_types(src, tgt)
                if available:
                    parts.append(f"({src})->({tgt}): OK ({len(available)} relations)")
                    continue
                # No edges: show what each type connects to
                src_neighbors = set()
                for neighbor, r, d in self.schema.adjacency.get(src, []):
                    src_neighbors.add(neighbor)
                tgt_neighbors = set()
                for neighbor, r, d in self.schema.adjacency.get(tgt, []):
                    tgt_neighbors.add(neighbor)
                src_in_schema = src in self.schema.types
                tgt_in_schema = tgt in self.schema.types
                diag = f"({src})->({tgt}): NO EDGES"
                if not src_in_schema:
                    diag += f" [{src} not in schema]"
                elif src_neighbors:
                    diag += f" [{src} connects to: {', '.join(sorted(src_neighbors)[:5])}]"
                if not tgt_in_schema:
                    diag += f" [{tgt} not in schema]"
                elif tgt_neighbors:
                    diag += f" [{tgt} reached from: {', '.join(sorted(tgt_neighbors)[:5])}]"
                parts.append(diag)
        return "\n".join(parts) if parts else "No type pairs to diagnose"

    def _serialize_kopl_for_prompt(self, kopl: KoPLOperation) -> str:
        """Format a KoPLOperation as readable text for inclusion in a correction prompt."""
        operations = kopl.children if kopl.children else [kopl]
        ops_list = []
        for op in operations:
            for rel in op.relations:
                entry = {"src_type": rel.src_type, "tgt_type": rel.tgt_type, "relation": rel.relation_hint or ""}
                if op.anchor_name:
                    entry["anchor_name"] = op.anchor_name
                ops_list.append(str(entry))
        final_op = kopl.op_type.value if kopl.op_type else "relate"
        return f"operations: [{', '.join(ops_list)}]\nfinal_operation: \"{final_op}\""

    def _generate_corrected_type_kopl(
        self,
        question: str,
        entity_name: Optional[str],
        entity_type: Optional[str],
        target_type: Optional[str],
        failed_kopl: KoPLOperation,
        diagnosis: str,
    ) -> Optional[KoPLOperation]:
        """Generate a corrected Type-KoPL program using diagnostic feedback."""
        type_list = ", ".join(sorted(self.schema.types))
        available_relations = self._get_available_relations()
        entity_info = ""
        if entity_name:
            entity_info = f"Known entity: {entity_name}"
            if entity_type:
                entity_info += f" (type: {entity_type})"
            if target_type:
                entity_info += f"\nTarget type (what the question asks for): {target_type}"

        examples = self._get_prompt_examples()
        serialized = self._serialize_kopl_for_prompt(failed_kopl)

        prompt = f"""Your previous Type-KoPL program for this question produced no valid paths in the knowledge graph.

Question: {question}
{entity_info}

Available node types: {type_list}

{available_relations}

Your previous answer:
{serialized}

Diagnosis (why it failed):
{diagnosis}

Please generate a CORRECTED Atomic Type-KoPL program. Fix the type names and relation hints based on the diagnosis above. Use only types that appear in "Available node types".

{examples}

Return a JSON object with operations array, final_operation, and optionally filters."""

        return self._invoke_and_parse_kopl(prompt, entity_name, temperature=0)

    def _invoke_and_parse_kopl(
        self,
        prompt: str,
        entity_name: Optional[str],
        temperature: float = 0,
    ) -> Optional[KoPLOperation]:
        """LLMを呼び出してAtomicKoPLProgramSchemaをパースし、KoPLOperationに変換する"""

        from langchain.chat_models import init_chat_model

        if temperature > 0:
            llm = init_chat_model(
                self.llm.model_name if hasattr(self.llm, 'model_name') else self.llm.model,
                model_provider="openai",
                temperature=temperature,
                max_tokens=8192,
            )
            llm_with_output = llm.with_structured_output(AtomicKoPLProgramSchema)
        else:
            llm_with_output = self.llm.with_structured_output(AtomicKoPLProgramSchema)

        try:
            result = llm_with_output.invoke(prompt)

            # 有効なタイプのセット（名寄せ用にlower -> 正式名のマッピングを作成）
            valid_types = self.schema.types
            type_normalizer = {t.lower(): t for t in valid_types}

            def normalize_type(t: Optional[str]) -> Optional[str]:
                """タイプ名を正規化（case-insensitive マッチング + alias fallback）"""
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
                # TYPE_ALIAS_MAP fallback: e.g., "person" -> "us_president" if in schema
                alias_target = TYPE_ALIAS_MAP.get(t_lower)
                if alias_target and alias_target in type_normalizer:
                    return type_normalizer[alias_target]
                # Reverse alias: if LLM outputs a canonical type like "Person"
                # that maps to multiple subtypes, pick the first one in schema
                for alias_key, alias_val in TYPE_ALIAS_MAP.items():
                    if alias_val == t_lower and alias_key in type_normalizer:
                        return type_normalizer[alias_key]
                return None

            # final_operationをチェック
            is_intersection = result.final_operation.lower() == "intersection"

            # 各操作のanchor_nameをチェックしてintersectionかどうかを判定
            anchors_in_ops = [
                op.anchor_name for op in result.operations if op.anchor_name
            ]

            # INTERSECTIONクエリの処理（複数アンカー）
            if is_intersection and len(anchors_in_ops) >= 2:
                children = []
                for op in result.operations:
                    if not op.anchor_name:
                        continue
                    op_src = normalize_type(op.src_type)
                    op_tgt = normalize_type(op.tgt_type)
                    op_rel = op.relation.strip() if op.relation else None

                    if not op_src or not op_tgt or not op_rel:
                        continue

                    child_op = KoPLOperation(
                        op_type=OperationType.RELATE,
                        relations=[
                            TypeRelation(
                                src_type=op_src,
                                tgt_type=op_tgt,
                                relation_hint=op_rel,
                            )
                        ],
                        anchor_name=op.anchor_name,
                    )
                    children.append(child_op)

                if children:
                    return KoPLOperation(
                        op_type=OperationType.INTERSECTION,
                        children=children,
                        filters=result.filters if result.filters else None,
                    )

            # PATH クエリの処理（単一アンカー、マルチホップ）
            # 各ホップをatomic operationとして保持
            atomic_relations = []
            # 入力のentity_nameを優先、LLM生成は fallback
            anchor_name = entity_name

            for i, op in enumerate(result.operations):
                # タイプ名を正規化
                op_src = normalize_type(op.src_type)
                op_tgt = normalize_type(op.tgt_type)
                op_rel = op.relation.strip() if op.relation else None

                if not op_src or not op_tgt or not op_rel:
                    continue

                # entity_name がない場合のみLLM生成のanchorを使用
                if i == 0 and not anchor_name:
                    anchor_name = op.anchor_name

                # 各ホップをTypeRelationとして追加
                atomic_relations.append(
                    TypeRelation(
                        src_type=op_src,
                        tgt_type=op_tgt,
                        relation_hint=op_rel,
                    )
                )

            if not atomic_relations:
                raise ValueError("No valid operations generated")

            # atomic operationsをrelationsリストとして保持
            return KoPLOperation(
                op_type=OperationType.RELATE,
                relations=atomic_relations,
                anchor_name=anchor_name,
                filters=result.filters if result.filters else None,
            )

        except Exception as e:
            print(f"Error generating Type-KoPL: {e}")
            # フォールバック
            if entity_name:
                return KoPLOperation(
                    op_type=OperationType.RELATE,
                    relations=[
                        TypeRelation(src_type="", tgt_type="", relation_hint=None)
                    ],
                    anchor_name=entity_name,
                )
            return None

    def _hybrid_schema_search(self, kopl_program: KoPLOperation) -> List[SchemaPath]:
        """Phase 2: ハイブリッド探索（Type Path単位のステップワイズBFS）

        各ステップで:
        1. BFS: 該当type間で利用可能なrelationを列挙
        2. ベクトル剪定: relation_hintとの類似度でTop-K選択
        3. 次ステップへ
        """

        all_paths = []

        # 子操作がある場合は各子操作に対して探索
        operations = kopl_program.children if kopl_program.children else [kopl_program]

        for op in operations:
            if not op.relations:
                continue

            # type pathとrelation hintsを抽出
            type_path = [op.relations[0].src_type]
            relation_hints = []
            for rel in op.relations:
                if rel.tgt_type:
                    type_path.append(rel.tgt_type)
                if rel.relation_hint:
                    relation_hints.append(rel.relation_hint)

            # ステップワイズBFS + ベクトル剪定
            paths = self._stepwise_bfs_with_pruning(type_path, relation_hints)
            all_paths.extend(paths)

        # 重複除去
        seen = set()
        unique_paths = []
        for p in all_paths:
            key = (tuple(p.types), tuple(p.relations))
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        return unique_paths

    def _find_paths_between_types(
        self, src_type: str, tgt_type: str, max_depth: int = 2
    ) -> List[SchemaPath]:
        """2つのtype間のパスを探索（同じタイプ間も対応）

        Args:
            src_type: 開始タイプ
            tgt_type: 終了タイプ
            max_depth: 最大探索深さ

        Returns:
            SchemaPath のリスト
        """
        # スキーマなしモードの場合、KGから動的にパスを探索
        if not self.use_schema_relations:
            return self._find_paths_between_types_from_kg(src_type, tgt_type, max_depth)

        # src_type == tgt_type の場合は特別処理（2-hopパスを探す）
        if src_type == tgt_type:
            # 中間ノードを経由するパスを探す
            paths = []
            for neighbor, rel1, dir1 in self.schema.adjacency.get(src_type, []):
                if neighbor != src_type:  # 自己ループは除外
                    # neighbor から tgt_type への逆方向のリレーションを探す
                    for neighbor2, rel2, dir2 in self.schema.adjacency.get(
                        neighbor, []
                    ):
                        if neighbor2 == tgt_type:
                            # src -> neighbor -> tgt のパス
                            paths.append(
                                SchemaPath(
                                    types=[src_type, neighbor, tgt_type],
                                    relations=[rel1, rel2],
                                    source="expand-bfs",
                                    directions=[dir1, dir2],
                                )
                            )
            return paths
        else:
            # 異なるタイプ間は通常のfind_shortest_pathsを使用
            return self.schema.find_shortest_paths(src_type, tgt_type)

    def _find_paths_between_types_from_kg(
        self, src_type: str, tgt_type: str, max_depth: int = 2
    ) -> List[SchemaPath]:
        """KGから動的に2つのtype間のパスを探索（スキーマなしモード用）"""
        paths = []

        # まず直接リレーションを確認
        direct_relations = self._get_relations_between_types(src_type, tgt_type)
        if direct_relations:
            for rel, direction in direct_relations:
                paths.append(
                    SchemaPath(
                        types=[src_type, tgt_type],
                        relations=[rel],
                        source="kg-direct",
                        directions=[direction],
                    )
                )
            return paths

        # 直接リレーションがない場合、中間タイプを経由するパスを探す
        if max_depth >= 2:
            # 全タイプを取得してキャッシュ
            if not hasattr(self, "_all_types_cache"):
                self._all_types_cache = set(self.finder.get_all_types())

            for mid_type in self._all_types_cache:
                if mid_type == src_type and mid_type == tgt_type:
                    continue

                # src -> mid のリレーション
                src_to_mid = self._get_relations_between_types(src_type, mid_type)
                if not src_to_mid:
                    continue

                # mid -> tgt のリレーション
                mid_to_tgt = self._get_relations_between_types(mid_type, tgt_type)
                if not mid_to_tgt:
                    continue

                # パスを構築（各リレーションの最初の候補を使用）
                rel1, dir1 = src_to_mid[0]
                rel2, dir2 = mid_to_tgt[0]
                paths.append(
                    SchemaPath(
                        types=[src_type, mid_type, tgt_type],
                        relations=[rel1, rel2],
                        source="kg-2hop",
                        directions=[dir1, dir2],
                    )
                )

        return paths

    def _expand_type_path(
        self,
        type_path: List[str],
        relation_hints: List[str],
    ) -> Tuple[List[str], List[str]]:
        """直接リレーションがないtype間をBFSで補完してtype_pathを展開

        Args:
            type_path: LLMが予測したtype path（例: [Movie, Person, Person]）
            relation_hints: 各ステップのrelation hint

        Returns:
            expanded_type_path: 展開後のtype path（例: [Movie, Person, Movie, Person]）
            expanded_hints: 展開後のrelation hints
        """
        if len(type_path) < 2:
            return type_path, relation_hints

        expanded_types = [type_path[0]]
        expanded_hints = []

        for i in range(len(type_path) - 1):
            src_type = type_path[i]
            tgt_type = type_path[i + 1]
            hint = relation_hints[i] if i < len(relation_hints) else None

            # 直接リレーションがあるか確認
            direct_relations = self._get_relations_between_types(src_type, tgt_type)

            if direct_relations:
                # 直接リレーションがある場合はそのまま
                expanded_types.append(tgt_type)
                expanded_hints.append(hint)
            else:
                # 直接リレーションがない場合はBFSで中間パスを探索
                intermediate_paths = self._find_paths_between_types(src_type, tgt_type)

                if intermediate_paths:
                    # 最短パスを使用（複数ある場合は最初のもの）
                    best_path = intermediate_paths[0]
                    # 中間タイプを追加（src_typeは既に追加済みなのでスキップ）
                    for j, t in enumerate(best_path.types[1:]):
                        expanded_types.append(t)
                        # 中間ステップのhintは元のhintを分割して使用するか、Noneにする
                        expanded_hints.append(
                            hint if j == len(best_path.types) - 2 else None
                        )
                else:
                    # BFSでもパスが見つからない場合はそのまま（後で失敗する）
                    expanded_types.append(tgt_type)
                    expanded_hints.append(hint)

        return expanded_types, expanded_hints

    def _stepwise_bfs_with_pruning(
        self,
        type_path: List[str],
        relation_hints: List[str],
        top_k_per_step: int = 3,
    ) -> List[SchemaPath]:
        """Type Path単位のステップワイズBFS + ベクトル剪定

        Args:
            type_path: 予測されたtype path（例: [Movie, Person, Movie, Person]）
            relation_hints: 各ステップのrelation hint（例: [DIRECTED_BY, DIRECTED_BY, STARRED_ACTORS]）
            top_k_per_step: 各ステップで残す候補数
        """
        if len(type_path) < 2:
            return []

        # 直接リレーションがないtype間をBFSで補完
        expanded_type_path, expanded_hints = self._expand_type_path(
            type_path, relation_hints
        )

        # 各ステップで候補relationを収集・剪定
        step_relations: List[List[Tuple[str, str]]] = []  # [(relation, direction), ...]

        for i in range(len(expanded_type_path) - 1):
            src_type = expanded_type_path[i]
            tgt_type = expanded_type_path[i + 1]
            hint = expanded_hints[i] if i < len(expanded_hints) else None

            # BFS: src_type -> tgt_type で利用可能なrelationを列挙
            available_relations = self._get_relations_between_types(src_type, tgt_type)

            if not available_relations:
                # このステップで利用可能なrelationがない場合、空リストを返す
                return []

            # Conditional per-hop pruning: only when candidate count is large
            # Small schemas (PrimeKGQA ~5 rels/step): keep all (pruning drops valid rels)
            # Large schemas (WebQSP ~50+ rels/step): prune to top_k (600+ candidates otherwise)
            if hint and len(available_relations) > 15:
                pruned = self._prune_relations_by_hint(
                    available_relations, hint, top_k=top_k_per_step
                )
                step_relations.append(pruned)
            else:
                step_relations.append(available_relations)

        # 各ステップの候補を組み合わせてパスを構築
        return self._build_paths_from_steps(expanded_type_path, step_relations)

    def _verify_and_correct_type_path(
        self, kopl_program: KoPLOperation
    ) -> KoPLOperation:
        """Phase 1.5: Verify that each type pair in the KoPL program has edges in schema.

        If a predicted type pair has no edges:
        1. Try swapping src/tgt direction
        2. Try finding a replacement type that has edges to the other type

        This prevents "Found 0 candidate paths" in Phase 2.
        """
        # Gather all operations (children for intersection, else the program itself)
        operations = kopl_program.children if kopl_program.children else [kopl_program]

        for op in operations:
            for rel in op.relations:
                src = rel.src_type
                tgt = rel.tgt_type
                if not src or not tgt:
                    continue

                # Check if this type pair has any relations in schema
                available = self._get_relations_between_types(src, tgt)
                if available:
                    continue  # OK, edges exist

                # Try swap: if _get_relations_between_types already checks both
                # directions via adjacency, a swap wouldn't help for schema mode.
                # But for schema-less mode (KG query), direction matters.
                available_rev = self._get_relations_between_types(tgt, src)
                if available_rev:
                    rel.src_type, rel.tgt_type = rel.tgt_type, rel.src_type
                    continue

                # Neither direction works — leave as-is.
                # Replacing with a different type is too risky (causes regressions).

        return kopl_program

    def _get_relations_between_types(
        self, src_type: str, tgt_type: str
    ) -> List[Tuple[str, str]]:
        """2つのtype間で利用可能なrelationを列挙（方向付き）

        Returns:
            List of (relation_name, direction) tuples
        """
        # スキーマなしモードの場合、KGから動的に取得
        if not self.use_schema_relations:
            cache_key = (src_type, tgt_type)
            if cache_key not in self._relation_cache:
                self._relation_cache[cache_key] = (
                    self.finder.get_relations_between_types(src_type, tgt_type)
                )
            return self._relation_cache[cache_key]

        # スキーマありモードの場合、既存のロジック
        relations = []

        # 順方向: src -> tgt
        for neighbor, rel, direction in self.schema.adjacency.get(src_type, []):
            if neighbor == tgt_type:
                relations.append((rel, direction))

        # 逆方向: tgt -> src (実際のエッジは逆だが、クエリ方向としては src -> tgt)
        for neighbor, rel, direction in self.schema.adjacency.get(tgt_type, []):
            if neighbor == src_type:
                # 逆方向なので direction を反転
                reversed_dir = "<-" if direction == "->" else "->"
                relations.append((rel, reversed_dir))

        # 重複除去
        return list(set(relations))

    def _prune_relations_by_hint(
        self,
        relations: List[Tuple[str, str]],
        hint: str,
        top_k: int,
    ) -> List[Tuple[str, str]]:
        """relation hintとの類似度でTop-K選択"""
        if not relations:
            return []

        # 完全一致があればそれを優先
        for rel, direction in relations:
            if rel.lower() == hint.lower():
                return [(rel, direction)]

        # ベクトル類似度で選択
        hint_vec = np.array(self.embeddings.embed_query(hint))
        rel_texts = [rel for rel, _ in relations]
        rel_vecs = self.embeddings.embed_documents(rel_texts)

        scored = []
        for i, (rel, direction) in enumerate(relations):
            rel_vec = np.array(rel_vecs[i])
            similarity = np.dot(hint_vec, rel_vec) / (
                np.linalg.norm(hint_vec) * np.linalg.norm(rel_vec) + 1e-8
            )
            scored.append((rel, direction, similarity))

        # スコア順にソートしてTop-K
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(rel, direction) for rel, direction, _ in scored[:top_k]]

    def _build_paths_from_steps(
        self,
        type_path: List[str],
        step_relations: List[List[Tuple[str, str]]],
    ) -> List[SchemaPath]:
        """各ステップの候補relationからパスを構築"""
        if not step_relations:
            return []

        # 最初のステップで初期化
        current_paths = []
        for rel, direction in step_relations[0]:
            current_paths.append(
                SchemaPath(
                    types=[type_path[0], type_path[1]],
                    relations=[rel],
                    source="stepwise-bfs",
                    directions=[direction],
                )
            )

        # 2番目以降のステップを追加
        for step_idx in range(1, len(step_relations)):
            new_paths = []
            next_type = type_path[step_idx + 1]

            for path in current_paths:
                for rel, direction in step_relations[step_idx]:
                    new_path = SchemaPath(
                        types=path.types + [next_type],
                        relations=path.relations + [rel],
                        source="stepwise-bfs",
                        directions=path.directions + [direction],
                    )
                    new_paths.append(new_path)

            current_paths = new_paths

        return current_paths

    def _compute_hint_match_score(
        self,
        path: SchemaPath,
        relation_hints: List[str],
    ) -> float:
        """relation_hintsとパスのマッチ度を計算（0-1の範囲）

        完全一致の場合は1.0、部分一致の場合は一致数/hints数
        Uses RELATION_TO_NATURAL_LANGUAGE mapping for better semantic matching.
        """
        if not relation_hints or not path.relations:
            return 0.0

        # パスのrelationsを正規化（自然言語形式を使用）
        path_rels_normalized = []
        for r in path.relations:
            noun, verb = get_nl_forms(r)
            # noun and verb forms for matching
            path_rels_normalized.append(
                (r.lower().replace("_", " "), noun.lower(), verb.lower())
            )

        hints_normalized = [h.lower().replace("_", " ") for h in relation_hints]

        # 順序を考慮してマッチングをカウント
        matches = 0
        for i, hint in enumerate(hints_normalized):
            if i < len(path_rels_normalized):
                raw_rel, noun, verb = path_rels_normalized[i]
                # Check all forms for match
                for form in [raw_rel, noun, verb]:
                    if hint == form or hint in form or form in hint:
                        matches += 1
                        break

        return matches / len(hints_normalized)

    def _vector_pruning(
        self,
        question: str,
        candidate_paths: List[SchemaPath],
        top_k: Optional[int] = None,
        relation_hints: Optional[List[str]] = None,
    ) -> List[SchemaPath]:
        """Phase 3: 質問文とのベクトル類似度でTop-K選択

        Args:
            question: 質問文
            candidate_paths: 候補パス
            top_k: 選択するパス数
            relation_hints: KoPLから抽出されたrelation hints（per-hop cosine用）
        """

        if not candidate_paths:
            return []

        if top_k is None:
            top_k = self.top_k_paths

        # 質問文のベクトル化
        question_vec = np.array(self.embeddings.embed_query(question))

        # パステキストをバッチでベクトル化（効率化）
        # 自然言語表現を使用してベクトル剪定の精度を向上
        path_texts = [path.to_natural_language() for path in candidate_paths]
        path_vecs = self.embeddings.embed_documents(path_texts)

        # hintsがある場合、per-hop cosine用にhintとrelationの埋め込みを事前計算
        hint_vecs = None
        rel_vec_cache: Dict[str, np.ndarray] = {}
        if relation_hints:
            # hint を自然言語形式に正規化して埋め込み
            hint_texts = [h.lower().replace("_", " ") for h in relation_hints]
            hint_vecs = [
                np.array(v) for v in self.embeddings.embed_documents(hint_texts)
            ]
            # 候補パス中の全ユニークrelationを収集・埋め込み
            unique_rels = set()
            for path in candidate_paths:
                for r in path.relations:
                    unique_rels.add(r)
            if unique_rels:
                unique_rels_list = list(unique_rels)
                # 自然言語形式で埋め込み（hintとの表層差を吸収）
                rel_nl_texts = [get_nl_forms(r)[0] for r in unique_rels_list]
                rel_embeddings = self.embeddings.embed_documents(rel_nl_texts)
                for rel, vec in zip(unique_rels_list, rel_embeddings):
                    rel_vec_cache[rel] = np.array(vec)

        # 各パスのスコアを計算
        for i, path in enumerate(candidate_paths):
            path_vec = np.array(path_vecs[i])
            # コサイン類似度（質問文 vs パステキスト）
            similarity = float(
                np.dot(question_vec, path_vec)
                / (np.linalg.norm(question_vec) * np.linalg.norm(path_vec) + 1e-8)
            )

            # hintsがある場合、per-hop cosine類似度でブースト
            if hint_vecs and rel_vec_cache:
                hop_scores = []
                for j, rel in enumerate(path.relations):
                    if j < len(hint_vecs) and rel in rel_vec_cache:
                        h_vec = hint_vecs[j]
                        r_vec = rel_vec_cache[rel]
                        hop_cos = float(
                            np.dot(h_vec, r_vec)
                            / (np.linalg.norm(h_vec) * np.linalg.norm(r_vec) + 1e-8)
                        )
                        hop_scores.append(hop_cos)
                if hop_scores:
                    avg_hop_cosine = sum(hop_scores) / len(hop_scores)
                    path.score = similarity + avg_hop_cosine
                else:
                    path.score = similarity
            else:
                path.score = similarity

        # スコア順にソートしてTop-K
        candidate_paths.sort(key=lambda p: p.score, reverse=True)
        return candidate_paths[:top_k]

    def _retrieve_entities(
        self, kopl_program: KoPLOperation, selected_paths: List[SchemaPath]
    ) -> List[EntitySet]:
        """Phase 4: Cypherでエンティティ取得

        APSPにより選択されたパスはすでに適切な長さなので、
        各アンカーに対して全選択パスを実行する。

        Note: _try_direct_paths fallback is disabled for intersection queries
        because direct 1-hop paths from each anchor can produce different
        relation types, inflating false positives in the set intersection.
        """

        graph = self.finder.graph
        entity_sets = []
        is_intersection = kopl_program.op_type == OperationType.INTERSECTION

        operations = kopl_program.children if kopl_program.children else [kopl_program]

        for op in operations:
            anchor_name = op.anchor_name

            if not anchor_name:
                continue

            child_entities: Set[str] = set()
            for path in selected_paths:
                entities = self._execute_cypher_for_path(graph, path, anchor_name)
                if not entities and self.use_llm_cypher:
                    entities = self._execute_cypher_llm_generated(path, anchor_name, self._current_question)
                if entities:
                    child_entities.update(entities)
                    entity_sets.append(EntitySet(entities=entities, source_path=path, anchor_name=anchor_name))

            # _try_direct_paths fallback disabled: introduces false positives
            # and was not present in the paper-version pipeline.

        return entity_sets

    def _execute_cypher_llm_generated(
        self, path: SchemaPath, anchor_name: str, question: str
    ) -> Set[str]:
        """Phase 4 alternative: LLM generates Cypher query from schema path.

        Based on KGT paper's Subgraph_construction approach. Falls back to
        template-based Cypher on failure.
        """
        graph = self.finder.graph

        # Build schema path description
        path_desc_parts = []
        for i, t in enumerate(path.types):
            path_desc_parts.append(f"({t})")
            if i < len(path.relations):
                path_desc_parts.append(f"-[{path.relations[i]}]-")
        path_desc = "".join(path_desc_parts)

        # Collect available types
        available_types = ", ".join(sorted(set(path.types)))

        prompt = f"""Generate a Cypher query to answer this question using the given schema path.

Schema path: {path_desc}
Anchor entity: {anchor_name}
Question: {question}
Available node types: {available_types}

Rules:
1. Use toLower() for entity name matching: WHERE toLower(n.name) = toLower($anchor)
2. Use undirected relationships: -[r:REL]- (not ->)
3. Return target entity names: RETURN DISTINCT target.name AS answer
4. Limit to 50 results
5. Use backticks for type/relation names containing special characters (/, space, -, .)
6. Parameter name must be $anchor for the anchor entity

Examples:
-- 1-hop query
MATCH (a:Drug)-[r:TREATMENT]-(b:Cancer)
WHERE toLower(a.name) = toLower($anchor)
RETURN DISTINCT b.name AS answer LIMIT 50

-- 1-hop query with special characters in type/relation names (MUST use backticks)
MATCH (a:drug)-[r:`enzyme`]-(target:`gene/protein`)
WHERE toLower(a.name) = toLower($anchor)
RETURN DISTINCT target.name AS answer LIMIT 50

-- 2-hop query
MATCH (a:SnvFull)-[r1:HAS_VAR]-(cc:CancerCell)-[r2:SENSITIVITY_TO]-(b:Drug)
WHERE toLower(a.name) = toLower($anchor)
RETURN DISTINCT b.name AS answer LIMIT 50

-- 2-hop query with backticks for special characters
MATCH (a:drug)-[r1:`transporter`]-(tp:`gene/protein`)-[r2:`ppi`]-(target:`gene/protein`)
WHERE toLower(a.name) = toLower($anchor)
RETURN DISTINCT target.name AS answer LIMIT 50

Generate ONLY the Cypher query, nothing else:"""

        try:
            response = self.llm.invoke(prompt)
            cypher = response.content.strip()

            # Strip markdown code fences if present
            if cypher.startswith("```"):
                lines = cypher.split("\n")
                # Remove first and last lines (fences)
                lines = [l for l in lines if not l.strip().startswith("```")]
                cypher = "\n".join(lines).strip()

            # Auto-fix: wrap type/relation names with special chars in backticks
            import re
            # Fix unquoted labels like :gene/protein -> :`gene/protein`
            cypher = re.sub(
                r':(?!`)([A-Za-z][A-Za-z0-9_]*[/\s\-.][A-Za-z0-9_/\s\-.]*)',
                lambda m: f':`{m.group(1)}`',
                cypher,
            )

            # Validate: reject dangerous operations
            cypher_upper = cypher.upper()
            dangerous_keywords = ["DELETE", "DETACH", "DROP", "CREATE", "SET ", "REMOVE", "MERGE"]
            for kw in dangerous_keywords:
                if kw in cypher_upper:
                    print(f"[LLM Cypher] Rejected dangerous query containing '{kw}': {cypher}")
                    return self._execute_cypher_for_path(graph, path, anchor_name)

            # Validate: must contain RETURN and MATCH
            if "MATCH" not in cypher_upper or "RETURN" not in cypher_upper:
                print(f"[LLM Cypher] Invalid query (missing MATCH/RETURN): {cypher}")
                return self._execute_cypher_for_path(graph, path, anchor_name)

            print(f"[LLM Cypher] Generated: {cypher}")

            records = graph.run(cypher, anchor=anchor_name).data()
            answers = {r["answer"] for r in records if r.get("answer")}

            # CVT resolution for WebQSP
            if self.kg_type == "webqsp" and answers:
                cvt_mids = {a for a in answers if a.startswith("m.")}
                if cvt_mids:
                    named_answers = answers - cvt_mids
                    for mid in cvt_mids:
                        try:
                            resolve_cypher = """
                            MATCH (cvt {name: $mid})-[r]-(target)
                            WHERE target.name IS NOT NULL AND NOT target.name STARTS WITH 'm.'
                            RETURN DISTINCT target.name AS name
                            """
                            resolved = graph.run(resolve_cypher, mid=mid).data()
                            for rec in resolved:
                                if rec["name"]:
                                    named_answers.add(rec["name"])
                        except Exception:
                            pass
                    if named_answers:
                        answers = named_answers

            if answers:
                print(f"[LLM Cypher] Retrieved {len(answers)} entities")
                return answers
            else:
                print("[LLM Cypher] No results, falling back to template Cypher")
                return self._execute_cypher_for_path(graph, path, anchor_name)

        except Exception as e:
            print(f"[LLM Cypher] Error: {e}, falling back to template Cypher")
            return self._execute_cypher_for_path(graph, path, anchor_name)

    def _get_node_labels(self, name: str) -> Set[str]:
        """Neo4jノードのラベルを取得（キャッシュ付き）"""
        cache = getattr(self, '_node_label_cache', None)
        if cache is None:
            self._node_label_cache = {}
            cache = self._node_label_cache

        name_lower = name.lower()
        if name_lower in cache:
            return cache[name_lower]

        try:
            graph = self.finder.graph
            result = graph.run(
                "MATCH (n) WHERE toLower(n.name) = toLower($name) "
                "RETURN labels(n) AS labels LIMIT 1",
                name=name,
            ).data()
            labels = set(result[0]["labels"]) if result else set()
        except Exception:
            labels = set()

        cache[name_lower] = labels
        return labels

    def _orient_path_for_anchor(
        self, path: SchemaPath, anchor_name: str
    ) -> SchemaPath:
        """アンカーのラベルがパス末端に一致する場合、パスを反転する。

        _build_cypher_for_path は常に types[0] にアンカーを配置するため、
        アンカーが実際には types[-1] 側にある場合はパスを反転して正しい
        Cypher を生成できるようにする。
        """
        if len(path.types) < 2 or path.types[0] == path.types[-1]:
            return path

        anchor_labels = self._get_node_labels(anchor_name)
        if not anchor_labels:
            return path

        start_match = path.types[0] in anchor_labels
        end_match = path.types[-1] in anchor_labels

        if end_match and not start_match:
            # パスを反転: types, relations, directions を逆順に
            rev_types = list(reversed(path.types))
            rev_relations = list(reversed(path.relations))
            rev_directions = []
            for d in reversed(path.directions if path.directions else []):
                rev_directions.append("<-" if d == "->" else "->")

            return SchemaPath(
                types=rev_types,
                relations=rev_relations,
                source=path.source,
                score=path.score,
                directions=rev_directions,
            )

        return path

    def _build_cypher_for_path(
        self, path: SchemaPath, anchor_name: str
    ) -> Tuple[Optional[str], dict]:
        """パスからCypherクエリ文字列とパラメータを構築する。

        Returns:
            (cypher_str, params) または Cypherを構築できない場合は (None, {})
        """
        # アンカーが末端側にある場合、パスを反転
        path = self._orient_path_for_anchor(path, anchor_name)

        ALIAS_TYPES = {
            "Cancer": ["Cancer", "CancerAlias"],
            "Drug": ["Drug", "DrugAlias"],
        }

        def get_label(t: str) -> str:
            if "/" in t or "." in t or " " in t:
                return f"`{t}`"
            if self.kg_type == "pcqa" and t in ALIAS_TYPES:
                return "|".join(ALIAS_TYPES[t])
            return t

        def get_rel(r: str) -> str:
            if " " in r or "-" in r or "/" in r or "." in r:
                return f"`{r}`"
            return r

        if self._compound_search_term:
            where_anchor = "toLower(a.name) CONTAINS toLower($search_term)"
            params = {"search_term": self._compound_search_term}
        else:
            where_anchor = "toLower(a.name) = toLower($anchor_name)"
            params = {"anchor_name": anchor_name}

        if len(path.types) < 2:
            return None, {}

        has_cvt_collapsed = any('..' in r for r in path.relations)

        if has_cvt_collapsed:
            parts = [f"(a:{get_label(path.types[0])})"]
            cvt_idx = 0
            type_idx = 1
            for i, rel in enumerate(path.relations):
                is_last = (i == len(path.relations) - 1)
                end_var = "b" if is_last else f"m{i}"
                end_label = get_label(path.types[type_idx])
                if '..' in rel:
                    r1, r2 = rel.split('..', 1)
                    cvt_var = f"cvt{cvt_idx}"
                    cvt_idx += 1
                    parts.append(f"-[:{get_rel(r1)}]-({cvt_var}:CVT)-[:{get_rel(r2)}]-({end_var}:{end_label})")
                else:
                    parts.append(f"-[:{get_rel(rel)}]-({end_var}:{end_label})")
                type_idx += 1
            match_pattern = "".join(parts)
            cypher = f"""
            MATCH {match_pattern}
            WHERE {where_anchor} AND a <> b
            RETURN DISTINCT b.name AS answer"""
        elif len(path.types) == 2:
            cypher = f"""
            MATCH (a:{get_label(path.types[0])})-[r:{get_rel(path.relations[0])}]-(b:{get_label(path.types[1])})
            WHERE {where_anchor} AND a <> b
            RETURN DISTINCT b.name AS answer"""
        elif len(path.types) == 3:
            cypher = f"""
            MATCH (a:{get_label(path.types[0])})-[r1:{get_rel(path.relations[0])}]-(mid:{get_label(path.types[1])})-[r2:{get_rel(path.relations[1])}]-(b:{get_label(path.types[2])})
            WHERE {where_anchor} AND a <> mid AND mid <> b AND a <> b
            RETURN DISTINCT b.name AS answer"""
        elif len(path.types) == 4:
            cypher = f"""
            MATCH (a:{get_label(path.types[0])})-[r1:{get_rel(path.relations[0])}]-(m1:{get_label(path.types[1])})-[r2:{get_rel(path.relations[1])}]-(m2:{get_label(path.types[2])})-[r3:{get_rel(path.relations[2])}]-(b:{get_label(path.types[3])})
            WHERE {where_anchor} AND a <> m1 AND m1 <> m2 AND m2 <> b AND a <> b
            RETURN DISTINCT b.name AS answer"""
        else:
            return None, {}

        return cypher, params

    def _cypher_trial(
        self, path: SchemaPath, anchor_name: str, limit: int = 5
    ) -> List[str]:
        """LIMIT N でCypherを試行し、結果エンティティのサンプルを返す。空なら空リスト。"""
        cypher, params = self._build_cypher_for_path(path, anchor_name)
        if not cypher:
            return []
        trial_cypher = cypher.rstrip() + f" LIMIT {limit}"
        try:
            records = self.finder.graph.run(trial_cypher, **params).data()
            return [r["answer"] for r in records if r.get("answer")]
        except Exception:
            return []

    def _run_cypher_trials(
        self, paths: List[SchemaPath], anchor_name: str, limit: int = 5
    ) -> List[Tuple[SchemaPath, List[str]]]:
        """各パスでCypher trialを実行。(path, sample_entities)のリスト。空結果パスは除外。"""
        results = []
        for p in paths:
            samples = self._cypher_trial(p, anchor_name, limit=limit)
            if samples:
                results.append((p, samples))
        return results

    def _execute_cypher_for_path(
        self, graph, path: SchemaPath, anchor_name: str
    ) -> Set[str]:
        """パスに対してCypherを実行（方向を無視 - 双方向トラバーサル）"""
        cypher, params = self._build_cypher_for_path(path, anchor_name)
        if not cypher:
            return set()

        try:
            records = graph.run(cypher, **params).data()
            answers = {r["answer"] for r in records if r["answer"]}

            # Only do CVT resolution for Freebase-based KGs (WebQSP)
            if self.kg_type == "webqsp":
                cvt_mids = {a for a in answers if a.startswith("m.")}
                if cvt_mids:
                    named_answers = answers - cvt_mids
                    for mid in cvt_mids:
                        try:
                            resolve_cypher = """
                            MATCH (cvt {name: $mid})-[r]-(target)
                            WHERE target.name IS NOT NULL AND NOT target.name STARTS WITH 'm.'
                            RETURN DISTINCT target.name AS name
                            """
                            resolved = graph.run(resolve_cypher, mid=mid).data()
                            for rec in resolved:
                                if rec["name"]:
                                    named_answers.add(rec["name"])
                        except Exception:
                            pass
                    if named_answers:
                        answers = named_answers
            return answers
        except Exception as e:
            print(f"Cypher error: {e}")
            return set()

    def _cypher_relax_labels(
        self, graph, path: SchemaPath, params: dict, where_anchor: str
    ) -> Set[str]:
        """Execute Cypher with type labels removed (relation-only constraint).

        When strict label matching fails (e.g., entity is stored under a
        different type than expected), removing labels while keeping the
        relation constraint often recovers correct answers.
        """
        try:
            if len(path.types) == 2:
                rel = path.relations[0]
                if " " in rel or "-" in rel or "/" in rel or "." in rel:
                    rel = f"`{rel}`"
                cypher = f"""
                MATCH (a)-[r:{rel}]-(b)
                WHERE {where_anchor} AND a <> b
                RETURN DISTINCT b.name AS answer
                LIMIT 200
                """
            elif len(path.types) == 3:
                r0 = path.relations[0]
                r1 = path.relations[1]
                if " " in r0 or "-" in r0 or "/" in r0 or "." in r0:
                    r0 = f"`{r0}`"
                if " " in r1 or "-" in r1 or "/" in r1 or "." in r1:
                    r1 = f"`{r1}`"
                cypher = f"""
                MATCH (a)-[:{r0}]-(mid)-[:{r1}]-(b)
                WHERE {where_anchor} AND a <> mid AND mid <> b AND a <> b
                RETURN DISTINCT b.name AS answer
                LIMIT 200
                """
            else:
                return set()
            records = graph.run(cypher, **params).data()
            return {r["answer"] for r in records if r["answer"]}
        except Exception:
            return set()

    def _apply_kopl_operations(
        self, kopl_program: KoPLOperation, entity_sets: List[EntitySet]
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
