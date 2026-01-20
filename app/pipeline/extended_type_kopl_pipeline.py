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


@dataclass
class KoPLOperation:
    """KoPL操作"""
    op_type: OperationType
    relations: List[TypeRelation] = field(default_factory=list)
    children: List["KoPLOperation"] = field(default_factory=list)
    anchor_name: Optional[str] = None  # アンカーエンティティ名


@dataclass
class SchemaPath:
    """スキーマパス"""
    types: List[str]  # [src_type, ..., tgt_type]
    relations: List[str]  # [rel1, rel2, ...]
    source: str  # "global" or "stepwise"
    score: float = 0.0

    def to_text(self) -> str:
        parts = []
        for i, t in enumerate(self.types):
            parts.append(t)
            if i < len(self.relations):
                parts.append(f"-[{self.relations[i]}]->")
        return " ".join(parts)


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

class TypeRelationSchema(BaseModel):
    """Type間のリレーション（LLM出力用）"""
    src_type: str = Field(description="Source node type")
    tgt_type: str = Field(description="Target node type")
    intermediate_type: Optional[str] = Field(default=None, description="Intermediate type for 2-hop")


class KoPLOperationSchema(BaseModel):
    """KoPL操作（LLM出力用）"""
    operation: str = Field(description="Operation type: relate, intersection, union, exclude")
    relations: List[TypeRelationSchema] = Field(default=[], description="Type relations for relate operation")
    anchor_name: Optional[str] = Field(default=None, description="Anchor entity name if known")


class TypeKoPLProgramSchema(BaseModel):
    """Type-KoPLプログラム（LLM出力用）"""
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
        self.edges: List[Tuple[str, str, str]] = []  # (src, rel, tgt)
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.embeddings = None
        self.edge_embeddings: Dict[str, np.ndarray] = {}
        # APSP用
        self.apsp_dist: Dict[str, Dict[str, int]] = {}
        self.type_to_idx: Dict[str, int] = {}
        self.idx_to_type: Dict[int, str] = {}

    def add_edge(self, src_type: str, relation: str, tgt_type: str):
        self.edges.append((src_type, relation, tgt_type))
        self.types.add(src_type)
        self.types.add(tgt_type)
        # 有向グラフとして順方向のみ追加
        self.adjacency[src_type].append((tgt_type, relation))

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
        for src, rel, tgt in self.edges:
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
        queue = [(start_type, [start_type], [])]
        shortest_paths = []
        visited_at_depth = defaultdict(set)  # depth -> visited states

        while queue:
            current, type_path, rel_path = queue.pop(0)
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
                ))
                continue

            # まだゴールに到達していない場合、探索継続
            if current_depth < shortest_dist:
                for next_type, relation in self.adjacency[current]:
                    # サイクル防止（ただし自己参照は1回許可）
                    if next_type not in type_path:
                        state = (next_type, tuple(type_path + [next_type]), tuple(rel_path + [relation]))
                        if state not in visited_at_depth[current_depth + 1]:
                            visited_at_depth[current_depth + 1].add(state)
                            queue.append((next_type, type_path + [next_type], rel_path + [relation]))
                    elif next_type == current and type_path.count(next_type) < 2:
                        # 自己参照（PPI等）
                        state = (next_type, tuple(type_path + [next_type]), tuple(rel_path + [relation]))
                        if state not in visited_at_depth[current_depth + 1]:
                            visited_at_depth[current_depth + 1].add(state)
                            queue.append((next_type, type_path + [next_type], rel_path + [relation]))

        return shortest_paths

    def compute_embeddings(self, embeddings):
        """エッジの埋め込みを計算"""
        self.embeddings = embeddings
        texts = [f"{s} {r} {t}" for s, r, t in self.edges]
        if texts:
            vectors = embeddings.embed_documents(texts)
            for i, (s, r, t) in enumerate(self.edges):
                key = f"{s}|{r}|{t}"
                self.edge_embeddings[key] = np.array(vectors[i])

    def global_bfs(self, start_type: str, end_type: str, max_depth: int = 3) -> List[SchemaPath]:
        """Global BFS: 全パスを探索（KGT方式）"""
        if start_type not in self.types or end_type not in self.types:
            return []

        # BFSで全パスを探索
        queue = [(start_type, [start_type], [])]
        all_paths = []
        visited_states = set()

        while queue:
            current, type_path, rel_path = queue.pop(0)

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
                ))
                continue  # ゴールに到達したら探索を続けない

            for next_type, relation in self.adjacency[current]:
                # サイクル防止（ただし自己参照リレーション(PPI等)は1回だけ許可）
                if next_type not in type_path:
                    queue.append((
                        next_type,
                        type_path + [next_type],
                        rel_path + [relation],
                    ))
                elif next_type == current and type_path.count(next_type) < 2:
                    # 自己参照（例: gene/protein -> gene/protein via ppi）を許可
                    # ただし同じタイプが2回以上出現するのは防止
                    queue.append((
                        next_type,
                        type_path + [next_type],
                        rel_path + [relation],
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
                combined_paths.append(SchemaPath(
                    types=combined_types,
                    relations=combined_rels,
                    source="stepwise",
                ))

        return combined_paths


def build_schema_graph() -> SchemaGraph:
    """スキーマグラフを構築"""
    from dataset_construction.schema_v2 import SCHEMA_GRAPH

    schema = SchemaGraph()
    for src, rel, tgt, _ in SCHEMA_GRAPH:
        schema.add_edge(src, rel, tgt)

    # APSPを計算
    schema.compute_apsp()

    return schema


# =============================================================================
# Extended Type-KoPL Pipeline
# =============================================================================

class ExtendedTypeKoPLPipeline:
    """Extended Type-KoPL パイプライン"""

    def __init__(
        self,
        model: str = BASEMODEL,
        embedding_model: str = "text-embedding-3-small",
        top_k_paths: int = 3,
        max_depth: int = 3,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings

        self.llm = init_chat_model(model, model_provider="openai", temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.schema = build_schema_graph()
        self.schema.compute_embeddings(self.embeddings)

        self.finder = GraphPathFinder()

        self.top_k_paths = top_k_paths
        self.max_depth = max_depth
        self.max_candidate_paths = 50  # 候補パスの最大数を制限

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
                # "gene/protein" のような複合ラベルを優先
                labels = records[0]["labels"]
                for lbl in labels:
                    if lbl.lower() in ["drug", "disease", "gene/protein", "exposure", "biological_process", "molecular_function", "cellular_component", "pathway", "anatomy", "effect/phenotype"]:
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
        if kopl_program:
            log.append(f"  Generated KoPL program with {len(kopl_program.relations)} relations")
            if kopl_program.children:
                log.append(f"  Final operation: {kopl_program.op_type}")
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

        # Phase 3: ベクトル剪定
        log.append("Phase 3: Vector-based Pruning")
        selected_paths = self._vector_pruning(question, candidate_paths)
        log.append(f"  Selected {len(selected_paths)} paths")
        for p in selected_paths:
            log.append(f"    {p.to_text()} (score: {p.score:.4f})")

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

    def _generate_type_kopl(
        self,
        question: str,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None
    ) -> Optional[KoPLOperation]:
        """Phase 1: LLMでType-KoPLプログラムを生成"""

        type_list = ", ".join(sorted(self.schema.types))

        entity_info = ""
        if entity_name:
            entity_info = f"Known entity: {entity_name}"
            if entity_type:
                entity_info += f" (type: {entity_type})"

        prompt = f"""Convert this question into a Type-KoPL program.

Question: {question}
{entity_info}

Available node types: {type_list}

CRITICAL RULES:
1. src_type MUST be the type of the anchor entity (the known entity in the question)
2. tgt_type MUST be the type of what the question asks for
3. For 2-hop chains, use intermediate_type to specify the middle node type
4. For intersection queries (BOTH conditions), use multiple operations

DIRECTION IS CRITICAL:
- If anchor is a DRUG and question asks for DISEASES: src_type=drug, tgt_type=disease
- If anchor is a DISEASE and question asks for DRUGS: src_type=disease, tgt_type=drug
- The path always starts from the anchor (src_type) and ends at the answer (tgt_type)

Examples:

Question: "What diseases are associated with BRCA1?" (BRCA1 is a gene)
→ Anchor type: gene/protein, Target: disease
operations: [{{operation: "relate", relations: [{{src_type: "gene/protein", tgt_type: "disease"}}], anchor_name: "BRCA1"}}]

Question: "Which diseases are linked to genes targeted by Tacrolimus?" (Tacrolimus is a drug)
→ Anchor type: drug, Target: disease, Via: gene/protein
operations: [{{operation: "relate", relations: [{{src_type: "drug", tgt_type: "disease", intermediate_type: "gene/protein"}}], anchor_name: "Tacrolimus"}}]

Question: "Which drugs target genes linked to heart failure?" (heart failure is a disease)
→ Anchor type: disease, Target: drug, Via: gene/protein
operations: [{{operation: "relate", relations: [{{src_type: "disease", tgt_type: "drug", intermediate_type: "gene/protein"}}], anchor_name: "heart failure"}}]

Question: "Which genes are targeted by both DrugA and DrugB?"
→ INTERSECTION query with two anchors:
operations: [
  {{operation: "relate", relations: [{{src_type: "drug", tgt_type: "gene/protein"}}], anchor_name: "DrugA"}},
  {{operation: "relate", relations: [{{src_type: "drug", tgt_type: "gene/protein"}}], anchor_name: "DrugB"}}
]
final_operation: "intersection"

Return a JSON object with operations and final_operation."""

        llm_with_output = self.llm.with_structured_output(TypeKoPLProgramSchema)

        try:
            result = llm_with_output.invoke(prompt)

            # 有効なタイプのセット
            valid_types = self.schema.types

            # KoPLOperationに変換
            operations = []
            for op_schema in result.operations:
                relations = []
                for r in op_schema.relations:
                    # タイプ名をクリーンアップ
                    src = r.src_type.strip().rstrip("}],")
                    tgt = r.tgt_type.strip().rstrip("}],")
                    inter = r.intermediate_type.strip().rstrip("}],") if r.intermediate_type else None

                    # 有効なタイプかチェック
                    if src in valid_types and tgt in valid_types:
                        relations.append(TypeRelation(
                            src_type=src,
                            tgt_type=tgt,
                            intermediate_type=inter if inter in valid_types else None,
                        ))

                if relations:
                    operations.append(KoPLOperation(
                        op_type=OperationType(op_schema.operation),
                        relations=relations,
                        anchor_name=op_schema.anchor_name,
                    ))

            # 最終操作を構築
            if len(operations) == 1:
                return operations[0]
            elif len(operations) > 1:
                # フォールバック: 2つの操作でチェーン構造を形成している場合、マージする
                # 例: [drug->gene, gene->disease] => drug->disease via gene
                if len(operations) == 2:
                    op1, op2 = operations[0], operations[1]
                    # op1にanchorがあり、op2にanchorがない場合、チェーンの可能性
                    if op1.anchor_name and not op2.anchor_name:
                        if op1.relations and op2.relations:
                            r1 = op1.relations[0]
                            r2 = op2.relations[0]
                            # op1のtgtとop2のsrcが一致する場合、チェーンとしてマージ
                            if r1.tgt_type == r2.src_type:
                                merged_rel = TypeRelation(
                                    src_type=r1.src_type,
                                    tgt_type=r2.tgt_type,
                                    intermediate_type=r1.tgt_type,  # 中間タイプ
                                )
                                return KoPLOperation(
                                    op_type=OperationType.RELATE,
                                    relations=[merged_rel],
                                    anchor_name=op1.anchor_name,
                                )

                final_op_type = OperationType(result.final_operation)
                return KoPLOperation(
                    op_type=final_op_type,
                    children=operations,
                )
            else:
                raise ValueError("No valid operations generated")

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
        """

        all_paths = []

        # 子操作がある場合は各子操作に対して探索
        operations = kopl_program.children if kopl_program.children else [kopl_program]

        for op in operations:
            for rel in op.relations:
                if not rel.src_type or not rel.tgt_type:
                    continue

                if rel.intermediate_type:
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
        candidate_paths: List[SchemaPath]
    ) -> List[SchemaPath]:
        """Phase 3: ベクトル類似度でTop-K選択"""

        if not candidate_paths:
            return []

        # 質問をベクトル化
        question_vec = np.array(self.embeddings.embed_query(question))

        # パステキストをバッチでベクトル化（効率化）
        path_texts = [path.to_text() for path in candidate_paths]
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
        return candidate_paths[:self.top_k_paths]

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
        """パスに対してCypherを実行"""

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        if len(path.types) < 2:
            return set()

        # パスの長さに応じてCypherを構築（有向マッチング）
        if len(path.types) == 2:
            # 1-hop
            cypher = f"""
            MATCH (a:{get_label(path.types[0])})-[r:{path.relations[0]}]->(b:{get_label(path.types[1])})
            WHERE a.name = $anchor_name
            RETURN DISTINCT b.name AS answer
            LIMIT 100
            """
        elif len(path.types) == 3:
            # 2-hop
            cypher = f"""
            MATCH (a:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(mid:{get_label(path.types[1])})-[r2:{path.relations[1]}]->(b:{get_label(path.types[2])})
            WHERE a.name = $anchor_name AND a <> mid AND mid <> b AND a <> b
            RETURN DISTINCT b.name AS answer
            LIMIT 100
            """
        else:
            # 3-hop以上は複雑になるのでスキップ
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
