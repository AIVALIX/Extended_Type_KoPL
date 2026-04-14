"""
Knowledge Graph Configuration

PrimeKGQAとMetaQAの切り替えを管理
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class KGType(str, Enum):
    """Knowledge Graphの種類"""
    PRIMEKGQA = "primekgqa"
    METAQA = "metaqa"
    PCQA = "pcqa"
    WEBQSP = "webqsp"
    KQAPRO = "kqapro"


@dataclass
class Neo4jConfig:
    """Neo4j接続設定"""
    uri: str
    user: str
    password: str

    @classmethod
    def from_env(cls, kg_type: KGType) -> "Neo4jConfig":
        """環境に応じた設定を返す"""
        import os

        if kg_type == KGType.PRIMEKGQA:
            return cls(
                uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password"),
            )
        elif kg_type == KGType.METAQA:
            return cls(
                uri=os.getenv("NEO4J_METAQA_URI", "bolt://neo4j_metaqa:7687"),
                user=os.getenv("NEO4J_METAQA_USER", "neo4j"),
                password=os.getenv("NEO4J_METAQA_PASSWORD", "password"),
            )
        elif kg_type == KGType.PCQA:
            return cls(
                uri=os.getenv("NEO4J_PCQA_URI", "bolt://neo4j_pcqa:7687"),
                user=os.getenv("NEO4J_PCQA_USER", "neo4j"),
                password=os.getenv("NEO4J_PCQA_PASSWORD", "password"),
            )
        elif kg_type == KGType.WEBQSP:
            return cls(
                uri=os.getenv("NEO4J_WEBQSP_URI", "bolt://neo4j_webqsp:7687"),
                user=os.getenv("NEO4J_WEBQSP_USER", "neo4j"),
                password=os.getenv("NEO4J_WEBQSP_PASSWORD", "password"),
            )
        else:
            raise ValueError(f"Unknown KG type: {kg_type}")


@dataclass
class DatasetConfig:
    """データセット設定"""
    name: str
    path: Path
    entity_key: str  # サンプル中のエンティティ名のキー
    answer_key: str  # 回答ノードのキー
    relation_keys: List[str]  # リレーションのキー（1-hop: ["relation"], 2-hop: ["rel1", "rel2"]）
    query_type: str  # one_hop, two_hop, two_intersection, three_intersection


@dataclass
class KGConfig:
    """Knowledge Graph全体の設定"""
    kg_type: KGType
    neo4j: Neo4jConfig
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    schema_types: List[str] = field(default_factory=list)

    @classmethod
    def primekgqa(cls) -> "KGConfig":
        """PrimeKGQAの設定"""
        return cls(
            kg_type=KGType.PRIMEKGQA,
            neo4j=Neo4jConfig.from_env(KGType.PRIMEKGQA),
            datasets={
                "one_hop": DatasetConfig(
                    name="one_hop",
                    path=Path("result/dataset_v2/one_hop.jsonl"),
                    entity_key="anchor_name",
                    answer_key="answer_nodes",
                    relation_keys=["relation"],
                    query_type="one_hop",
                ),
                "two_hop": DatasetConfig(
                    name="two_hop",
                    path=Path("result/dataset_v2/two_hop.jsonl"),
                    entity_key="anchor_name",
                    answer_key="answer_nodes",
                    relation_keys=["rel1", "rel2"],
                    query_type="two_hop",
                ),
                "two_intersection": DatasetConfig(
                    name="two_intersection",
                    path=Path("result/dataset_v2/two_intersection.jsonl"),
                    entity_key="anchor_a_name",
                    answer_key="answer_nodes",
                    relation_keys=["anchor_a_rel", "anchor_b_rel"],
                    query_type="two_intersection",
                ),
                "three_intersection": DatasetConfig(
                    name="three_intersection",
                    path=Path("result/dataset_v2/three_intersection.jsonl"),
                    entity_key="anchor_a_name",
                    answer_key="answer_nodes",
                    relation_keys=["anchor_a_rel", "anchor_b_rel", "anchor_c_rel"],
                    query_type="three_intersection",
                ),
            },
            schema_types=[
                "anatomy", "biological_process", "cellular_component",
                "disease", "drug", "effect/phenotype", "exposure",
                "gene/protein", "molecular_function", "pathway",
            ],
        )

    @classmethod
    def metaqa(cls) -> "KGConfig":
        """MetaQAの設定"""
        return cls(
            kg_type=KGType.METAQA,
            neo4j=Neo4jConfig.from_env(KGType.METAQA),
            datasets={
                "1hop": DatasetConfig(
                    name="1hop",
                    path=Path("result/metaqa/1hop.jsonl"),
                    entity_key="entity",
                    answer_key="answers",
                    relation_keys=["relation"],
                    query_type="one_hop",
                ),
                "2hop": DatasetConfig(
                    name="2hop",
                    path=Path("result/metaqa/2hop.jsonl"),
                    entity_key="entity",
                    answer_key="answers",
                    relation_keys=["relation1", "relation2"],
                    query_type="two_hop",
                ),
                "3hop": DatasetConfig(
                    name="3hop",
                    path=Path("result/metaqa/3hop.jsonl"),
                    entity_key="entity",
                    answer_key="answers",
                    relation_keys=["relation1", "relation2", "relation3"],
                    query_type="three_hop",
                ),
            },
            schema_types=[
                "movie", "actor", "director", "writer", "genre", "year", "language", "tag",
            ],
        )

    @classmethod
    def pcqa(cls) -> "KGConfig":
        """PcQA (Pan-cancer QA)の設定"""
        return cls(
            kg_type=KGType.PCQA,
            neo4j=Neo4jConfig.from_env(KGType.PCQA),
            datasets={
                "all": DatasetConfig(
                    name="all",
                    path=Path("data/pcqa/qa/all.jsonl"),
                    entity_key="entity",
                    answer_key="answers",
                    relation_keys=["relation"],
                    query_type="mixed",  # 1-hop queries
                ),
            },
            schema_types=[
                "Cancer", "CancerCell", "Drug",
                "Genesymbol", "GeneticDisease", "SnvFull", "Fusion", "ClinicalTrial",
            ],
        )

    @classmethod
    def webqsp(cls) -> "KGConfig":
        """WebQSP (Freebase subset)の設定"""
        return cls(
            kg_type=KGType.WEBQSP,
            neo4j=Neo4jConfig.from_env(KGType.WEBQSP),
            datasets={
                "test": DatasetConfig(
                    name="test",
                    path=Path("data/webqsp/qa/test.jsonl"),
                    entity_key="entity",
                    answer_key="answer_nodes",
                    relation_keys=["relation"],
                    query_type="mixed",
                ),
            },
            schema_types=[
                "Entity", "CVT", "Film", "Person", "City_Town_Village",
                "Book", "Musical_Recording", "TV_Episode", "Musical_Album",
                "Organization", "Country", "Author", "Location",
                "Deceased_Person", "Musical_Artist", "Politician",
                "Human_Language", "TV_Program", "American_football_player",
                "College_University",
            ],
        )

    @classmethod
    def get(cls, kg_type: str) -> "KGConfig":
        """KGタイプから設定を取得"""
        if kg_type == "primekgqa":
            return cls.primekgqa()
        elif kg_type == "metaqa":
            return cls.metaqa()
        elif kg_type == "pcqa":
            return cls.pcqa()
        elif kg_type == "webqsp":
            return cls.webqsp()
        else:
            raise ValueError(f"Unknown KG type: {kg_type}. Available: primekgqa, metaqa, pcqa, webqsp")


# デフォルト設定（後方互換性のため）
DEFAULT_KG = KGType.PRIMEKGQA


def get_default_config() -> KGConfig:
    """デフォルトのKG設定を取得"""
    return KGConfig.get(DEFAULT_KG.value)
