"""
PrimeKG v2 Schema Definition
改良版PrimeKGのスキーマ定義と質問テンプレート
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

# =============================================================================
# Node Types
# =============================================================================
NODE_TYPES = [
    "anatomy",
    "biological_process",
    "cellular_component",
    "disease",
    "drug",
    "effect/phenotype",
    "exposure",
    "gene/protein",
    "molecular_function",
    "pathway",
]

# =============================================================================
# Schema Graph (Source -> Relation -> Target)
# NOTE: 逆方向リレーション (targeted_by, carrier_for, etc.) は削除
# =============================================================================
SCHEMA_GRAPH: List[Tuple[str, str, str, int]] = [
    # anatomy
    ("anatomy", "absent gene", "gene/protein", 19887),
    ("anatomy", "expressed gene", "gene/protein", 1518203),
    ("anatomy", "parent-child", "anatomy", 28064),

    # biological_process
    ("biological_process", "parent-child", "biological_process", 105772),

    # cellular_component
    ("cellular_component", "parent-child", "cellular_component", 9690),

    # disease
    ("disease", "linked exposure", "exposure", 2304),
    ("disease", "parent-child", "disease", 64388),
    ("disease", "phenotype absent", "effect/phenotype", 1193),
    ("disease", "phenotype present", "effect/phenotype", 150317),

    # drug
    ("drug", "carrier", "gene/protein", 864),
    ("drug", "contraindication", "disease", 30675),
    ("drug", "enzyme", "gene/protein", 5317),
    ("drug", "indication", "disease", 9388),
    ("drug", "off-label use", "disease", 2568),
    ("drug", "side effect", "effect/phenotype", 64784),
    ("drug", "synergistic interaction", "drug", 2672628),
    ("drug", "target", "gene/protein", 16380),
    ("drug", "transporter", "gene/protein", 3092),

    # effect/phenotype
    ("effect/phenotype", "parent-child", "effect/phenotype", 37472),

    # exposure
    ("exposure", "interacts with", "biological_process", 1625),
    ("exposure", "interacts with", "cellular_component", 10),
    ("exposure", "interacts with", "gene/protein", 1212),
    ("exposure", "interacts with", "molecular_function", 45),
    ("exposure", "linked to", "disease", 2304),
    ("exposure", "parent-child", "exposure", 4140),

    # gene/protein
    ("gene/protein", "associated with", "disease", 80411),
    ("gene/protein", "expression absent", "anatomy", 19887),
    ("gene/protein", "expression present", "anatomy", 1518203),
    ("gene/protein", "interacts with", "biological_process", 144805),
    ("gene/protein", "interacts with", "cellular_component", 83402),
    ("gene/protein", "interacts with", "molecular_function", 69530),
    ("gene/protein", "interacts with", "pathway", 42646),
    ("gene/protein", "ppi", "gene/protein", 642150),

    # molecular_function
    ("molecular_function", "parent-child", "molecular_function", 27148),

    # pathway
    ("pathway", "parent-child", "pathway", 5070),
]

# =============================================================================
# 2-Hop Path Templates (意味のあるパスのみ)
# =============================================================================
TWO_HOP_TEMPLATES: List[Dict] = [
    # Drug -> Gene -> Disease パターン
    {
        "name": "drug_target_gene_disease",
        "path": ("drug", "target", "gene/protein", "associated with", "disease"),
        "question_templates": [
            "What diseases are associated with genes targeted by {anchor}?",
            "Which diseases involve genes that {anchor} targets?",
            "{anchor} targets certain genes. What diseases are these genes associated with?",
        ],
    },
    {
        "name": "drug_enzyme_gene_disease",
        "path": ("drug", "enzyme", "gene/protein", "associated with", "disease"),
        "question_templates": [
            "What diseases are linked to genes that metabolize {anchor}?",
            "Which diseases are associated with the enzymes that process {anchor}?",
        ],
    },

    # Disease -> Gene -> Drug パターン
    {
        "name": "disease_gene_drug_target",
        "path": ("disease", "associated with", "gene/protein", "target", "drug"),
        "question_templates": [
            "What drugs target genes associated with {anchor}?",
            "Which drugs target the genes linked to {anchor}?",
            "Find drugs that target genes implicated in {anchor}.",
        ],
    },

    # Gene -> Gene -> Disease パターン (PPI)
    {
        "name": "gene_ppi_gene_disease",
        "path": ("gene/protein", "ppi", "gene/protein", "associated with", "disease"),
        "question_templates": [
            "What diseases are associated with proteins that interact with {anchor}?",
            "Which diseases involve interaction partners of {anchor}?",
            "{anchor} interacts with other proteins. What diseases are these proteins linked to?",
        ],
    },

    # Drug -> Disease -> Phenotype パターン
    {
        "name": "drug_indication_disease_phenotype",
        "path": ("drug", "indication", "disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in diseases treated by {anchor}?",
            "Which symptoms or phenotypes characterize diseases that {anchor} is indicated for?",
        ],
    },

    # NOTE: gene_expression_anatomy_gene は除外（ファンアウトが大きすぎる）

    # Gene -> Biological Process -> Gene パターン
    {
        "name": "gene_process_gene",
        "path": ("gene/protein", "interacts with", "biological_process", "interacts with", "gene/protein"),
        "question_templates": [
            "What genes participate in the same biological processes as {anchor}?",
            "Which genes share biological process involvement with {anchor}?",
        ],
    },

    # NOTE: drug_sideeffect_drug は除外（ファンアウトが大きすぎる）

    # Exposure -> Disease -> Drug パターン
    {
        "name": "exposure_disease_drug",
        "path": ("exposure", "linked to", "disease", "indication", "drug"),
        "question_templates": [
            "What drugs treat diseases linked to {anchor} exposure?",
            "Which drugs are used for conditions associated with {anchor}?",
        ],
    },
]

# =============================================================================
# 1-Hop Path Templates
# =============================================================================
ONE_HOP_TEMPLATES: List[Dict] = [
    {
        "name": "drug_targets",
        "path": ("drug", "target", "gene/protein"),
        "question_templates": [
            "What genes does {anchor} target?",
            "Which proteins are targeted by {anchor}?",
            "What are the molecular targets of {anchor}?",
        ],
    },
    {
        "name": "drug_indications",
        "path": ("drug", "indication", "disease"),
        "question_templates": [
            "What diseases is {anchor} used to treat?",
            "What are the indications for {anchor}?",
            "Which conditions can be treated with {anchor}?",
        ],
    },
    {
        "name": "drug_side_effects",
        "path": ("drug", "side effect", "effect/phenotype"),
        "question_templates": [
            "What are the side effects of {anchor}?",
            "Which adverse effects are associated with {anchor}?",
        ],
    },
    {
        "name": "gene_diseases",
        "path": ("gene/protein", "associated with", "disease"),
        "question_templates": [
            "What diseases are associated with {anchor}?",
            "Which diseases involve {anchor}?",
            "What conditions is {anchor} implicated in?",
        ],
    },
    {
        "name": "gene_pathways",
        "path": ("gene/protein", "interacts with", "pathway"),
        "question_templates": [
            "What pathways does {anchor} participate in?",
            "Which biological pathways involve {anchor}?",
        ],
    },
    {
        "name": "gene_expression",
        "path": ("gene/protein", "expression present", "anatomy"),
        "question_templates": [
            "Where is {anchor} expressed?",
            "In which tissues or organs is {anchor} expressed?",
            "What are the expression sites of {anchor}?",
        ],
    },
    {
        "name": "disease_phenotypes",
        "path": ("disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What are the phenotypes of {anchor}?",
            "Which symptoms characterize {anchor}?",
            "What clinical features are present in {anchor}?",
        ],
    },
    {
        "name": "disease_genes",
        "path": ("disease", "associated with", "gene/protein"),
        "question_templates": [
            "What genes are associated with {anchor}?",
            "Which genes are implicated in {anchor}?",
        ],
    },
    {
        "name": "disease_drugs",
        "path": ("disease", "indication", "drug"),
        "question_templates": [
            "What drugs are used to treat {anchor}?",
            "Which medications are indicated for {anchor}?",
        ],
    },
    {
        "name": "gene_ppi",
        "path": ("gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins interact with {anchor}?",
            "Which proteins are interaction partners of {anchor}?",
            "What are the protein-protein interactions of {anchor}?",
        ],
    },
]

# =============================================================================
# Intersection Templates (2-anchor and 3-anchor)
# =============================================================================
TWO_ANCHOR_INTERSECTION_TEMPLATES: List[Dict] = [
    {
        "name": "drugs_common_target",
        "anchors": [("drug", "target"), ("drug", "target")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are targeted by both {anchor_a} and {anchor_b}?",
            "Which proteins are common targets of {anchor_a} and {anchor_b}?",
            "Find genes that both {anchor_a} and {anchor_b} target.",
        ],
    },
    {
        "name": "genes_common_disease",
        "anchors": [("gene/protein", "associated with"), ("gene/protein", "associated with")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are associated with both {anchor_a} and {anchor_b}?",
            "Which diseases involve both {anchor_a} and {anchor_b}?",
        ],
    },
    {
        "name": "drugs_common_indication",
        "anchors": [("drug", "indication"), ("drug", "indication")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are treated by both {anchor_a} and {anchor_b}?",
            "Which conditions can be treated with either {anchor_a} or {anchor_b}?",
        ],
    },
    {
        "name": "drugs_common_side_effect",
        "anchors": [("drug", "side effect"), ("drug", "side effect")],
        "intersection_type": "effect/phenotype",
        "question_templates": [
            "What side effects are shared by {anchor_a} and {anchor_b}?",
            "Which adverse effects occur with both {anchor_a} and {anchor_b}?",
        ],
    },
    {
        "name": "genes_common_pathway",
        "anchors": [("gene/protein", "interacts with"), ("gene/protein", "interacts with")],
        "intersection_type": "pathway",
        "question_templates": [
            "What pathways involve both {anchor_a} and {anchor_b}?",
            "Which biological pathways do {anchor_a} and {anchor_b} both participate in?",
        ],
    },
    {
        "name": "diseases_common_gene",
        "anchors": [("disease", "associated with"), ("disease", "associated with")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are associated with both {anchor_a} and {anchor_b}?",
            "Which genes are implicated in both {anchor_a} and {anchor_b}?",
        ],
    },
    {
        "name": "drug_disease_common_gene",
        "anchors": [("drug", "target"), ("disease", "associated with")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are both targeted by {anchor_a} and associated with {anchor_b}?",
            "Which genes does {anchor_a} target that are also linked to {anchor_b}?",
        ],
    },
]

THREE_ANCHOR_INTERSECTION_TEMPLATES: List[Dict] = [
    {
        "name": "three_drugs_common_target",
        "anchors": [("drug", "target"), ("drug", "target"), ("drug", "target")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are targeted by {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which proteins are common targets of all three drugs: {anchor_a}, {anchor_b}, and {anchor_c}?",
        ],
    },
    {
        "name": "three_genes_common_disease",
        "anchors": [("gene/protein", "associated with"), ("gene/protein", "associated with"), ("gene/protein", "associated with")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are associated with {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which diseases involve all three genes: {anchor_a}, {anchor_b}, and {anchor_c}?",
        ],
    },
    {
        "name": "three_genes_common_pathway",
        "anchors": [("gene/protein", "interacts with"), ("gene/protein", "interacts with"), ("gene/protein", "interacts with")],
        "intersection_type": "pathway",
        "question_templates": [
            "What pathways involve {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which biological pathways do all three proteins participate in?",
        ],
    },
]


def get_cypher_label(node_type: str) -> str:
    """ノードタイプをCypher用のラベルに変換"""
    if "/" in node_type:
        return f"`{node_type}`"
    return node_type


def print_schema_summary():
    """スキーマのサマリーを表示"""
    print("=" * 60)
    print("PrimeKG v2 Schema Summary")
    print("=" * 60)

    print(f"\nNode Types: {len(NODE_TYPES)}")
    for nt in NODE_TYPES:
        print(f"  - {nt}")

    print(f"\nRelationships: {len(SCHEMA_GRAPH)}")

    # リレーションタイプごとにグループ化
    rel_counts = {}
    for src, rel, tgt, cnt in SCHEMA_GRAPH:
        if rel not in rel_counts:
            rel_counts[rel] = 0
        rel_counts[rel] += cnt

    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  - {rel}: {cnt:,}")

    print(f"\nQuery Templates:")
    print(f"  - 1-hop: {len(ONE_HOP_TEMPLATES)}")
    print(f"  - 2-hop: {len(TWO_HOP_TEMPLATES)}")
    print(f"  - 2-anchor intersection: {len(TWO_ANCHOR_INTERSECTION_TEMPLATES)}")
    print(f"  - 3-anchor intersection: {len(THREE_ANCHOR_INTERSECTION_TEMPLATES)}")


if __name__ == "__main__":
    print_schema_summary()
