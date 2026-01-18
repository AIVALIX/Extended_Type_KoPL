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
# =============================================================================
SCHEMA_GRAPH: List[Tuple[str, str, str, int]] = [
    # anatomy
    ("anatomy", "absent_gene", "gene/protein", 19887),
    ("anatomy", "expressed_gene", "gene/protein", 1518203),
    ("anatomy", "parent_child", "anatomy", 28064),

    # biological_process
    ("biological_process", "interacted_by", "exposure", 1625),
    ("biological_process", "interacted_by", "gene/protein", 144805),
    ("biological_process", "parent_child", "biological_process", 105772),

    # cellular_component
    ("cellular_component", "interacted_by", "exposure", 10),
    ("cellular_component", "interacted_by", "gene/protein", 83402),
    ("cellular_component", "parent_child", "cellular_component", 9690),

    # disease
    ("disease", "associated_with", "gene/protein", 80411),
    ("disease", "contraindicated_for", "drug", 30675),
    ("disease", "linked_exposure", "exposure", 2304),
    ("disease", "off_label_drug", "drug", 2568),
    ("disease", "parent_child", "disease", 64388),
    ("disease", "phenotype_absent", "effect/phenotype", 1193),
    ("disease", "phenotype_present", "effect/phenotype", 150317),
    ("disease", "treated_by", "drug", 9388),

    # drug
    ("drug", "carrier", "gene/protein", 864),
    ("drug", "contraindication", "disease", 30675),
    ("drug", "enzyme", "gene/protein", 5317),
    ("drug", "indication", "disease", 9388),
    ("drug", "off_label_use", "disease", 2568),
    ("drug", "side_effect", "effect/phenotype", 64784),
    ("drug", "synergistic_interaction", "drug", 2672628),
    ("drug", "target", "gene/protein", 16380),
    ("drug", "transporter", "gene/protein", 3092),

    # effect/phenotype
    ("effect/phenotype", "caused_by_drug", "drug", 64784),
    ("effect/phenotype", "disease_with_phenotype", "disease", 150317),
    ("effect/phenotype", "disease_without_phenotype", "disease", 1193),
    ("effect/phenotype", "parent_child", "effect/phenotype", 37472),

    # exposure
    ("exposure", "interacts_with", "biological_process", 1625),
    ("exposure", "interacts_with", "cellular_component", 10),
    ("exposure", "interacts_with", "gene/protein", 1212),
    ("exposure", "interacts_with", "molecular_function", 45),
    ("exposure", "linked_to", "disease", 2304),
    ("exposure", "parent_child", "exposure", 4140),

    # gene/protein
    ("gene/protein", "associated_disease", "disease", 80411),
    ("gene/protein", "carrier_for", "drug", 864),
    ("gene/protein", "expression_absent", "anatomy", 19887),
    ("gene/protein", "expression_present", "anatomy", 1518203),
    ("gene/protein", "interacted_by", "exposure", 1212),
    ("gene/protein", "interacts_with", "biological_process", 144805),
    ("gene/protein", "interacts_with", "cellular_component", 83402),
    ("gene/protein", "interacts_with", "molecular_function", 69530),
    ("gene/protein", "interacts_with", "pathway", 42646),
    ("gene/protein", "metabolized_by", "drug", 5317),
    ("gene/protein", "ppi", "gene/protein", 642150),
    ("gene/protein", "targeted_by", "drug", 16380),
    ("gene/protein", "transported_by", "drug", 3092),

    # molecular_function
    ("molecular_function", "interacted_by", "exposure", 45),
    ("molecular_function", "interacted_by", "gene/protein", 69530),
    ("molecular_function", "parent_child", "molecular_function", 27148),

    # pathway
    ("pathway", "interacted_by", "gene/protein", 42646),
    ("pathway", "parent_child", "pathway", 5070),
]

# =============================================================================
# 2-Hop Path Templates (意味のあるパスのみ)
# =============================================================================
TWO_HOP_TEMPLATES: List[Dict] = [
    # Drug -> Gene -> Disease パターン
    {
        "name": "drug_target_gene_disease",
        "path": ("drug", "target", "gene/protein", "associated_disease", "disease"),
        "question_templates": [
            "What diseases are associated with genes targeted by {anchor}?",
            "Which diseases involve genes that {anchor} targets?",
            "{anchor} targets certain genes. What diseases are these genes associated with?",
        ],
    },
    {
        "name": "drug_enzyme_gene_disease",
        "path": ("drug", "enzyme", "gene/protein", "associated_disease", "disease"),
        "question_templates": [
            "What diseases are linked to genes that metabolize {anchor}?",
            "Which diseases are associated with the enzymes that process {anchor}?",
        ],
    },

    # Disease -> Gene -> Drug パターン
    {
        "name": "disease_gene_drug_target",
        "path": ("disease", "associated_with", "gene/protein", "targeted_by", "drug"),
        "question_templates": [
            "What drugs target genes associated with {anchor}?",
            "Which drugs target the genes linked to {anchor}?",
            "Find drugs that target genes implicated in {anchor}.",
        ],
    },

    # Gene -> Gene -> Disease パターン (PPI)
    {
        "name": "gene_ppi_gene_disease",
        "path": ("gene/protein", "ppi", "gene/protein", "associated_disease", "disease"),
        "question_templates": [
            "What diseases are associated with proteins that interact with {anchor}?",
            "Which diseases involve interaction partners of {anchor}?",
            "{anchor} interacts with other proteins. What diseases are these proteins linked to?",
        ],
    },

    # Drug -> Disease -> Phenotype パターン
    {
        "name": "drug_indication_disease_phenotype",
        "path": ("drug", "indication", "disease", "phenotype_present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in diseases treated by {anchor}?",
            "Which symptoms or phenotypes characterize diseases that {anchor} is indicated for?",
        ],
    },

    # Gene -> Anatomy -> Gene パターン
    {
        "name": "gene_expression_anatomy_gene",
        "path": ("gene/protein", "expression_present", "anatomy", "expressed_gene", "gene/protein"),
        "question_templates": [
            "What other genes are expressed in the same tissues as {anchor}?",
            "Which genes share expression sites with {anchor}?",
            "Find genes co-expressed with {anchor} in the same anatomical locations.",
        ],
    },

    # Gene -> Biological Process -> Gene パターン
    {
        "name": "gene_process_gene",
        "path": ("gene/protein", "interacts_with", "biological_process", "interacted_by", "gene/protein"),
        "question_templates": [
            "What genes participate in the same biological processes as {anchor}?",
            "Which genes share biological process involvement with {anchor}?",
        ],
    },

    # Drug -> Side Effect -> Drug パターン
    {
        "name": "drug_sideeffect_drug",
        "path": ("drug", "side_effect", "effect/phenotype", "caused_by_drug", "drug"),
        "question_templates": [
            "What other drugs share side effects with {anchor}?",
            "Which drugs cause similar side effects as {anchor}?",
        ],
    },

    # Exposure -> Disease -> Drug パターン
    {
        "name": "exposure_disease_drug",
        "path": ("exposure", "linked_to", "disease", "treated_by", "drug"),
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
        "path": ("drug", "side_effect", "effect/phenotype"),
        "question_templates": [
            "What are the side effects of {anchor}?",
            "Which adverse effects are associated with {anchor}?",
        ],
    },
    {
        "name": "gene_diseases",
        "path": ("gene/protein", "associated_disease", "disease"),
        "question_templates": [
            "What diseases are associated with {anchor}?",
            "Which diseases involve {anchor}?",
            "What conditions is {anchor} implicated in?",
        ],
    },
    {
        "name": "gene_pathways",
        "path": ("gene/protein", "interacts_with", "pathway"),
        "question_templates": [
            "What pathways does {anchor} participate in?",
            "Which biological pathways involve {anchor}?",
        ],
    },
    {
        "name": "gene_expression",
        "path": ("gene/protein", "expression_present", "anatomy"),
        "question_templates": [
            "Where is {anchor} expressed?",
            "In which tissues or organs is {anchor} expressed?",
            "What are the expression sites of {anchor}?",
        ],
    },
    {
        "name": "disease_phenotypes",
        "path": ("disease", "phenotype_present", "effect/phenotype"),
        "question_templates": [
            "What are the phenotypes of {anchor}?",
            "Which symptoms characterize {anchor}?",
            "What clinical features are present in {anchor}?",
        ],
    },
    {
        "name": "disease_genes",
        "path": ("disease", "associated_with", "gene/protein"),
        "question_templates": [
            "What genes are associated with {anchor}?",
            "Which genes are implicated in {anchor}?",
        ],
    },
    {
        "name": "disease_drugs",
        "path": ("disease", "treated_by", "drug"),
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
        "anchors": [("gene/protein", "associated_disease"), ("gene/protein", "associated_disease")],
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
        "anchors": [("drug", "side_effect"), ("drug", "side_effect")],
        "intersection_type": "effect/phenotype",
        "question_templates": [
            "What side effects are shared by {anchor_a} and {anchor_b}?",
            "Which adverse effects occur with both {anchor_a} and {anchor_b}?",
        ],
    },
    {
        "name": "genes_common_pathway",
        "anchors": [("gene/protein", "interacts_with"), ("gene/protein", "interacts_with")],
        "intersection_type": "pathway",
        "question_templates": [
            "What pathways involve both {anchor_a} and {anchor_b}?",
            "Which biological pathways do {anchor_a} and {anchor_b} both participate in?",
        ],
    },
    {
        "name": "diseases_common_gene",
        "anchors": [("disease", "associated_with"), ("disease", "associated_with")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are associated with both {anchor_a} and {anchor_b}?",
            "Which genes are implicated in both {anchor_a} and {anchor_b}?",
        ],
    },
    {
        "name": "drug_disease_common_gene",
        "anchors": [("drug", "target"), ("disease", "associated_with")],
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
        "anchors": [("gene/protein", "associated_disease"), ("gene/protein", "associated_disease"), ("gene/protein", "associated_disease")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are associated with {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which diseases involve all three genes: {anchor_a}, {anchor_b}, and {anchor_c}?",
        ],
    },
    {
        "name": "three_genes_common_pathway",
        "anchors": [("gene/protein", "interacts_with"), ("gene/protein", "interacts_with"), ("gene/protein", "interacts_with")],
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
