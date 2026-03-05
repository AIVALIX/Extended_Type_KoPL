"""
PrimeKG v3 Schema Definition
明確な意味を持つリレーションのみを使用
"""

from typing import List, Dict

# =============================================================================
# Node Types
# =============================================================================
NODE_TYPES = [
    "disease",
    "drug",
    "effect/phenotype",
    "exposure",
    "gene/protein",
]

# =============================================================================
# Relations with Clear Semantics (QAに適したリレーションのみ)
# =============================================================================
# Format: (source_type, relation, target_type, count)

CLEAR_RELATIONS: List[Dict] = [
    # Drug -> Gene/Protein (薬と標的の関係)
    {"src": "drug", "rel": "target", "tgt": "gene/protein", "meaning": "薬の標的タンパク質"},
    {"src": "drug", "rel": "carrier", "tgt": "gene/protein", "meaning": "薬のキャリアタンパク質"},
    {"src": "drug", "rel": "enzyme", "tgt": "gene/protein", "meaning": "薬を代謝する酵素"},
    {"src": "drug", "rel": "transporter", "tgt": "gene/protein", "meaning": "薬を輸送するタンパク質"},

    # Drug -> Disease (薬と疾患の関係)
    {"src": "drug", "rel": "indication", "tgt": "disease", "meaning": "承認適応症"},
    {"src": "drug", "rel": "off-label use", "tgt": "disease", "meaning": "適応外使用"},
    {"src": "drug", "rel": "contraindication", "tgt": "disease", "meaning": "禁忌"},

    # Drug -> Effect/Phenotype (薬と副作用)
    {"src": "drug", "rel": "side effect", "tgt": "effect/phenotype", "meaning": "副作用"},

    # Gene/Protein -> Gene/Protein (タンパク質相互作用)
    {"src": "gene/protein", "rel": "ppi", "tgt": "gene/protein", "meaning": "タンパク質間相互作用"},

    # Disease -> Effect/Phenotype (疾患と症状)
    {"src": "disease", "rel": "phenotype present", "tgt": "effect/phenotype", "meaning": "疾患で見られる症状"},
    {"src": "disease", "rel": "phenotype absent", "tgt": "effect/phenotype", "meaning": "疾患で見られない症状"},

    # Exposure -> Disease (環境因子と疾患)
    {"src": "exposure", "rel": "linked to", "tgt": "disease", "meaning": "環境因子と疾患の関連"},
]

# Schema Graph for pipeline compatibility
SCHEMA_GRAPH: List[tuple] = [
    (r["src"], r["rel"], r["tgt"]) for r in CLEAR_RELATIONS
]

# =============================================================================
# 1-Hop Templates
# =============================================================================
ONE_HOP_TEMPLATES: List[Dict] = [
    # Drug -> Gene/Protein
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
        "name": "drug_carriers",
        "path": ("drug", "carrier", "gene/protein"),
        "question_templates": [
            "What proteins carry {anchor}?",
            "Which carrier proteins transport {anchor}?",
        ],
    },
    {
        "name": "drug_enzymes",
        "path": ("drug", "enzyme", "gene/protein"),
        "question_templates": [
            "What enzymes metabolize {anchor}?",
            "Which proteins are involved in {anchor} metabolism?",
        ],
    },
    {
        "name": "drug_transporters",
        "path": ("drug", "transporter", "gene/protein"),
        "question_templates": [
            "What transporters are involved with {anchor}?",
            "Which transporter proteins interact with {anchor}?",
        ],
    },

    # Drug -> Disease
    {
        "name": "drug_indications",
        "path": ("drug", "indication", "disease"),
        "question_templates": [
            "What diseases is {anchor} indicated for?",
            "Which conditions can be treated with {anchor}?",
            "What are the approved indications for {anchor}?",
        ],
    },
    {
        "name": "drug_offlabel",
        "path": ("drug", "off-label use", "disease"),
        "question_templates": [
            "What diseases is {anchor} used off-label for?",
            "For which conditions is {anchor} used without official approval?",
        ],
    },
    {
        "name": "drug_contraindications",
        "path": ("drug", "contraindication", "disease"),
        "question_templates": [
            "What are the contraindications for {anchor}?",
            "In which conditions should {anchor} not be used?",
            "When is {anchor} contraindicated?",
        ],
    },

    # Drug -> Effect/Phenotype
    {
        "name": "drug_side_effects",
        "path": ("drug", "side effect", "effect/phenotype"),
        "question_templates": [
            "What are the side effects of {anchor}?",
            "Which adverse effects are associated with {anchor}?",
            "What adverse reactions can {anchor} cause?",
        ],
    },

    # Gene/Protein -> Gene/Protein
    {
        "name": "gene_ppi",
        "path": ("gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins interact with {anchor}?",
            "Which proteins are interaction partners of {anchor}?",
            "What are the protein-protein interactions of {anchor}?",
        ],
    },

    # Disease -> Effect/Phenotype
    {
        "name": "disease_phenotypes",
        "path": ("disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What are the phenotypes of {anchor}?",
            "Which symptoms characterize {anchor}?",
            "What clinical features are present in {anchor}?",
        ],
    },

    # Exposure -> Disease
    {
        "name": "exposure_diseases",
        "path": ("exposure", "linked to", "disease"),
        "question_templates": [
            "What diseases are linked to {anchor} exposure?",
            "Which conditions are associated with {anchor}?",
        ],
    },
]

# =============================================================================
# 2-Hop Templates
# =============================================================================
TWO_HOP_TEMPLATES: List[Dict] = [
    # Drug -> Gene -> Gene (via target -> ppi)
    {
        "name": "drug_target_ppi_gene",
        "path": ("drug", "target", "gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins interact with genes targeted by {anchor}?",
            "Which proteins are interaction partners of {anchor}'s targets?",
        ],
    },

    # NOTE: gene_ppi_gene_ppi_gene は除外（クエリが重すぎる: 640K x 640K）

    # Drug -> Disease -> Phenotype
    {
        "name": "drug_indication_disease_phenotype",
        "path": ("drug", "indication", "disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in diseases treated by {anchor}?",
            "Which symptoms characterize diseases that {anchor} is indicated for?",
        ],
    },

    # Drug -> Disease -> Drug (via contraindication -> indication)
    {
        "name": "drug_contraindication_disease_indication",
        "path": ("drug", "contraindication", "disease", "indication", "drug"),
        "question_templates": [
            "What drugs treat diseases for which {anchor} is contraindicated?",
            "Which medications are used for conditions where {anchor} cannot be used?",
        ],
    },

    # Exposure -> Disease -> Drug
    {
        "name": "exposure_disease_drug",
        "path": ("exposure", "linked to", "disease", "indication", "drug"),
        "question_templates": [
            "What drugs treat diseases linked to {anchor} exposure?",
            "Which medications are used for conditions associated with {anchor}?",
        ],
    },

    # Disease -> Phenotype -> Disease (common phenotype)
    # Note: This requires phenotype absent to make sense
    {
        "name": "disease_phenotype_disease",
        "path": ("disease", "phenotype present", "effect/phenotype", "phenotype present", "disease"),
        "question_templates": [
            "What diseases share phenotypes with {anchor}?",
            "Which conditions have similar symptoms to {anchor}?",
        ],
    },
]

# =============================================================================
# Intersection Templates (2-anchor)
# =============================================================================
TWO_ANCHOR_INTERSECTION_TEMPLATES: List[Dict] = [
    # Common targets
    {
        "name": "drugs_common_target",
        "anchors": [("drug", "target"), ("drug", "target")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are targeted by both {anchor_a} and {anchor_b}?",
            "Which proteins are common targets of {anchor_a} and {anchor_b}?",
        ],
    },
    # Common indications
    {
        "name": "drugs_common_indication",
        "anchors": [("drug", "indication"), ("drug", "indication")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are treated by both {anchor_a} and {anchor_b}?",
            "Which conditions can be treated with both {anchor_a} and {anchor_b}?",
        ],
    },
    # Common side effects
    {
        "name": "drugs_common_side_effect",
        "anchors": [("drug", "side effect"), ("drug", "side effect")],
        "intersection_type": "effect/phenotype",
        "question_templates": [
            "What side effects are shared by {anchor_a} and {anchor_b}?",
            "Which adverse effects occur with both {anchor_a} and {anchor_b}?",
        ],
    },
    # Common PPI partners
    {
        "name": "genes_common_ppi",
        "anchors": [("gene/protein", "ppi"), ("gene/protein", "ppi")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What proteins interact with both {anchor_a} and {anchor_b}?",
            "Which proteins are common interaction partners of {anchor_a} and {anchor_b}?",
        ],
    },
    # Drug target + Disease treatment
    {
        "name": "drug_disease_common_gene",
        "anchors": [("drug", "target"), ("drug", "target")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are targeted by both {anchor_a} and {anchor_b}?",
            "Which proteins do both {anchor_a} and {anchor_b} target?",
        ],
    },
    # Common contraindications
    {
        "name": "drugs_common_contraindication",
        "anchors": [("drug", "contraindication"), ("drug", "contraindication")],
        "intersection_type": "disease",
        "question_templates": [
            "What conditions are contraindications for both {anchor_a} and {anchor_b}?",
            "In which diseases should neither {anchor_a} nor {anchor_b} be used?",
        ],
    },
]

# =============================================================================
# Intersection Templates (3-anchor)
# =============================================================================
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
        "name": "three_drugs_common_side_effect",
        "anchors": [("drug", "side effect"), ("drug", "side effect"), ("drug", "side effect")],
        "intersection_type": "effect/phenotype",
        "question_templates": [
            "What side effects are shared by {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which adverse effects occur with all three drugs?",
        ],
    },
    {
        "name": "three_genes_common_ppi",
        "anchors": [("gene/protein", "ppi"), ("gene/protein", "ppi"), ("gene/protein", "ppi")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What proteins interact with {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which proteins are common interaction partners of all three?",
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
    print("PrimeKG v3 Schema Summary (Clear Relations Only)")
    print("=" * 60)

    print(f"\nNode Types: {len(NODE_TYPES)}")
    for nt in NODE_TYPES:
        print(f"  - {nt}")

    print(f"\nClear Relations: {len(CLEAR_RELATIONS)}")
    for r in CLEAR_RELATIONS:
        print(f"  - {r['src']} -[{r['rel']}]-> {r['tgt']}: {r['meaning']}")

    print(f"\nQuery Templates:")
    print(f"  - 1-hop: {len(ONE_HOP_TEMPLATES)}")
    print(f"  - 2-hop: {len(TWO_HOP_TEMPLATES)}")
    print(f"  - 2-anchor intersection: {len(TWO_ANCHOR_INTERSECTION_TEMPLATES)}")
    print(f"  - 3-anchor intersection: {len(THREE_ANCHOR_INTERSECTION_TEMPLATES)}")


if __name__ == "__main__":
    print_schema_summary()
