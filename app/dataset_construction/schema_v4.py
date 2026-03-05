"""
PrimeKG v4 Schema Definition
v3からの改善:
1. 質問テンプレートの改善（スキーマ名との対応を明確化）
2. two_hopパターンの拡充（より多様なパスを追加）
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
# 1-Hop Templates (改善版: スキーマ名との対応を明確化)
# =============================================================================
ONE_HOP_TEMPLATES: List[Dict] = [
    # Drug -> Gene/Protein
    {
        "name": "drug_targets",
        "path": ("drug", "target", "gene/protein"),
        "question_templates": [
            "What genes does {anchor} target?",
            "Which proteins are targets of {anchor}?",
            "What are the target proteins of {anchor}?",
            "Which gene/protein targets does {anchor} have?",
        ],
    },
    {
        "name": "drug_carriers",
        "path": ("drug", "carrier", "gene/protein"),
        "question_templates": [
            "What are the carrier proteins of {anchor}?",
            "Which proteins serve as carriers for {anchor}?",
            "What carrier proteins does {anchor} use?",
        ],
    },
    {
        "name": "drug_enzymes",
        "path": ("drug", "enzyme", "gene/protein"),
        "question_templates": [
            "What enzymes metabolize {anchor}?",
            "Which enzyme proteins process {anchor}?",
            "What are the enzyme targets of {anchor}?",
        ],
    },
    {
        "name": "drug_transporters",
        "path": ("drug", "transporter", "gene/protein"),
        "question_templates": [
            "What are the transporter proteins for {anchor}?",
            "Which transporter proteins handle {anchor}?",
            "What transporter targets does {anchor} have?",
        ],
    },

    # Drug -> Disease
    {
        "name": "drug_indications",
        "path": ("drug", "indication", "disease"),
        "question_templates": [
            "What diseases is {anchor} indicated for?",
            "What are the indications for {anchor}?",
            "Which diseases have {anchor} as an indicated treatment?",
            "For what conditions is {anchor} an indicated drug?",
        ],
    },
    {
        "name": "drug_offlabel",
        "path": ("drug", "off-label use", "disease"),
        "question_templates": [
            "What diseases is {anchor} used off-label for?",
            "What are the off-label uses of {anchor}?",
            "For which conditions is {anchor} used off-label?",
        ],
    },
    {
        "name": "drug_contraindications",
        "path": ("drug", "contraindication", "disease"),
        "question_templates": [
            "What are the contraindications for {anchor}?",
            "For which diseases is {anchor} contraindicated?",
            "What conditions are contraindications of {anchor}?",
            "When is {anchor} a contraindicated drug?",
        ],
    },

    # Drug -> Effect/Phenotype
    {
        "name": "drug_side_effects",
        "path": ("drug", "side effect", "effect/phenotype"),
        "question_templates": [
            "What are the side effects of {anchor}?",
            "Which side effect phenotypes are caused by {anchor}?",
            "What adverse side effects does {anchor} have?",
        ],
    },

    # Gene/Protein -> Gene/Protein (改善: "ppi"を明示)
    {
        "name": "gene_ppi",
        "path": ("gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins have ppi relationships with {anchor}?",
            "Which proteins are ppi partners of {anchor}?",
            "What are the protein-protein interaction (ppi) partners of {anchor}?",
            "Which genes have ppi connections with {anchor}?",
        ],
    },

    # Disease -> Effect/Phenotype
    {
        "name": "disease_phenotypes",
        "path": ("disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in {anchor}?",
            "Which phenotype symptoms characterize {anchor}?",
            "What are the present phenotypes of {anchor}?",
        ],
    },

    # Exposure -> Disease
    {
        "name": "exposure_diseases",
        "path": ("exposure", "linked to", "disease"),
        "question_templates": [
            "What diseases are linked to {anchor}?",
            "Which conditions have a linked relationship with {anchor}?",
            "What diseases is {anchor} linked to?",
        ],
    },
]

# =============================================================================
# 2-Hop Templates (拡充版: より多様なパスを追加)
# =============================================================================
TWO_HOP_TEMPLATES: List[Dict] = [
    # =========================================================================
    # Drug -> Gene -> Gene パターン
    # =========================================================================
    # Drug -[target]-> Gene -[ppi]-> Gene
    {
        "name": "drug_target_ppi",
        "path": ("drug", "target", "gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins have ppi with the targets of {anchor}?",
            "Which genes are ppi partners of {anchor}'s target proteins?",
            "What are the ppi connections of proteins targeted by {anchor}?",
        ],
    },
    # Drug -[enzyme]-> Gene -[ppi]-> Gene (新規追加)
    {
        "name": "drug_enzyme_ppi",
        "path": ("drug", "enzyme", "gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins have ppi with enzymes that metabolize {anchor}?",
            "Which genes are ppi partners of {anchor}'s enzyme proteins?",
        ],
    },
    # Drug -[carrier]-> Gene -[ppi]-> Gene (新規追加)
    {
        "name": "drug_carrier_ppi",
        "path": ("drug", "carrier", "gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins have ppi with carrier proteins of {anchor}?",
            "Which genes are ppi partners of {anchor}'s carrier proteins?",
        ],
    },
    # Drug -[transporter]-> Gene -[ppi]-> Gene (新規追加)
    {
        "name": "drug_transporter_ppi",
        "path": ("drug", "transporter", "gene/protein", "ppi", "gene/protein"),
        "question_templates": [
            "What proteins have ppi with transporters of {anchor}?",
            "Which genes are ppi partners of {anchor}'s transporter proteins?",
        ],
    },

    # =========================================================================
    # Drug -> Disease -> Phenotype パターン
    # =========================================================================
    # Drug -[indication]-> Disease -[phenotype present]-> Phenotype
    {
        "name": "drug_indication_phenotype",
        "path": ("drug", "indication", "disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in diseases indicated for {anchor}?",
            "Which symptoms characterize diseases that have {anchor} as indication?",
        ],
    },
    # Drug -[contraindication]-> Disease -[phenotype present]-> Phenotype (新規追加)
    {
        "name": "drug_contraindication_phenotype",
        "path": ("drug", "contraindication", "disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in diseases where {anchor} is contraindicated?",
            "Which symptoms characterize diseases that have {anchor} as contraindication?",
        ],
    },
    # Drug -[off-label use]-> Disease -[phenotype present]-> Phenotype (新規追加)
    {
        "name": "drug_offlabel_phenotype",
        "path": ("drug", "off-label use", "disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in diseases with off-label use of {anchor}?",
            "Which symptoms characterize conditions where {anchor} is used off-label?",
        ],
    },

    # =========================================================================
    # Drug -> Disease -> Drug パターン
    # =========================================================================
    # Drug -[contraindication]-> Disease -[indication]-> Drug
    {
        "name": "drug_contraindication_indication",
        "path": ("drug", "contraindication", "disease", "indication", "drug"),
        "question_templates": [
            "What drugs have indication for diseases where {anchor} is contraindicated?",
            "Which medications are indicated for conditions that are contraindications of {anchor}?",
        ],
    },
    # Drug -[indication]-> Disease -[contraindication]-> Drug (新規追加: 逆方向)
    {
        "name": "drug_indication_contraindication",
        "path": ("drug", "indication", "disease", "contraindication", "drug"),
        "question_templates": [
            "What drugs are contraindicated for diseases indicated for {anchor}?",
            "Which medications have contraindication for conditions that {anchor} treats?",
        ],
    },
    # Drug -[indication]-> Disease -[indication]-> Drug (新規追加)
    {
        "name": "drug_indication_indication",
        "path": ("drug", "indication", "disease", "indication", "drug"),
        "question_templates": [
            "What other drugs share indications with {anchor}?",
            "Which drugs are also indicated for diseases that {anchor} treats?",
        ],
    },
    # Drug -[indication]-> Disease -[off-label use]-> Drug (新規追加)
    {
        "name": "drug_indication_offlabel",
        "path": ("drug", "indication", "disease", "off-label use", "drug"),
        "question_templates": [
            "What drugs are used off-label for diseases indicated for {anchor}?",
            "Which medications have off-label use for conditions that {anchor} is indicated for?",
        ],
    },

    # =========================================================================
    # Exposure -> Disease -> Drug/Phenotype パターン
    # =========================================================================
    # Exposure -[linked to]-> Disease -[indication]-> Drug
    {
        "name": "exposure_disease_indication",
        "path": ("exposure", "linked to", "disease", "indication", "drug"),
        "question_templates": [
            "What drugs are indicated for diseases linked to {anchor}?",
            "Which medications treat conditions that are linked to {anchor} exposure?",
        ],
    },
    # Exposure -[linked to]-> Disease -[phenotype present]-> Phenotype (新規追加)
    {
        "name": "exposure_disease_phenotype",
        "path": ("exposure", "linked to", "disease", "phenotype present", "effect/phenotype"),
        "question_templates": [
            "What phenotypes are present in diseases linked to {anchor}?",
            "Which symptoms characterize conditions linked to {anchor} exposure?",
        ],
    },

    # =========================================================================
    # Disease -> Phenotype -> Disease パターン
    # =========================================================================
    # Disease -[phenotype present]-> Phenotype -[phenotype present]-> Disease
    {
        "name": "disease_phenotype_disease",
        "path": ("disease", "phenotype present", "effect/phenotype", "phenotype present", "disease"),
        "question_templates": [
            "What diseases share present phenotypes with {anchor}?",
            "Which conditions have the same phenotype symptoms as {anchor}?",
        ],
    },

    # =========================================================================
    # Gene -> Gene -> Drug パターン (新規追加: 逆引き)
    # =========================================================================
    # Gene -[ppi]-> Gene -[target]-> Drug
    {
        "name": "gene_ppi_target_drug",
        "path": ("gene/protein", "ppi", "gene/protein", "target", "drug"),
        "question_templates": [
            "What drugs target proteins that have ppi with {anchor}?",
            "Which medications have target relationships with ppi partners of {anchor}?",
        ],
    },
]

# =============================================================================
# Intersection Templates (2-anchor) - v3と同じ
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
            "What diseases have both {anchor_a} and {anchor_b} as indicated treatments?",
            "Which conditions are indications for both {anchor_a} and {anchor_b}?",
        ],
    },
    # Common side effects
    {
        "name": "drugs_common_side_effect",
        "anchors": [("drug", "side effect"), ("drug", "side effect")],
        "intersection_type": "effect/phenotype",
        "question_templates": [
            "What side effects are shared by {anchor_a} and {anchor_b}?",
            "Which side effect phenotypes occur with both {anchor_a} and {anchor_b}?",
        ],
    },
    # Common PPI partners
    {
        "name": "genes_common_ppi",
        "anchors": [("gene/protein", "ppi"), ("gene/protein", "ppi")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What proteins have ppi with both {anchor_a} and {anchor_b}?",
            "Which genes are ppi partners of both {anchor_a} and {anchor_b}?",
        ],
    },
    # Common contraindications
    {
        "name": "drugs_common_contraindication",
        "anchors": [("drug", "contraindication"), ("drug", "contraindication")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are contraindications for both {anchor_a} and {anchor_b}?",
            "Which conditions should neither {anchor_a} nor {anchor_b} be used for?",
        ],
    },
    # Common enzymes (新規追加)
    {
        "name": "drugs_common_enzyme",
        "anchors": [("drug", "enzyme"), ("drug", "enzyme")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What enzymes metabolize both {anchor_a} and {anchor_b}?",
            "Which enzyme proteins process both {anchor_a} and {anchor_b}?",
        ],
    },
]

# =============================================================================
# Intersection Templates (3-anchor) - v3と同じ
# =============================================================================
THREE_ANCHOR_INTERSECTION_TEMPLATES: List[Dict] = [
    {
        "name": "three_drugs_common_target",
        "anchors": [("drug", "target"), ("drug", "target"), ("drug", "target")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What genes are targeted by {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which proteins are common targets of all three: {anchor_a}, {anchor_b}, and {anchor_c}?",
        ],
    },
    # NOTE: side_effect と ppi はクエリが重すぎてメモリエラーになるため除外
    # {
    #     "name": "three_drugs_common_side_effect",
    #     ...
    # },
    # {
    #     "name": "three_genes_common_ppi",
    #     ...
    # },
    # 軽量なテンプレートを追加
    {
        "name": "three_drugs_common_indication",
        "anchors": [("drug", "indication"), ("drug", "indication"), ("drug", "indication")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are indicated for {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which conditions have all three drugs as indicated treatments?",
        ],
    },
    {
        "name": "three_drugs_common_enzyme",
        "anchors": [("drug", "enzyme"), ("drug", "enzyme"), ("drug", "enzyme")],
        "intersection_type": "gene/protein",
        "question_templates": [
            "What enzymes metabolize {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which enzyme proteins process all three drugs?",
        ],
    },
    {
        "name": "three_drugs_common_contraindication",
        "anchors": [("drug", "contraindication"), ("drug", "contraindication"), ("drug", "contraindication")],
        "intersection_type": "disease",
        "question_templates": [
            "What diseases are contraindications for {anchor_a}, {anchor_b}, and {anchor_c}?",
            "Which conditions should all three drugs not be used for?",
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
    print("PrimeKG v4 Schema Summary")
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

    print("\n2-hop patterns:")
    for t in TWO_HOP_TEMPLATES:
        path = t["path"]
        print(f"  - {path[0]} -[{path[1]}]-> {path[2]} -[{path[3]}]-> {path[4]}")


if __name__ == "__main__":
    print_schema_summary()
