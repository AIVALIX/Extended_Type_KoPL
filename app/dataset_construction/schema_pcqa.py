"""
PcQA Schema Definition - KGT Official Format

Pan-cancer QA Knowledge Graph schema definition
Based on KGT official implementation (Graph_Schema_based_inference.py)
and actual Neo4j data.
"""

# PcQA Schema Graph - KGT Official Format
# (src_type, relation, tgt_type, direction)
# Direction is always "->" since we define both directions explicitly
SCHEMA_GRAPH = [
    # ========================================
    # Drug - Genesymbol (activation/inhibition)
    # ========================================
    ("Drug", "ACTIVATION_TO", "Genesymbol", "->"),
    ("Genesymbol", "ACTIVATION_TO", "Drug", "<-"),
    ("Drug", "INHIBITION_TO", "Genesymbol", "->"),
    ("Genesymbol", "INHIBITION_TO", "Drug", "<-"),
    # DrugAlias
    ("DrugAlias", "INHIBITION_TO", "Genesymbol", "->"),
    ("Genesymbol", "INHIBITION_TO", "DrugAlias", "<-"),

    # ========================================
    # Drug - Cancer (treatment)
    # ========================================
    ("Drug", "TREATMENT", "Cancer", "->"),
    ("Cancer", "TREATMENT", "Drug", "<-"),
    ("Drug", "TREATMENT", "CancerAlias", "->"),
    ("CancerAlias", "TREATMENT", "Drug", "<-"),
    ("Drug", "TREATMENT", "CancerCell", "->"),
    ("CancerCell", "TREATMENT", "Drug", "<-"),
    # DrugAlias
    ("DrugAlias", "TREATMENT", "Cancer", "->"),
    ("Cancer", "TREATMENT", "DrugAlias", "<-"),

    # ========================================
    # CancerCell - Drug (resistance/sensitivity)
    # ========================================
    ("CancerCell", "RESISTANCE_TO", "Drug", "->"),
    ("Drug", "RESISTANCE_TO", "CancerCell", "<-"),
    ("CancerCell", "SENSITIVITY_TO", "Drug", "->"),
    ("Drug", "SENSITIVITY_TO", "CancerCell", "<-"),

    # ========================================
    # CancerCell - Cancer (originated from)
    # ========================================
    ("CancerCell", "ORIGINATED_FROM", "Cancer", "->"),
    ("Cancer", "ORIGINATED_FROM", "CancerCell", "<-"),
    ("CancerCell", "ORIGINATED_FROM", "CancerAlias", "->"),
    ("CancerAlias", "ORIGINATED_FROM", "CancerCell", "<-"),

    # ========================================
    # CancerCell - Variants (has_var)
    # ========================================
    ("CancerCell", "HAS_VAR", "SnvFull", "->"),
    ("SnvFull", "HAS_VAR", "CancerCell", "<-"),
    ("CancerCell", "HAS_VAR", "Fusion", "->"),
    ("Fusion", "HAS_VAR", "CancerCell", "<-"),

    # ========================================
    # SnvFull/Fusion - Genesymbol (has_gene)
    # ========================================
    ("SnvFull", "HAS_GENE", "Genesymbol", "->"),
    ("Genesymbol", "HAS_GENE", "SnvFull", "<-"),
    ("Fusion", "HAS_3GENE", "Genesymbol", "->"),
    ("Genesymbol", "HAS_3GENE", "Fusion", "<-"),
    # Reverse direction (some data has this)
    ("Genesymbol", "HAS_VAR", "SnvFull", "->"),
    ("SnvFull", "HAS_VAR", "Genesymbol", "<-"),

    # ========================================
    # Genesymbol - Cancer (driving)
    # ========================================
    ("Genesymbol", "DRIVING_TO", "Cancer", "->"),
    ("Cancer", "DRIVING_TO", "Genesymbol", "<-"),
    ("Genesymbol", "DRIVING_TO", "CancerAlias", "->"),
    ("CancerAlias", "DRIVING_TO", "Genesymbol", "<-"),

    # ========================================
    # Genesymbol - Genesymbol (regulation)
    # ========================================
    ("Genesymbol", "POSITIVE_REGULATED", "Genesymbol", "->"),
    ("Genesymbol", "NEGATIVE_REGULATED", "Genesymbol", "->"),
    ("Genesymbol", "SYNTHETIC_LETHALITY", "Genesymbol", "->"),

    # ========================================
    # Genesymbol - GeneticDisease (cause)
    # ========================================
    ("Genesymbol", "CAUSE_TO", "GeneticDisease", "->"),
    ("GeneticDisease", "CAUSE_TO", "Genesymbol", "<-"),

    # ========================================
    # GeneticDisease - Cancer (develop)
    # ========================================
    ("GeneticDisease", "DEVELOP_TO", "Cancer", "->"),
    ("Cancer", "DEVELOP_TO", "GeneticDisease", "<-"),
    ("GeneticDisease", "DEVELOP_TO", "CancerAlias", "->"),
    ("CancerAlias", "DEVELOP_TO", "GeneticDisease", "<-"),

    # ========================================
    # ClinicalTrial - Cancer/Drug (include)
    # ========================================
    ("ClinicalTrial", "INCLUDE_A", "Cancer", "->"),
    ("Cancer", "INCLUDE_A", "ClinicalTrial", "<-"),
    ("ClinicalTrial", "INCLUDE_A", "Drug", "->"),
    ("Drug", "INCLUDE_A", "ClinicalTrial", "<-"),
    ("ClinicalTrial", "INCLUDE_A", "Genesymbol", "->"),
    ("Genesymbol", "INCLUDE_A", "ClinicalTrial", "<-"),

    # ========================================
    # Aliases (IS_A)
    # ========================================
    ("DrugAlias", "IS_A", "Drug", "->"),
    ("Drug", "IS_A", "DrugAlias", "<-"),
    ("CancerAlias", "IS_A", "Cancer", "->"),
    ("Cancer", "IS_A", "CancerAlias", "<-"),
    ("CancerAlias", "IS_A", "CancerAlias", "->"),
    ("Cancer", "IS_A", "Cancer", "->"),
    ("Drug", "IS_A", "Drug", "->"),
    ("Genesymbol", "IS_A", "Genesymbol", "->"),

    # ========================================
    # Drug - Cancer (induce - side effect)
    # ========================================
    ("Drug", "INDUCE_TO", "Cancer", "->"),
    ("Cancer", "INDUCE_TO", "Drug", "<-"),
]

# PcQA Entity Types (exact case from Neo4j)
ENTITY_TYPES = [
    "Cancer",
    "CancerCell",
    "CancerAlias",
    "Drug",
    "DrugAlias",
    "Genesymbol",
    "SnvFull",
    "Fusion",
    "GeneticDisease",
    "ClinicalTrial",
]

# Attribute list for each entity type (KGT format)
# Used for question analysis
ENTITY_ATTRIBUTES = {
    "Drug": [
        "drug.id", "drug.name", "drug.name_en", "drug.description",
        "drug.class_type", "drug.nmpa_approved", "drug.fda_approved",
    ],
    "Cancer": [
        "cancer.name", "cancer.description", "cancer.id", "cancer.name_en",
    ],
    "Genesymbol": [
        "genesymbol.id", "genesymbol.name", "genesymbol.description",
        "genesymbol.oncogene", "genesymbol.full_name",
    ],
    "GeneticDisease": [
        "geneticdisease.id", "geneticdisease.name", "geneticdisease.name_en",
        "geneticdisease.description",
    ],
    "ClinicalTrial": [
        "clinicaltrial.id", "clinicaltrial.name", "clinicaltrial.description",
        "clinicaltrial.phase", "clinicaltrial.status", "clinicaltrial.gender",
    ],
    "CancerCell": [
        "cancercell.id", "cancercell.name",
    ],
    "SnvFull": [
        "snvfull.id", "snvfull.name", "snvfull.biological_effect",
        "snvfull.oncogenic", "snvfull.variant_type",
    ],
    "Fusion": [
        "fusion.id", "fusion.name",
    ],
    "CancerAlias": [
        "canceralias.id", "canceralias.name",
    ],
    "DrugAlias": [
        "drugalias.id", "drugalias.name",
    ],
}

# Filterable properties for KoPL FILTER operation
# Maps node_type -> property_name -> {type, values (optional)}
FILTERABLE_PROPERTIES = {
    "Drug": {
        "fda_approved": {"type": "string", "values": ["YES", "NO"]},
        "nmpa_approved": {"type": "string", "values": ["YES", "NO"]},
        "class_type": {"type": "string"},
    },
    "ClinicalTrial": {
        "phase": {"type": "string"},
        "status": {"type": "string"},
        "gender": {"type": "string"},
    },
    "SnvFull": {
        "oncogenic": {"type": "string"},
        "biological_effect": {"type": "string"},
    },
}
