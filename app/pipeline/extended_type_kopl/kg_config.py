"""
ETK Pipeline KG Configuration

Encapsulates all KG-specific constants, prompts, and behavior for the
Extended Type-KoPL pipeline.  Adding a new KG (e.g. WebQSP) should only
require adding a new factory function here -- zero changes in pipeline.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ETKKGConfig:
    """Unified KG configuration consumed by ExtendedTypeKoPLPipeline.

    Every KG-specific constant that was previously hardcoded in pipeline.py
    is captured here so that the pipeline logic itself is KG-agnostic.
    """

    # ── identity ──────────────────────────────────────────────────
    kg_type: str  # "primekgqa", "metaqa", "pcqa", "webqsp", ...

    # ── schema ────────────────────────────────────────────────────
    entity_types: List[str]
    type_priority: List[str]  # for picking the canonical label when a node has multiple
    type_alias_map: Dict[str, str]  # LLM output normalisation -> canonical type

    # ── relation NL mapping ───────────────────────────────────────
    # {relation_name: (noun_form, verb_phrase)}
    relation_to_nl: Dict[str, Tuple[str, str]]

    # ── prompts ───────────────────────────────────────────────────
    domain_label: str  # e.g. "PrimeKGQA (Biomedical domain)"
    kopl_examples: str  # few-shot examples for _generate_type_kopl
    available_relations_text: str  # injected into the KoPL prompt
    entity_extraction_examples: str  # injected into _extract_entity_kgt_style

    # ── PcQA-style property filtering ─────────────────────────────
    filterable_properties: Dict[str, Dict[str, Any]]  # node_type -> prop -> info

    # ── answer format ─────────────────────────────────────────────
    answer_mode: str  # "entity_set" | "subgraph_nl"
    #   entity_set  -> Phase 4 returns entity names, Phase 5 does set logic
    #   subgraph_nl -> Phase 4 retrieves subgraph with properties, Phase 6 LLM generates NL

    # ── alias types (for Cypher label expansion) ──────────────────
    # e.g. {"Cancer": ["Cancer", "CancerAlias"]}
    alias_types: Dict[str, List[str]]

    # ── entity validation ─────────────────────────────────────────
    has_alias_resolution: bool  # use IS_A alias resolution during entity validation
    has_name_en_field: bool  # try name_en field during entity validation

    # ── compound entity resolution (PcQA CancerCell style) ────────
    has_compound_entities: bool  # enable CancerCell compound name resolution
    compound_entity_types: List[str] = field(default_factory=list)  # types that trigger compound resolution
    compound_keywords: List[str] = field(default_factory=list)  # keywords in question text

    # ── important property keys (for subgraph retrieval) ──────────
    important_entity_props: List[str] = field(default_factory=list)
    important_rel_props: List[str] = field(default_factory=list)


# =============================================================================
#  Relation-to-NL mappings  (shared across configs)
# =============================================================================

_METAQA_REL_NL: Dict[str, Tuple[str, str]] = {
    "DIRECTED_BY": ("director", "directed"),
    "STARRED_ACTORS": ("actor", "starred"),
    "WRITTEN_BY": ("writer", "written"),
    "IN_LANGUAGE": ("language", "in language"),
    "HAS_GENRE": ("genre", "genre"),
    "HAS_TAGS": ("tags", "tags"),
    "RELEASE_YEAR": ("release year", "released"),
    "HAS_IMDB_RATING": ("rating", "rated"),
    "HAS_IMDB_VOTES": ("votes", "votes"),
}

_PRIMEKGQA_REL_NL: Dict[str, Tuple[str, str]] = {
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
}

_PCQA_REL_NL: Dict[str, Tuple[str, str]] = {
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

# Combined mapping (kept for backward compat -- pipeline.py still uses
# get_nl_forms() which looks up a single global dict).
ALL_RELATION_NL: Dict[str, Tuple[str, str]] = {}
ALL_RELATION_NL.update(_METAQA_REL_NL)
ALL_RELATION_NL.update(_PRIMEKGQA_REL_NL)
ALL_RELATION_NL.update(_PCQA_REL_NL)


# =============================================================================
#  Type Alias Map (shared across configs, but only PCQA uses it today)
# =============================================================================

_PCQA_TYPE_ALIAS: Dict[str, str] = {
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
}

_WEBQSP_TYPE_ALIAS: Dict[str, str] = {
    # WebQSP (Freebase) type aliases – collapse redundant types to canonical ones
    # Keys lowercased to match pipeline's TYPE_ALIAS_MAP lookup convention
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


# =============================================================================
#  Prompt snippets
# =============================================================================

_PRIMEKGQA_EXAMPLES = """Examples for PrimeKGQA (Biomedical domain):

1. 1-hop: "What diseases is Metformin indicated for?"
operations: [
  {{"src_type": "drug", "tgt_type": "disease", "relation": "indication", "anchor_name": "Metformin"}}
]
final_operation: "relate"

2. 2-hop: "What phenotypes are present in diseases treated by Aspirin?"
operations: [
  {{"src_type": "drug", "tgt_type": "disease", "relation": "indication", "anchor_name": "Aspirin"}},
  {{"src_type": "disease", "tgt_type": "effect/phenotype", "relation": "phenotype present"}}
]
final_operation: "relate"

3. Intersection: "What genes are targeted by both Aspirin and Ibuprofen?"
operations: [
  {{"src_type": "drug", "tgt_type": "gene/protein", "relation": "target", "anchor_name": "Aspirin"}},
  {{"src_type": "drug", "tgt_type": "gene/protein", "relation": "target", "anchor_name": "Ibuprofen"}}
]
final_operation: "intersection\""""

_METAQA_EXAMPLES = """Examples for MetaQA (Movie domain):

1. 1-hop: "Who directed Titanic?"
operations: [
  {{"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY", "anchor_name": "Titanic"}}
]
final_operation: "relate"

2. 2-hop: "Who directed the movies that Tom Hanks starred in?"
operations: [
  {{"src_type": "Person", "tgt_type": "Movie", "relation": "STARRED_ACTORS", "anchor_name": "Tom Hanks"}},
  {{"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY"}}
]
final_operation: "relate"

3. 3-hop: "Who starred in the movies written by the writers of The Matrix?"
operations: [
  {{"src_type": "Movie", "tgt_type": "Person", "relation": "WRITTEN_BY", "anchor_name": "The Matrix"}},
  {{"src_type": "Person", "tgt_type": "Movie", "relation": "WRITTEN_BY"}},
  {{"src_type": "Movie", "tgt_type": "Person", "relation": "STARRED_ACTORS"}}
]
final_operation: "relate"

4. 3-hop: "What languages are spoken in movies directed by the director of Titanic?"
operations: [
  {{"src_type": "Movie", "tgt_type": "Person", "relation": "DIRECTED_BY", "anchor_name": "Titanic"}},
  {{"src_type": "Person", "tgt_type": "Movie", "relation": "DIRECTED_BY"}},
  {{"src_type": "Movie", "tgt_type": "Language", "relation": "IN_LANGUAGE"}}
]
final_operation: "relate"

5. 3-hop: "Who wrote movies starring actors from Candleshoe?"
operations: [
  {{"src_type": "Movie", "tgt_type": "Person", "relation": "STARRED_ACTORS", "anchor_name": "Candleshoe"}},
  {{"src_type": "Person", "tgt_type": "Movie", "relation": "STARRED_ACTORS"}},
  {{"src_type": "Movie", "tgt_type": "Person", "relation": "WRITTEN_BY"}}
]
final_operation: "relate"

IMPORTANT:
- Build path from anchor to answer: each operation's tgt_type should match next operation's src_type
- Only the first operation has anchor_name
- For 3-hop questions, carefully trace the full path: anchor entity -> intermediate entities -> intermediate entities -> answer entities"""

_PCQA_EXAMPLES = """Examples for PcQA (Pan-cancer QA domain):

1. 1-hop (drug->cancer): "What type of cancer can bexarotene treat?"
operations: [
  {{"src_type": "Drug", "tgt_type": "Cancer", "relation": "TREATMENT", "anchor_name": "bexarotene"}}
]
final_operation: "relate"

2. 2-hop (mutation->cancercell->drug): "What drugs is ALK-p.L1196M in giant cell lung cancer resistant to?"
operations: [
  {{"src_type": "SnvFull", "tgt_type": "CancerCell", "relation": "HAS_VAR", "anchor_name": "ALK-p.L1196M"}},
  {{"src_type": "CancerCell", "tgt_type": "Drug", "relation": "RESISTANCE_TO"}}
]
final_operation: "relate"

3. 2-hop (cancer->cancercell->mutation): "What genetic mutations are present in ovarian cancer?"
operations: [
  {{"src_type": "Cancer", "tgt_type": "CancerCell", "relation": "ORIGINATED_FROM", "anchor_name": "ovarian cancer"}},
  {{"src_type": "CancerCell", "tgt_type": "SnvFull", "relation": "HAS_VAR"}}
]
final_operation: "relate"

4. 1-hop (cancer->drug): "What drugs can treat renal cell carcinoma?"
operations: [
  {{"src_type": "Cancer", "tgt_type": "Drug", "relation": "TREATMENT", "anchor_name": "renal cell carcinoma"}}
]
final_operation: "relate"

5. 2-hop (mutation->cancercell->cancer): "What type of cancer can be driven by PTEN-p.R173C?"
operations: [
  {{"src_type": "SnvFull", "tgt_type": "CancerCell", "relation": "HAS_VAR", "anchor_name": "PTEN-p.R173C"}},
  {{"src_type": "CancerCell", "tgt_type": "Cancer", "relation": "ORIGINATED_FROM"}}
]
final_operation: "relate"

6. 1-hop (drug->gene): "Which genes does alectinib inhibit?"
operations: [
  {{"src_type": "Drug", "tgt_type": "Genesymbol", "relation": "INHIBITION_TO", "anchor_name": "alectinib"}}
]
final_operation: "relate"

7. 1-hop (gene->drug): "What drugs inhibit FGFR3?"
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "FGFR3"}}
]
final_operation: "relate"

8a. 1-hop (gene->drug): "What targeted therapies are available for ERBB2?"
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "ERBB2"}}
]
final_operation: "relate"

8b. 1-hop (gene->drug): "What drugs can treat cancers with ALK mutations?"
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "ALK"}}
]
final_operation: "relate"

8c. 1-hop (gene->drug): "What genetic mutations need to be tested for erlotinib?"
NOTE: Despite mentioning "mutations", the anchor entity is the GENE (e.g. EGFR), and the answer is DRUGS that inhibit it.
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "EGFR"}}
]
final_operation: "relate"

8d. 1-hop (gene->drug): "What is the relationship between BRAF and the therapeutic effect of PLX8394?"
NOTE: When asking about relationship between a gene and a drug, the answer is DRUGS that inhibit the gene.
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "BRAF"}}
]
final_operation: "relate"

8e. 1-hop (gene->drug): "How does EGFR gene mutation affect the efficacy of osimertinib?"
NOTE: Questions about "efficacy" or "therapeutic effect" of a drug on a gene -> answer is drugs that inhibit the gene.
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "EGFR"}}
]
final_operation: "relate"

9. 2-hop (gene->drug->cancer): "What cancers can be treated by drugs targeting BRAF?"
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "BRAF"}},
  {{"src_type": "Drug", "tgt_type": "Cancer", "relation": "TREATMENT"}}
]
final_operation: "relate"

IMPORTANT:
- Build path from anchor to answer: each operation's tgt_type should match next operation's src_type
- Only the first operation has anchor_name
- For queries about mutations in a cancer, path MUST go through CancerCell:
  * Cancer -> CancerCell (ORIGINATED_FROM) -> SnvFull (HAS_VAR)
- For queries about what cancer a mutation drives, path MUST go through CancerCell:
  * SnvFull -> CancerCell (HAS_VAR) -> Cancer (ORIGINATED_FROM)
- Do NOT use Genesymbol -> Cancer (DRIVING_TO) for mutation-related queries
- Relations are BIDIRECTIONAL: (Drug)-[INHIBITION_TO]->(Genesymbol) can be traversed as Genesymbol->Drug
  * "What drugs inhibit gene X?" -> src_type=Genesymbol, tgt_type=Drug, relation=INHIBITION_TO
  * "What genes does drug X inhibit?" -> src_type=Drug, tgt_type=Genesymbol, relation=INHIBITION_TO
- Similarly, (Drug)-[TREATMENT]->(Cancer) can be traversed as Cancer->Drug
- The ANSWER TYPE determines tgt_type, the ANCHOR determines src_type

8. 1-hop + FILTER (gene->drug, NMPA filter): "What are the NMPA-approved drugs for cancers with BRAF mutations?"
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "BRAF"}}
]
filters: [
  {{"node_type": "Drug", "property_name": "nmpa_approved", "operator": "=", "value": "YES"}}
]
final_operation: "relate"

9. 1-hop + FILTER (gene->drug, FDA filter): "What FDA-approved drugs target CDK4?"
operations: [
  {{"src_type": "Genesymbol", "tgt_type": "Drug", "relation": "INHIBITION_TO", "anchor_name": "CDK4"}}
]
filters: [
  {{"node_type": "Drug", "property_name": "fda_approved", "operator": "=", "value": "YES"}}
]
final_operation: "relate"

FILTER RULES:
- Use filters when the question mentions approval status (FDA-approved, NMPA-approved) or other property constraints
- filters is an array of objects: {{"node_type": "...", "property_name": "...", "operator": "=", "value": "..."}}
- node_type must match one of the types in the operations
- If no filtering is needed, omit the filters field or set it to []"""

_PCQA_AVAILABLE_RELATIONS = """Available relations (src_type)-[RELATION]->(tgt_type):
  (Drug)-[ACTIVATION_TO]->(Genesymbol)
  (Drug)-[INHIBITION_TO]->(Genesymbol)
  (Drug)-[TREATMENT]->(Cancer)
  (Drug)-[INDUCE_TO]->(Cancer)
  (CancerCell)-[RESISTANCE_TO]->(Drug)
  (CancerCell)-[SENSITIVITY_TO]->(Drug)
  (CancerCell)-[ORIGINATED_FROM]->(Cancer)
  (CancerCell)-[HAS_VAR]->(SnvFull)
  (CancerCell)-[HAS_VAR]->(Fusion)
  (SnvFull)-[HAS_GENE]->(Genesymbol)
  (Fusion)-[HAS_3GENE]->(Genesymbol)
  (Genesymbol)-[DRIVING_TO]->(Cancer)
  (Genesymbol)-[CAUSE_TO]->(GeneticDisease)
  (GeneticDisease)-[DEVELOP_TO]->(Cancer)
NOTE: INHIBITION_TO means a drug inhibits/targets a gene. Use this when asking about drugs for a gene's mutations.
NOTE: Relations can be traversed in EITHER direction. src_type is the ANCHOR entity's type, tgt_type is the ANSWER entity's type.
  e.g. "What drugs inhibit FGFR3?" -> src_type=Genesymbol, tgt_type=Drug, relation=INHIBITION_TO (anchor is the gene)"""

_PCQA_ENTITY_EXTRACTION_EXAMPLES = """Examples for PcQA (Pan-cancer QA):
- "What types of cancer can be treated with irinotecan?"
  -> entity_name: "irinotecan", entity_type: "Drug", target_type: "Cancer"
- "Which genes can be activated by codeine?"
  -> entity_name: "codeine", entity_type: "Drug", target_type: "Genesymbol"
- "Which types of cancer are associated with MET?"
  -> entity_name: "MET", entity_type: "Genesymbol", target_type: "Cancer"
- "What drugs can treat cancers with TERT mutations?"
  -> entity_name: "TERT", entity_type: "Genesymbol", target_type: "Drug"
- "What drugs is ALK-p.L1196M in giant cell lung cancer resistant to?"
  -> entity_name: "ALK-p.L1196M", entity_type: "SnvFull", target_type: "Drug"
- "What are the fusion genes in melanoma?"
  -> entity_name: "melanoma", entity_type: "Cancer", target_type: "Fusion"
- "How would you describe docetaxel?"
  -> entity_name: "docetaxel", entity_type: "Drug", target_type: "Drug"
- "What are the NMPA-approved drugs for cancers with DDR2 mutations?"
  -> entity_name: "DDR2", entity_type: "Genesymbol", target_type: "Drug"
"""

_WEBQSP_EXAMPLES = """Examples for WebQSP (Freebase / open-domain):

1. 1-hop (Country -> Language): "What language do they speak in Jamaica?"
operations: [
  {{"src_type": "Country", "tgt_type": "Human_Language", "relation": "location.country.official_language", "anchor_name": "Jamaica"}}
]
final_operation: "relate"

2. 1-hop (Person -> Location): "Where was Nelson Mandela born?"
operations: [
  {{"src_type": "Person", "tgt_type": "Location", "relation": "people.person.place_of_birth", "anchor_name": "Nelson Mandela"}}
]
final_operation: "relate"

3. 1-hop (Film -> Person): "Who directed Gladiator?"
operations: [
  {{"src_type": "Film", "tgt_type": "Person", "relation": "film.film.directed_by", "anchor_name": "Gladiator"}}
]
final_operation: "relate"

4. 1-hop (Book -> Person): "Who wrote Pride and Prejudice?"
operations: [
  {{"src_type": "Book", "tgt_type": "Person", "relation": "book.written_work.author", "anchor_name": "Pride and Prejudice"}}
]
final_operation: "relate"

5. 1-hop (Country -> Country): "What countries are part of the United Kingdom?"
operations: [
  {{"src_type": "Country", "tgt_type": "Country", "relation": "location.country.administrative_divisions", "anchor_name": "United Kingdom"}}
]
final_operation: "relate"

6. 2-hop (Film -> Person -> Country): "What country is the director of Gladiator from?"
operations: [
  {{"src_type": "Film", "tgt_type": "Person", "relation": "film.film.directed_by", "anchor_name": "Gladiator"}},
  {{"src_type": "Person", "tgt_type": "Country", "relation": "people.person.nationality"}}
]
final_operation: "relate"

IMPORTANT:
- Freebase relations use dot-notation: domain.type.property (e.g., people.person.nationality, film.film.directed_by)
- Build path from anchor to answer: each operation's tgt_type should match next operation's src_type
- Only the first operation has anchor_name
- Use "Person" as tgt_type for people (not Film_director, Film_actor, Author -- those are aliases for Person)
- Common Freebase relation domains: people, film, location, book, music, government, sports, organization, education"""

_WEBQSP_ENTITY_EXTRACTION_EXAMPLES = """Examples for WebQSP (open-domain):
- "What language do they speak in Jamaica?"
  -> entity_name: "Jamaica", entity_type: "Country", target_type: "Human_Language"
- "Where was Nelson Mandela born?"
  -> entity_name: "Nelson Mandela", entity_type: "Person", target_type: "Location"
- "Who directed Gladiator?"
  -> entity_name: "Gladiator", entity_type: "Film", target_type: "Person"
- "Who wrote Pride and Prejudice?"
  -> entity_name: "Pride and Prejudice", entity_type: "Book", target_type: "Person"
- "What countries are part of the United Kingdom?"
  -> entity_name: "United Kingdom", entity_type: "Country", target_type: "Country"
"""

_PRIMEKGQA_ENTITY_EXTRACTION_EXAMPLES = ""
_METAQA_ENTITY_EXTRACTION_EXAMPLES = ""

# PcQA filterable properties
_PCQA_FILTERABLE: Dict[str, Dict[str, Any]] = {
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

# PcQA important properties for subgraph retrieval
_PCQA_IMPORTANT_ENTITY_PROPS = [
    "name", "name_en", "fda_approved", "nmpa_approved",
    "cancer_type", "drug_class", "target_gene", "mutation_type",
    "phase", "status", "gender", "location", "evidence_level", "class_type",
]
_PCQA_IMPORTANT_REL_PROPS = [
    "fda_approved", "nmpa_approved", "score",
    "evidence_level", "phase", "status", "class_type",
]


# =============================================================================
#  Auto-generation of relation NL forms
# =============================================================================


def auto_generate_relation_nl(relation_name: str) -> Tuple[str, str]:
    """Auto-generate (noun_form, verb_phrase) from a relation name.

    Works for any naming convention -- no manual curation required:
    - Freebase:    ``domain.type.property`` -> extract property, humanize
    - Underscore:  ``some_relation`` -> ``"some relation"``
    - CamelCase:   ``SomeRelation`` -> ``"some relation"``
    - UPPER_SNAKE: ``STARRED_ACTORS`` -> ``"starred actors"``

    Returns a ``(noun, verb)`` tuple where both forms are the humanised
    property name.  Since auto-generated verb forms (e.g. prefixing "has")
    can hurt embedding similarity, we keep noun == verb and rely on the
    embedding model to bridge the gap.
    """
    # 1. Extract the most specific segment
    if "." in relation_name:
        # Freebase style: domain.type.property -> take last segment
        prop = relation_name.rsplit(".", 1)[-1]
    else:
        prop = relation_name

    # 2. Expand CamelCase boundaries (e.g. "languagesSpoken" -> "languages Spoken")
    prop = re.sub(r"([a-z])([A-Z])", r"\1 \2", prop)
    # Also handle sequences of uppercase (e.g. "HTMLParser" -> "HTML Parser")
    prop = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", prop)

    # 3. Replace underscores / hyphens with spaces and lowercase
    noun = prop.replace("_", " ").replace("-", " ").strip().lower()

    # Collapse any double-spaces
    noun = re.sub(r"\s+", " ", noun)

    if not noun:
        noun = relation_name.lower()

    # 4. Verb phrase = noun (safe default for embedding similarity)
    return (noun, noun)


# =============================================================================
#  Factory functions
# =============================================================================


def build_etk_kg_config(kg_type: str) -> ETKKGConfig:
    """Build an ETKKGConfig for the given KG type.

    Raises ValueError for unknown KG types.
    """
    builders = {
        "primekgqa": _build_primekgqa,
        "metaqa": _build_metaqa,
        "pcqa": _build_pcqa,
        "webqsp": _build_webqsp,
        "kqapro": _build_kqapro,
    }
    builder = builders.get(kg_type)
    if builder is None:
        raise ValueError(
            f"Unknown KG type for ETK pipeline: '{kg_type}'. "
            f"Available: {list(builders.keys())}"
        )
    return builder()


def _build_primekgqa() -> ETKKGConfig:
    return ETKKGConfig(
        kg_type="primekgqa",
        entity_types=[
            "drug", "disease", "gene/protein", "exposure",
            "biological_process", "molecular_function",
            "cellular_component", "pathway", "anatomy", "effect/phenotype",
        ],
        type_priority=["drug", "disease", "gene/protein"],
        type_alias_map={},
        relation_to_nl=_PRIMEKGQA_REL_NL,
        domain_label="PrimeKGQA (Biomedical domain)",
        kopl_examples=_PRIMEKGQA_EXAMPLES,
        available_relations_text="",
        entity_extraction_examples=_PRIMEKGQA_ENTITY_EXTRACTION_EXAMPLES,
        filterable_properties={},
        answer_mode="entity_set",
        alias_types={},
        has_alias_resolution=False,
        has_name_en_field=False,
        has_compound_entities=False,
    )


def _build_metaqa() -> ETKKGConfig:
    return ETKKGConfig(
        kg_type="metaqa",
        entity_types=[
            "movie", "person", "organization", "text",
            "date", "language", "number",
        ],
        type_priority=[
            "movie", "person", "language", "date",
            "text", "number", "organization",
        ],
        type_alias_map={},
        relation_to_nl=_METAQA_REL_NL,
        domain_label="MetaQA (Movie domain)",
        kopl_examples=_METAQA_EXAMPLES,
        available_relations_text="",
        entity_extraction_examples=_METAQA_ENTITY_EXTRACTION_EXAMPLES,
        filterable_properties={},
        answer_mode="entity_set",
        alias_types={},
        has_alias_resolution=False,
        has_name_en_field=False,
        has_compound_entities=False,
    )


def _build_pcqa() -> ETKKGConfig:
    return ETKKGConfig(
        kg_type="pcqa",
        entity_types=[
            "cancer", "cancercell", "canceralias", "drug", "drugalias",
            "genesymbol", "snvfull", "fusion", "geneticdisease", "clinicaltrial",
        ],
        type_priority=["drug", "cancer", "genesymbol", "snvfull"],
        type_alias_map=_PCQA_TYPE_ALIAS,
        relation_to_nl=_PCQA_REL_NL,
        domain_label="PcQA (Pan-cancer QA domain)",
        kopl_examples=_PCQA_EXAMPLES,
        available_relations_text=_PCQA_AVAILABLE_RELATIONS,
        entity_extraction_examples=_PCQA_ENTITY_EXTRACTION_EXAMPLES,
        filterable_properties=_PCQA_FILTERABLE,
        answer_mode="subgraph_nl",
        alias_types={
            "Cancer": ["Cancer", "CancerAlias"],
            "Drug": ["Drug", "DrugAlias"],
        },
        has_alias_resolution=True,
        has_name_en_field=True,
        has_compound_entities=True,
        compound_entity_types=["genesymbol", "fusion"],
        compound_keywords=[
            "cell line", "cell lines", "cancer cell",
            "resistance", "resistant", "sensitivity", "sensitive",
        ],
        important_entity_props=_PCQA_IMPORTANT_ENTITY_PROPS,
        important_rel_props=_PCQA_IMPORTANT_REL_PROPS,
    )


def _build_webqsp() -> ETKKGConfig:
    """WebQSP configuration (Freebase-based open-domain KG).

    Entity types and type priority are derived from the actual Neo4j
    label distribution.  Relation NL mappings are auto-generated from
    Freebase-style dotted names via ``auto_generate_relation_nl``.
    """
    return ETKKGConfig(
        kg_type="webqsp",
        entity_types=[
            "Entity", "CVT", "Film", "Book", "Musical_Recording",
            "City_Town_Village", "TV_Episode", "Person",
            "Musical_Album", "Military_Conflict", "Author",
            "Administrative_Division", "Composition", "Deceased_Person",
            "Quotation", "Film_actor", "Organization",
            "Musical_Artist", "Film_character", "Political_party",
            "Event", "Politician", "Location",
            "Government_Agency", "AwardWinning_Work", "College_University",
            "Mountain", "Film_crewmember", "TV_Program",
            "US_County", "Celebrity", "Human_Language",
            "Tourist_attraction", "Building", "Airport",
            "Country", "Film_director", "Sports_Team",
            "Composer", "Visual_Artist", "Academic",
            "TV_Producer", "TV_Writer", "Profession",
            "Fictional_Character", "Film_producer", "Production_company",
            "Film_genre", "TV_Genre", "Media_genre", "Musical_genre",
            "Film_rating", "Film_writer", "Organization_leader",
            "Family", "Religion", "Risk_Factor",
            "US_President", "Monarch", "Athlete",
            "Basketball_Player", "Baseball_Player", "Football_player",
            "American_football_player",
        ],
        type_priority=[
            "Person", "Film", "Country", "City_Town_Village",
            "Author", "Book", "Film_director", "Film_actor",
            "Organization", "Human_Language", "Musical_Artist",
            "Politician", "US_President", "Location",
            "Sports_Team", "College_University",
        ],
        type_alias_map=_WEBQSP_TYPE_ALIAS,
        relation_to_nl={},  # auto-generated at runtime via auto_generate_relation_nl
        domain_label="WebQSP (Freebase open-domain)",
        kopl_examples=_WEBQSP_EXAMPLES,
        available_relations_text="",
        entity_extraction_examples=_WEBQSP_ENTITY_EXTRACTION_EXAMPLES,
        filterable_properties={},
        answer_mode="entity_set",
        alias_types={},
        has_alias_resolution=False,
        has_name_en_field=False,
        has_compound_entities=False,
    )


_KQAPRO_EXAMPLES = """Examples for KQA Pro (Wikidata open-domain):

--- Entity retrieval (answer_type: "entity") ---

1. 1-hop: "Who directed Forrest Gump?"
operations: [{{"src_type": "film", "tgt_type": "human", "relation": "director", "anchor_name": "Forrest Gump"}}]
final_operation: "relate"
answer_type: "entity"

2. 2-hop: "Who are the cast members of films directed by Steven Spielberg?"
operations: [
  {{"src_type": "human", "tgt_type": "film", "relation": "director", "anchor_name": "Steven Spielberg"}},
  {{"src_type": "film", "tgt_type": "human", "relation": "cast_member"}}
]
final_operation: "relate"
answer_type: "entity"

--- Intersection (answer_type: "entity", final_operation: "intersection") ---

3. "What film has the genre of romance and has Ava Gardner as a cast member?"
operations: [
  {{"src_type": "film", "tgt_type": "Concept", "relation": "genre", "anchor_name": "romance film"}},
  {{"src_type": "film", "tgt_type": "human", "relation": "cast_member", "anchor_name": "Ava Gardner"}}
]
final_operation: "intersection"
answer_type: "entity"

--- Count (answer_type: "count") ---

4. "How many films did Steven Spielberg direct?"
operations: [{{"src_type": "human", "tgt_type": "film", "relation": "director", "anchor_name": "Steven Spielberg"}}]
final_operation: "relate"
answer_type: "count"

--- Attribute query (answer_type: "attr") ---

5. "What is the population of Tokyo?"
operations: [{{"src_type": "city", "tgt_type": "city", "relation": "self", "anchor_name": "Tokyo"}}]
final_operation: "relate"
answer_type: "attr"
query_key: "population"

--- Relation query (answer_type: "relation") ---

6. "What is the relationship between Forrest Gump and English?"
operations: [
  {{"src_type": "film", "tgt_type": "film", "relation": "self", "anchor_name": "Forrest Gump"}}
]
final_operation: "relate"
answer_type: "relation"
select_entity_a: "Forrest Gump"
select_entity_b: "English"

--- Verification (answer_type: "verify") ---

7. "Was Forrest Gump released in 1994?"
operations: [{{"src_type": "film", "tgt_type": "film", "relation": "self", "anchor_name": "Forrest Gump"}}]
final_operation: "relate"
answer_type: "verify"
query_key: "publication date"
verify_value: "1994"
verify_op: "="

8. "Is 129586 the exploitation visa number of Bridget Jones's Diary?"
operations: [{{"src_type": "film", "tgt_type": "film", "relation": "self", "anchor_name": "Bridget Jones's Diary"}}]
final_operation: "relate"
answer_type: "verify"
query_key: "exploitation visa number"
verify_value: "129586"
verify_op: "="

--- Select between two entities (answer_type: "select") ---

9. "Does My Neighbor Totoro or Hannah Arendt have the longer run-time?"
operations: []
final_operation: "relate"
answer_type: "select"
query_key: "duration"
select_mode: "greater"
select_entity_a: "My Neighbor Totoro"
select_entity_b: "Hannah Arendt"

--- Select among filtered set (answer_type: "select") ---

10. "Which former French region has the smallest population?"
operations: [{{"src_type": "Concept", "tgt_type": "Concept", "relation": "self", "anchor_name": "former French region"}}]
final_operation: "relate"
answer_type: "select"
query_key: "population"
select_mode: "smallest"

--- Filter with property constraints ---

11. "Which person is a member of the Democratic Party and born on 1954-03-11?"
operations: [{{"src_type": "organization", "tgt_type": "human", "relation": "member_of_political_party", "anchor_name": "Democratic Party"}}]
final_operation: "relate"
answer_type: "entity"
filters: [{{"node_type": "human", "property_name": "date_of_birth", "operator": "=", "value": "1954-03-11"}}]

IMPORTANT:
- Build path from anchor to answer entity
- Use natural relation names with underscores (e.g., cast_member, place_of_birth)
- Only the first operation (or each branch in intersection) has anchor_name
- For attr/verify/select questions, operations may be empty or just locate the entity
- query_key must match the KG property name exactly"""

_KQAPRO_ENTITY_EXTRACTION_EXAMPLES = """Examples for KQA Pro (open-domain):
- "Who directed Forrest Gump?"
  -> entity_name: "Forrest Gump", entity_type: "film", target_type: "human"
- "What country is the University of Oxford in?"
  -> entity_name: "University of Oxford", entity_type: "university", target_type: "sovereign_state"
- "What genre is Star Wars?"
  -> entity_name: "Star Wars", entity_type: "film", target_type: "Concept"
"""


def _build_kqapro() -> ETKKGConfig:
    return ETKKGConfig(
        kg_type="kqapro",
        entity_types=[
            "human", "film", "sovereign_state", "city", "university",
            "television_series", "organization", "video_game", "award",
            "band", "business", "written_work", "Concept",
            "association_football_club", "administrative_territorial_entity",
        ],
        type_priority=[
            "human", "film", "sovereign_state", "city", "university",
            "organization", "television_series", "Concept",
        ],
        type_alias_map={},
        relation_to_nl={},
        domain_label="KQA Pro (Wikidata open-domain)",
        kopl_examples=_KQAPRO_EXAMPLES,
        available_relations_text="",
        entity_extraction_examples=_KQAPRO_ENTITY_EXTRACTION_EXAMPLES,
        filterable_properties={},
        answer_mode="entity_set",
        alias_types={},
        has_alias_resolution=False,
        has_name_en_field=False,
        has_compound_entities=False,
    )
