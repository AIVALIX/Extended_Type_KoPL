"""
Build PCQA evaluation dataset using LLM pattern classification + Cypher-verified answers.

For each question:
1. LLM classifies query pattern and extracts seed entity
2. Cypher query traverses KG to get actual answers
3. Output includes entity, path, filters, and verified answers

Usage (inside Docker):
    python /app/data/pcqa/build_eval_cypher.py
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel, Field
from py2neo import Graph


def _has_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


# --- Query Pattern Definitions ---

QUERY_PATTERNS = {
    "drug_treats_cancer": {
        "description": "What cancers can drug X treat?",
        "cypher": """
            MATCH (d)-[:TREATMENT]->(c)
            WHERE toLower(d.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(c)
            RETURN DISTINCT c.name AS name, labels(c) AS labels
        """,
        "path": "(Drug/DrugAlias)-[TREATMENT]->(Cancer/CancerAlias)",
        "seed_type": "Drug",
        "answer_type": "Cancer",
    },
    "gene_drives_cancer": {
        "description": "What cancers are associated with gene X?",
        "cypher": """
            MATCH (g:Genesymbol)-[:DRIVING_TO]->(c)
            WHERE toLower(g.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(c)
            RETURN DISTINCT c.name AS name, labels(c) AS labels
        """,
        "path": "(Genesymbol)-[DRIVING_TO]->(Cancer/CancerAlias)",
        "seed_type": "Genesymbol",
        "answer_type": "Cancer",
    },
    "mutation_drives_cancer": {
        "description": "What cancers can mutation X drive?",
        "cypher": """
            MATCH (s)<-[:HAS_VAR]-(cc:CancerCell)-[:ORIGINATED_FROM]->(c)
            WHERE toLower(s.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(c)
            RETURN DISTINCT c.name AS name, labels(c) AS labels
        """,
        "path": "(SnvFull/Fusion)<-[HAS_VAR]-(CancerCell)-[ORIGINATED_FROM]->(Cancer)",
        "seed_type": "SnvFull",
        "answer_type": "Cancer",
    },
    "drug_activates_gene": {
        "description": "What genes does drug X activate?",
        "cypher": """
            MATCH (d)-[:ACTIVATION_TO]->(g:Genesymbol)
            WHERE toLower(d.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(g)
            RETURN DISTINCT g.name AS name, labels(g) AS labels
        """,
        "path": "(Drug)-[ACTIVATION_TO]->(Genesymbol)",
        "seed_type": "Drug",
        "answer_type": "Genesymbol",
    },
    "drug_inhibits_gene": {
        "description": "What genes does drug X inhibit?",
        "cypher": """
            MATCH (d)-[:INHIBITION_TO]->(g:Genesymbol)
            WHERE toLower(d.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(g)
            RETURN DISTINCT g.name AS name, labels(g) AS labels
        """,
        "path": "(Drug)-[INHIBITION_TO]->(Genesymbol)",
        "seed_type": "Drug",
        "answer_type": "Genesymbol",
    },
    "gene_inhibited_by_drug": {
        "description": "What drugs inhibit gene X?",
        "cypher": """
            MATCH (d)-[:INHIBITION_TO]->(g:Genesymbol)
            WHERE toLower(g.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(d)
            RETURN DISTINCT d.name AS name, labels(d) AS labels
        """,
        "path": "(Drug)-[INHIBITION_TO]->(Genesymbol)",
        "seed_type": "Genesymbol",
        "answer_type": "Drug",
    },
    "gene_inhibited_by_drug_nmpa": {
        "description": "What NMPA-approved drugs inhibit gene X?",
        "cypher": """
            MATCH (d)-[:INHIBITION_TO]->(g:Genesymbol)
            WHERE toLower(g.name) = $seed
              AND d.nmpa_approved = 'YES'
              AND NOT 'RelationEmbedding' IN labels(d)
            RETURN DISTINCT d.name AS name, labels(d) AS labels
        """,
        "path": "(Drug {nmpa_approved:'YES'})-[INHIBITION_TO]->(Genesymbol)",
        "seed_type": "Genesymbol",
        "answer_type": "Drug",
        "filters": {"nmpa_approved": "YES"},
    },
    "gene_inhibited_by_drug_fda": {
        "description": "What FDA-approved drugs inhibit gene X?",
        "cypher": """
            MATCH (d)-[:INHIBITION_TO]->(g:Genesymbol)
            WHERE toLower(g.name) = $seed
              AND d.fda_approved = 'YES'
              AND NOT 'RelationEmbedding' IN labels(d)
            RETURN DISTINCT d.name AS name, labels(d) AS labels
        """,
        "path": "(Drug {fda_approved:'YES'})-[INHIBITION_TO]->(Genesymbol)",
        "seed_type": "Genesymbol",
        "answer_type": "Drug",
        "filters": {"fda_approved": "YES"},
    },
    "cancer_has_mutations": {
        "description": "What mutations are present in cancer X?",
        "cypher": """
            MATCH (c)<-[:ORIGINATED_FROM]-(cc:CancerCell)-[:HAS_VAR]->(s)
            WHERE toLower(c.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(s)
            RETURN DISTINCT s.name AS name, labels(s) AS labels
        """,
        "path": "(Cancer)<-[ORIGINATED_FROM]-(CancerCell)-[HAS_VAR]->(SnvFull/Fusion)",
        "seed_type": "Cancer",
        "answer_type": "SnvFull",
    },
    "cancer_has_fusions": {
        "description": "What fusion genes are in cancer X?",
        "cypher": """
            MATCH (c)<-[:ORIGINATED_FROM]-(cc:CancerCell)-[:HAS_VAR]->(f:Fusion)
            WHERE toLower(c.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(f)
            RETURN DISTINCT f.name AS name, labels(f) AS labels
        """,
        "path": "(Cancer)<-[ORIGINATED_FROM]-(CancerCell)-[HAS_VAR]->(Fusion)",
        "seed_type": "Cancer",
        "answer_type": "Fusion",
    },
    "cancercell_resistance_drug": {
        "description": "What drugs is gene X in cancer Y resistant to?",
        "cypher": """
            MATCH (cc:CancerCell)-[:RESISTANCE_TO]->(d)
            WHERE toLower(cc.name) CONTAINS $seed
              AND NOT 'RelationEmbedding' IN labels(d)
            RETURN DISTINCT d.name AS name, labels(d) AS labels
        """,
        "path": "(CancerCell)-[RESISTANCE_TO]->(Drug)",
        "seed_type": "CancerCell",
        "answer_type": "Drug",
    },
    "cancercell_sensitivity_drug": {
        "description": "What drugs is gene X in cancer Y sensitive to?",
        "cypher": """
            MATCH (cc:CancerCell)-[:SENSITIVITY_TO]->(d)
            WHERE toLower(cc.name) CONTAINS $seed
              AND NOT 'RelationEmbedding' IN labels(d)
            RETURN DISTINCT d.name AS name, labels(d) AS labels
        """,
        "path": "(CancerCell)-[SENSITIVITY_TO]->(Drug)",
        "seed_type": "CancerCell",
        "answer_type": "Drug",
    },
    "gene_causes_disease": {
        "description": "What diseases does gene X cause?",
        "cypher": """
            MATCH (g:Genesymbol)-[:CAUSE_TO]->(d:GeneticDisease)
            WHERE toLower(g.name) = $seed
            RETURN DISTINCT d.name AS name, labels(d) AS labels
        """,
        "path": "(Genesymbol)-[CAUSE_TO]->(GeneticDisease)",
        "seed_type": "Genesymbol",
        "answer_type": "GeneticDisease",
    },
    "disease_develops_cancer": {
        "description": "What cancers does disease X develop into?",
        "cypher": """
            MATCH (d:GeneticDisease)-[:DEVELOP_TO]->(c)
            WHERE toLower(d.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(c)
            RETURN DISTINCT c.name AS name, labels(c) AS labels
        """,
        "path": "(GeneticDisease)-[DEVELOP_TO]->(Cancer)",
        "seed_type": "GeneticDisease",
        "answer_type": "Cancer",
    },
    "gene_alias": {
        "description": "What is the alternative name for gene X?",
        "cypher": """
            MATCH (g1:Genesymbol)-[:IS_A]->(g2:Genesymbol)
            WHERE toLower(g1.name) = $seed
            RETURN DISTINCT g2.name AS name, labels(g2) AS labels
        """,
        "path": "(Genesymbol)-[IS_A]->(Genesymbol)",
        "seed_type": "Genesymbol",
        "answer_type": "Genesymbol",
    },
    "treatment_for_cancercell": {
        "description": "How to treat cancer X with gene Y? (find drugs via CancerCell)",
        "cypher": """
            MATCH (cc:CancerCell)-[:SENSITIVITY_TO]->(d)
            WHERE toLower(cc.name) CONTAINS $seed
              AND NOT 'RelationEmbedding' IN labels(d)
            RETURN DISTINCT d.name AS name, labels(d) AS labels
        """,
        "path": "(CancerCell)-[SENSITIVITY_TO]->(Drug)",
        "seed_type": "CancerCell",
        "answer_type": "Drug",
    },
    "disease_induce_cancer": {
        "description": "What cancer can disease X induce?",
        "cypher": """
            MATCH (d:GeneticDisease)-[:DEVELOP_TO]->(c)
            WHERE toLower(d.name) = $seed
              AND NOT 'RelationEmbedding' IN labels(c)
            RETURN DISTINCT c.name AS name, labels(c) AS labels
        """,
        "path": "(GeneticDisease)-[DEVELOP_TO]->(Cancer)",
        "seed_type": "GeneticDisease",
        "answer_type": "Cancer",
    },
}


# --- LLM Schema ---

class QueryClassification(BaseModel):
    """LLM classification of a PCQA question"""
    query_pattern: str = Field(
        description="One of: drug_treats_cancer, gene_drives_cancer, mutation_drives_cancer, "
                    "drug_activates_gene, drug_inhibits_gene, gene_inhibited_by_drug, "
                    "gene_inhibited_by_drug_nmpa, gene_inhibited_by_drug_fda, "
                    "cancer_has_mutations, cancer_has_fusions, "
                    "cancercell_resistance_drug, cancercell_sensitivity_drug, "
                    "gene_causes_disease, disease_develops_cancer, gene_alias, "
                    "treatment_for_cancercell, disease_induce_cancer, "
                    "not_entity_query"
    )
    seed_entity: str = Field(
        description="The seed entity from the question, exactly as it should appear in the KG. "
                    "For genes: uppercase (MET, TERT, EGFR). "
                    "For mutations: exact format (EGFR-p.L861R, PTEN-p.R173C). "
                    "For drugs: lowercase (doxorubicin, pomalidomide). "
                    "For cancers: lowercase (breast cancer, gastric cancer). "
                    "For diseases: as named (Li-Fraumeni syndrome). "
                    "For CancerCell queries (resistance/sensitivity/treatment): "
                    "use the gene name as seed (e.g., 'FLT3' from 'FLT3 in myeloid')."
    )
    seed_entity_secondary: Optional[str] = Field(
        default=None,
        description="Secondary seed entity if the query involves two entities. "
                    "For resistance/sensitivity: the cancer name. "
                    "For treatment_for_cancercell: the cancer name. "
                    "E.g., 'FLT3 in myeloid' -> seed='FLT3', secondary='myeloid'"
    )


CLASSIFICATION_PROMPT = """You are a biomedical knowledge graph expert. Classify this question into a query pattern.

Available patterns:
- drug_treats_cancer: "What cancers can drug X treat?" or "What types of cancer can X be used to treat?"
- gene_drives_cancer: "What cancers are associated with gene X?" (seed is a gene symbol like MET, TERT)
- mutation_drives_cancer: "What cancers can mutation X drive?" (seed is a mutation like PTEN-p.R173C, BRAF-p.T599K)
- drug_activates_gene: "What genes does drug X activate?"
- drug_inhibits_gene: "What genes does drug X inhibit?"
- gene_inhibited_by_drug: "What drugs can treat/inhibit gene X?" (general, no approval filter)
- gene_inhibited_by_drug_nmpa: "What NMPA-approved drugs for gene X?" (has NMPA filter)
- gene_inhibited_by_drug_fda: "What FDA-approved drugs for gene X?" (has FDA filter)
- cancer_has_mutations: "What mutations are in cancer X?"
- cancer_has_fusions: "What fusion genes are in cancer X?"
- cancercell_resistance_drug: "What drugs is gene X in cancer Y resistant to?"
- cancercell_sensitivity_drug: "What drugs is gene X in cancer Y sensitive to?"
- gene_causes_disease: "What diseases does gene X cause?"
- disease_develops_cancer: "What cancers does disease X develop into?"
- gene_alias: "What is the alternative name for gene X?"
- treatment_for_cancercell: "How to treat cancer X with gene Y mutation?" (find drugs via CancerCell sensitivity)
- disease_induce_cancer: "What cancer can disease X induce?"
- not_entity_query: Questions about descriptions, drug classes, clinical trial details, or relationship explanations that don't have entity-list answers

Important seed entity rules:
- Gene symbols are UPPERCASE: MET, TERT, EGFR, ALK, BRAF, etc.
- Mutations keep exact format: EGFR-p.L861R, PTEN-p.R173C, BRAF-p.T599K
- Drug names are lowercase: doxorubicin, pomalidomide, capecitabine
- Drug aliases keep original case: lonsurf, tukysa, meccnu
- Cancer names are lowercase: breast cancer, gastric cancer, colon cancer
- Disease names as-is: Li-Fraumeni syndrome, Bloom syndrome
- For "drugs treat cancers with X mutations" -> gene_inhibited_by_drug, seed = gene symbol (X)
- For "How to treat cancer with GENE?" -> treatment_for_cancercell, seed = GENE, secondary = cancer
- For resistance/sensitivity with "GENE in CANCER" -> seed = GENE, secondary = CANCER

Question: {question}
Answer: {answer}"""


def build_classifier():
    """Create LLM classifier"""
    from langchain.chat_models import init_chat_model

    if not os.getenv("OPENAI_API_KEY"):
        from core.config import get_settings
        settings = get_settings()
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
    return llm.with_structured_output(QueryClassification)


class KGQuerier:
    """Execute Cypher queries against Neo4j"""

    def __init__(self, graph: Graph):
        self.graph = graph
        self._build_name_cache()

    def _build_name_cache(self):
        """Cache all entity names for seed matching"""
        result = self.graph.run("""
            MATCH (n)
            WHERE n.name IS NOT NULL AND NOT 'RelationEmbedding' IN labels(n)
            RETURN DISTINCT n.name AS name, labels(n) AS labels
        """).data()

        self.name_map = {}  # lowercase -> (original_name, label)
        for r in result:
            name = r["name"]
            labels = [l for l in r["labels"] if l != "RelationEmbedding"]
            if labels:
                key = name.lower()
                if key not in self.name_map:
                    self.name_map[key] = (name, labels[0])

        print(f"Cached {len(self.name_map)} entity names")

    def find_seed(self, seed_name: str) -> Optional[Tuple[str, str]]:
        """Find seed entity in KG. Returns (name, label) or None."""
        key = seed_name.lower().strip()

        # Exact match
        if key in self.name_map:
            return self.name_map[key]

        # Try with common suffixes for cancer names
        if not any(t in key for t in ['cancer', 'carcinoma', 'sarcoma', 'lymphoma',
                                       'leukemia', 'melanoma', 'glioma', 'tumor']):
            for suffix in [' cancer', ' tumor']:
                trial = key + suffix
                if trial in self.name_map:
                    return self.name_map[trial]

        # Partial match: check if seed is contained in any KG name
        candidates = []
        for kg_key, (name, label) in self.name_map.items():
            if key in kg_key:
                candidates.append((name, label, len(kg_key) - len(key)))
        if candidates:
            candidates.sort(key=lambda x: x[2])
            return (candidates[0][0], candidates[0][1])

        return None

    def query(self, pattern_name: str, seed: str, seed_secondary: str = None) -> List[dict]:
        """Execute a query pattern and return answers."""
        pattern = QUERY_PATTERNS.get(pattern_name)
        if not pattern:
            return []

        seed_lower = seed.lower().strip()

        # For CancerCell patterns, construct search term
        if pattern["seed_type"] == "CancerCell" and seed_secondary:
            # CancerCell names contain gene+cancer, search by gene name
            seed_lower = seed_lower

        try:
            result = self.graph.run(pattern["cypher"], seed=seed_lower).data()
        except Exception as e:
            print(f"  Cypher error for {pattern_name}/{seed}: {e}")
            return []

        answers = []
        seen = set()
        for r in result:
            name = r.get("name", "")
            if not name or _has_chinese(name):
                continue
            name_lower = name.lower()
            if name_lower not in seen:
                seen.add(name_lower)
                answers.append({"name": name_lower})

        return answers


def main():
    import sys
    reuse_classifications = "--reuse" in sys.argv

    # Connect to Neo4j
    print("Connecting to Neo4j (pcqa)...")
    graph = Graph("bolt://neo4j_pcqa:7687", auth=("neo4j", "password"))

    # Build querier
    print("Building KG querier...")
    querier = KGQuerier(graph)

    # Load original data
    input_path = Path("/app/data/pcqa/PcQA.json")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from PcQA.json")

    # Classify questions
    classifications_path = Path("/app/data/pcqa/qa/classifications.jsonl")

    if reuse_classifications and classifications_path.exists():
        print("\n=== Reusing previous classifications (--reuse) ===")
        classifications = []
        with open(classifications_path) as f:
            for line in f:
                item = json.loads(line)
                classifications.append(QueryClassification(
                    query_pattern=item["query_pattern"],
                    seed_entity=item["seed_entity"],
                    seed_entity_secondary=item.get("seed_entity_secondary"),
                ))
        print(f"  Loaded {len(classifications)} classifications")
    else:
        print("\n=== Classifying questions with LLM ===")
        classifier = build_classifier()
        classifications = []

        for i, sample in enumerate(data):
            if (i + 1) % 20 == 1:
                print(f"  Processing {i+1}/{len(data)}...")

            prompt = CLASSIFICATION_PROMPT.format(
                question=sample["question"],
                answer=sample["answer"],
            )

            try:
                result = classifier.invoke(prompt)
                classifications.append(result)
            except Exception as e:
                print(f"  Error on sample {i+1}: {e}")
                classifications.append(QueryClassification(
                    query_pattern="not_entity_query",
                    seed_entity="",
                ))

            if (i + 1) % 50 == 0:
                time.sleep(1)

        # Save classifications
        with open(classifications_path, "w") as f:
            for c in classifications:
                f.write(json.dumps({
                    "query_pattern": c.query_pattern,
                    "seed_entity": c.seed_entity,
                    "seed_entity_secondary": c.seed_entity_secondary,
                }, ensure_ascii=False) + "\n")
        print(f"  Saved classifications to {classifications_path}")

    # Execute Cypher queries
    print("\n=== Cypher Query Execution ===")
    stats = {
        "total": len(data),
        "not_entity_query": 0,
        "seed_not_found": 0,
        "no_answers": 0,
        "with_answers": 0,
    }
    pattern_stats = {}

    converted = []

    for i, (sample, clf) in enumerate(zip(data, classifications)):
        pattern_name = clf.query_pattern
        pattern_stats[pattern_name] = pattern_stats.get(pattern_name, 0) + 1

        if pattern_name == "not_entity_query":
            stats["not_entity_query"] += 1
            continue

        if pattern_name not in QUERY_PATTERNS:
            stats["not_entity_query"] += 1
            continue

        pattern = QUERY_PATTERNS[pattern_name]

        # Find seed entity in KG
        seed_result = querier.find_seed(clf.seed_entity)
        if not seed_result:
            stats["seed_not_found"] += 1
            continue

        seed_name, seed_label = seed_result

        # Execute Cypher query
        answers = querier.query(pattern_name, clf.seed_entity, clf.seed_entity_secondary)

        if not answers:
            stats["no_answers"] += 1
            continue

        stats["with_answers"] += 1

        # Build path description
        path_desc = pattern["path"]
        filters = pattern.get("filters", {})

        converted.append({
            "question": sample["question"],
            "entity": seed_name.lower(),
            "entity_type": seed_label,
            "relation": pattern_name,
            "path": path_desc,
            "filters": filters,
            "answers": answers,
            "original_index": i + 1,
        })

    # Save eval dataset
    output_eval = Path("/app/data/pcqa/qa/eval_v2.jsonl")
    with open(output_eval, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Total samples: {stats['total']}")
    print(f"Not entity query (skipped): {stats['not_entity_query']}")
    print(f"Seed not found: {stats['seed_not_found']}")
    print(f"No answers from Cypher: {stats['no_answers']}")
    print(f"With verified answers: {stats['with_answers']}")
    print(f"\nPattern distribution:")
    for p, count in sorted(pattern_stats.items(), key=lambda x: -x[1]):
        print(f"  {p}: {count}")
    print(f"\nOutput: {output_eval} ({len(converted)} samples)")
    print(f"Previous eval_cleaned.jsonl had 288 samples")

    # Show samples
    print("\n=== Sample Output ===")
    for item in converted[:8]:
        idx = item["original_index"]
        print(f"[{idx}] Q: {item['question'][:60]}...")
        print(f"     entity: {item['entity']} ({item['entity_type']})")
        print(f"     path:   {item['path']}")
        if item['filters']:
            print(f"     filter: {item['filters']}")
        ans = [a['name'] for a in item['answers'][:5]]
        print(f"     answers({len(item['answers'])}): {ans}")
        print()


if __name__ == "__main__":
    main()
