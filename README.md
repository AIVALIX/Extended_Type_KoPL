# Extended Type-KoPL (ETK) — APF + PA-CIR

Reference implementation for the paper **"A KG-Structure Feedback Agent for Extended Type-KoPL: Anchor-Edge Feedback and Cypher-Trial Reranking."**

This codebase implements ETK with two new mechanisms:
- **APF (Anchor-Probe Feedback):** injects anchor-specific real-edge information into the Phase 4 → 1 correction prompt to reduce *empty-query failures*.
- **PA-CIR (Probe-Aware Cypher-Informed Reranking):** augments the Phase 3b reranker with lightweight Cypher-trial results to reduce *misselection of unreachable candidates*.

It also includes same-condition reimplementations of **KGT** and **SAFE** used for the comparison in the paper.

## Repository Layout

```
app/
  pipeline/
    extended_type_kopl/    # ETK + APF + PA-CIR
    kgt/                   # Same-condition KGT reimplementation
    safe/                  # Same-condition SAFE reimplementation
    run_evaluation.py      # Unified evaluation entry point
    run_ablation.py        # Ablation driver
  database/
    seed.py                # KG seeding (Neo4j import)
  ...
config/.env                # Runtime configuration (you create this)
docker-compose.yaml        # All services: one app container + five Neo4j
```

## 1. Prerequisites

- Docker and Docker Compose (Compose v2 plugin: `docker compose ...`)
- ~32 GB RAM recommended (five Neo4j 5.22 instances run concurrently)
- An OpenAI API key (for cloud LLM evaluation), and/or a LiteLLM proxy endpoint (for local LLM evaluation)

## 2. Environment Variables

Create `config/.env` with the following keys.

```env
# --- Neo4j credentials (must match docker-compose) ---
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# --- Default Neo4j URI used by the app container ---
# Per-KG URI is selected automatically from the --kg flag at run time.
NEO4J_URI=bolt://neo4j:7687

# --- LLM backends ---
# Required for OpenAI / gpt-4.1-mini runs:
OPENAI_API_KEY=sk-...

# Optional: LiteLLM proxy for local LLMs (gemma3:12B / gemma3:27B / llama3.1:8B)
LITELLM_BASE_URL=http://127.0.0.1:4000/v1
LITELLM_API_KEY=any-string

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=etk-apf-pacir
```

Per-KG Neo4j endpoints inside the Docker network are:

| KG | Container | Internal URI | Host port |
|---|---|---|---|
| PrimeKGQA | `neo4j_primekgqa` | `bolt://neo4j:7687` | `7687` |
| MetaQA | `neo4j_metaqa` | `bolt://neo4j_metaqa:7687` | `7688` |
| PcQA | `neo4j_pcqa` | `bolt://neo4j_pcqa:7687` | `7689` |
| WebQSP | `neo4j_webqsp` | `bolt://neo4j_webqsp:7687` | `7690` |
| KQA Pro | `neo4j_kqapro` | `bolt://neo4j_kqapro:7687` | `7691` |

The application code picks the correct URI automatically from the `--kg` flag.

## 3. Bringing Up the Stack

```bash
# Build and start all services in the background.
docker compose up -d --build

# Wait until every Neo4j container reports healthy (~30 s after first boot).
docker compose ps
```

The application container is named `python-primekgqa-experiment` (historical name; it hosts every KG, not only PrimeKG).

## 4. Loading the Knowledge Graphs

Each KG ships with `data/<kg>/import_nodes.csv` and `data/<kg>/import_rels.csv`. Use the unified seeder inside the app container:

```bash
docker exec -it python-primekgqa-experiment \
    python -m database.seed --kg metaqa primekgqa pcqa
```

Available `--kg` choices: `metaqa`, `primekgqa`, `pcqa`, `webqsp`, `kqapro`.
Pass multiple values to seed several KGs in one call.

If you need to overwrite an existing Neo4j database from raw CSVs:

```bash
docker compose run --rm --entrypoint="" neo4j_metaqa \
    neo4j-admin database import full \
        --nodes=/import/import_nodes.csv \
        --relationships=/import/import_rels.csv \
        --delimiter=, --array-delimiter=";" \
        --overwrite-destination=true neo4j
```

PcQA additionally requires building the entity-set gold (see Section 4.1 of the paper). Run:

```bash
docker exec -it python-primekgqa-experiment \
    python data/pcqa/build_eval_cypher.py
```

### 4.1 PrimeKGQA-Struct

The PrimeKG knowledge graph is too large to redistribute through git. Only the
Neo4j node import file (`data/primekgqa/import_nodes.csv`) is shipped. To
build the relationships file and PrimeKGQA-Struct QA splits:

```bash
# 1. Obtain the original PrimeKGQA dataset (Yan et al., ECAI 2024) and
#    drop the raw JSON files into data/primekgqa_original/.
#
#    Required files:
#      data/primekgqa_original/test_call_bioLLM.json
#      data/primekgqa_original/final_test.json    (optional)
#      data/primekgqa_original/final_val.json     (optional)

# 2. Convert to ETK's evaluation JSONL format (entity-set questions).
docker exec -it python-primekgqa-experiment \
    python -m dataset_construction.convert_primekgqa_original

# 3. Build the import_rels.csv from PrimeKG.
#    Obtain PrimeKG (https://primekg.helmholtz-muenchen.de/ or the Harvard
#    repository) and place its node / edge files under data/primekgqa/,
#    then run the Neo4j import as in the previous section.
```

## 5. Running the Comparison Experiments

All evaluation entry points are wrapped by `app/pipeline/run_evaluation.py`. The general form is:

```bash
docker exec -it python-primekgqa-experiment \
    python -m pipeline.run_evaluation \
        --kg <kg> \
        --pipeline <pipeline> \
        --dataset <subset> \
        --num-samples <N> --random --seed 42 \
        --workers <W> --per-sample-timeout 900 \
        --model <model-id> [--api-base <litellm-url>] \
        [pipeline-specific flags] \
        --output-dir result/<run-tag>
```

Pipeline names: `extended_type_kopl`, `safe`, `kgt`.

### 5.1 Best configuration (ETK + APF + PA-CIR)

```bash
docker exec -it python-primekgqa-experiment \
    python -m pipeline.run_evaluation \
        --kg metaqa --pipeline extended_type_kopl \
        --dataset 1hop 2hop 3hop \
        --num-samples 1000 --random --seed 42 \
        --workers 16 --per-sample-timeout 900 \
        --model gpt-4.1-mini \
        --reranker llm --max-correction-rounds 1 \
        --anchor-probe --cypher-informed-rerank \
        --output-dir result/etk_full_gpt41mini/metaqa
```

Subset names:
- MetaQA: `1hop`, `2hop`, `3hop`
- PrimeKGQA: `one_hop`, `two_hop`, `two_intersection`, `three_intersection`
- PcQA: `all`

### 5.2 Ablation (baseline / +APF / +PA-CIR / Full)

The four ablation configurations correspond to:

| Config | Flags added on top of `--reranker llm --max-correction-rounds 1` |
|---|---|
| baseline | (none) |
| +PA-CIR | `--anchor-probe --cypher-informed-rerank --no-apf-in-correction` |
| +APF    | `--anchor-probe` |
| Full    | `--anchor-probe --cypher-informed-rerank` |

Run each configuration as a separate invocation of `run_evaluation` with the matching flags, varying `--output-dir` per configuration.

### 5.3 Schema-driven baselines (KGT, SAFE)

```bash
# KGT
docker exec -it python-primekgqa-experiment \
    python -m pipeline.run_evaluation \
        --kg primekgqa --pipeline kgt \
        --dataset one_hop two_hop two_intersection three_intersection \
        --num-samples 500 --random --seed 42 \
        --workers 4 --per-sample-timeout 900 \
        --model gpt-4.1-mini \
        --output-dir result/kgt_gpt41mini/primekgqa

# SAFE
docker exec -it python-primekgqa-experiment \
    python -m pipeline.run_evaluation \
        --kg primekgqa --pipeline safe \
        --dataset one_hop two_hop \
        --num-samples 500 --random --seed 42 \
        --workers 4 --per-sample-timeout 900 \
        --model gpt-4.1-mini \
        --output-dir result/safe_gpt41mini/primekgqa
```

INTERSECTION subsets are not run for SAFE because the method does not structurally support them (left blank in Table 1).

### 5.4 Local LLM evaluation (gemma3, llama3.1)

Local LLMs are served through a LiteLLM proxy. Set `LITELLM_BASE_URL` in `.env`, then pass `--api-base` and `--model` accordingly:

```bash
--model ollama/gemma3:27b --api-base ${LITELLM_BASE_URL}
--model ollama/gemma3:12b --api-base ${LITELLM_BASE_URL}
--model ollama/llama3.1:8b --api-base ${LITELLM_BASE_URL}
```

For PrimeKG runs use `--workers 4` (the larger graph plus APF correction is unstable at higher concurrency).

## 6. Output Format

Per-run outputs land in `result/<run-tag>/<pipeline>/<kg>_<subset>/extended_type_kopl_<subset>.jsonl` (or `<pipeline>_<subset>.jsonl`). Each JSONL line contains the question, gold answer set, predicted answer set, per-sample timing, and the pipeline's intermediate state (Type-KoPL, selected path, Cypher result). Aggregation into per-subset F1 / Accuracy is left to the reader; the paper reports set-based F1.

## Citation

```bibtex
@inproceedings{oshima2026etk_apf_pacir,
    author    = {Oshima, Kotaro and Takahashi, Yoichi},
    title     = {A KG-Structure Feedback Agent for Extended Type-KoPL:
                 Anchor-Edge Feedback and Cypher-Trial Reranking},
    booktitle = {Proceedings of the International Joint Conference on Knowledge Graphs (IJCKG)},
    year      = {2026}
}
```

## License

Released for research reproduction purposes. See the manuscript for the disclosure of interests.
