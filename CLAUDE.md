# CLAUDE.md - Project Context for Claude Code

## Project Overview
Knowledge Graph Question Answering (KGQA) system with multiple pipeline implementations.

## Datasets
- **PrimeKGQA**: Medical/biomedical knowledge graph QA
  - Query types: one_hop, two_hop, two_intersection, three_intersection
- **MetaQA**: Movie knowledge graph QA
  - Query types: 1-hop, 2-hop, 3-hop

## Pipelines

### Extended Type-KoPL (`app/pipeline/extended_type_kopl/`)
LLM-based relation selection with entity type constraints.
- Uses vector search for initial candidates
- LLM reranker for relation selection
- Type-based answer pruning

### SAFE (`app/pipeline/safe/`)
Self-Ask based factual extraction pipeline.

### KGT (`app/pipeline/kgt/`)
Knowledge Graph Traversal pipeline.

## Running Evaluations
```bash
docker exec -it python-primekgqa-experiment python -m app.pipeline.run_evaluation \
  --kg primekgqa \
  --pipeline extended_type_kopl \
  --datasets one_hop two_hop \
  --n 100 --random
```

## Key Files
- `app/pipeline/run_evaluation.py` - Evaluation runner
- `app/database/search.py` - Vector and graph search functions
- `docker-compose.yaml` - Service definitions

## Current Performance (n=100, random)

### PrimeKGQA
| Dataset | Extended Type-KoPL |
|---------|-------------------|
| one_hop | 94% |
| two_hop | 78% |
| two_intersection | 96% |

### MetaQA
| Dataset | Extended Type-KoPL | SAFE | KGT |
|---------|-------------------|------|-----|
| 1-hop | 100% | 100% | 90% |
| 2-hop | 100% | 90% | 58% |
| 3-hop | 96% | - | 7% |

> Note: Results measured with `--n 100 --random`. Reproduce via:
> ```
> docker exec -it python-primekgqa-experiment python -m app.pipeline.run_evaluation --kg primekgqa --pipeline extended_type_kopl --datasets one_hop two_hop two_intersection --n 100 --random
> docker exec -it python-primekgqa-experiment python -m app.pipeline.run_evaluation --kg metaqa --pipeline extended_type_kopl --datasets 1-hop 2-hop 3-hop --n 100 --random
> docker exec -it python-primekgqa-experiment python -m app.pipeline.run_evaluation --kg metaqa --pipeline safe --datasets 1-hop 2-hop --n 100 --random
> docker exec -it python-primekgqa-experiment python -m app.pipeline.run_evaluation --kg metaqa --pipeline kgt --datasets 1-hop 2-hop 3-hop --n 100 --random
> ```

## Research Agents
See `.claude/agents/` for:
- `research-agent.md` - Paper exploration
- `improvement-ideas.md` - Improvement suggestions

## Commands
See `.claude/commands/` for available commands.
