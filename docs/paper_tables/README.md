# Paper Tables (IJCKG 2026 submission)

LaTeX-ready tables for the ETK paper. Each file stands alone and can be
`\input{}`-ed. Numbers are derived from `docs/ablation_report_20260414.md`.

| File | Content | Notes |
|---|---|---|
| `main_results.tex` | C baseline accuracy on 3 KGs × 2 LLMs | Top-level comparison, 2 rows |
| `ablation_gpt41mini.tex` | Full 7-config ablation with gpt-4.1-mini | Main ablation table |
| `ablation_gemma3.tex` | 5-config ablation with local gemma3:27b | D/G omitted (proxy saturation); errors annotated |
| `contribution_matrix.tex` | $\Delta$ per component × LLM × KG subset | Compact summary for "what each component buys" discussion |

## Provenance

All numbers are from runs under:

- `n=100` random sampled, `seed=42`, `workers=4`
- `--reranker llm`, `--max-correction-rounds 1` unless otherwise stated
- `gpt-4.1-mini`: re-measured 2026-04-15 after the capability-driven
  refactor (commit `9231c4c`) and the Cypher-escape fix (commit `bf7e31e`)
- `gemma3:27b`: measured 2026-04-15/16 overnight via LiteLLM proxy at
  `http://100.96.246.39:4000/v1`

Raw per-run logs: `result/ablation_20260415/`.
Run scripts: `scripts/metaqa_ablation_gpt41mini.sh`,
`scripts/primekgqa_ablation_gpt41mini.sh`,
`scripts/ablation_gemma3_27b.sh`.
