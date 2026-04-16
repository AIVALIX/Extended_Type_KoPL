#!/bin/bash
# Re-run MetaQA ablation A-G on gpt-4.1-mini after all refactors.
CONTAINER_OUT="/app/result/ablation_20260415/gpt41mini/metaqa"
HOST_LOG="/Users/ooshimakotarou/indigo_primekgqa/Indigo_sigswo_68/result/ablation_20260415"
mkdir -p "$HOST_LOG"
COMMON="--kg metaqa --pipeline extended_type_kopl --dataset 1hop 2hop 3hop --num-samples 100 --random --seed 42 --workers 4 --model gpt-4.1-mini"

run_cfg() {
  local name="$1"; shift
  echo "== MetaQA $name =="
  docker exec python-primekgqa-experiment python -m pipeline.run_evaluation $COMMON \
    --output-dir "$CONTAINER_OUT/${name}" "$@" \
    > "$HOST_LOG/metaqa_${name}.log" 2>&1
  echo "  exit $?"
  tr '\r' '\n' < "$HOST_LOG/metaqa_${name}.log" | grep -E "Accuracy|F1:" | tail -6
}

run_cfg A_vanilla       --no-anchor-reorient --max-correction-rounds 0 --reranker llm
run_cfg B_fixa          --max-correction-rounds 0 --reranker llm
run_cfg C_baseline      --max-correction-rounds 1 --reranker llm
run_cfg D_nkopl3_plain  --max-correction-rounds 1 --reranker llm --n-kopl-candidates 3 --plain-scoring
run_cfg E_noreranker    --max-correction-rounds 1 --reranker none
run_cfg F_cir           --max-correction-rounds 1 --reranker llm --cypher-informed-rerank
run_cfg G_nkopl3_enh    --max-correction-rounds 1 --reranker llm --n-kopl-candidates 3
echo "== MetaQA DONE =="
