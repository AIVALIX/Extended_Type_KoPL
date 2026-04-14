#!/bin/bash
# Re-run PcQA ablation on eval_v2.jsonl. Host-side log path, container-side output-dir.
CONTAINER_OUT="/app/result/ablation/gpt41mini"
HOST_LOG="/Users/ooshimakotarou/indigo_primekgqa/Indigo_sigswo_68/result/ablation"
COMMON="--kg pcqa --pipeline extended_type_kopl --dataset all --num-samples 100 --random --seed 42 --workers 4 --model gpt-4.1-mini"

run_cfg() {
  local name="$1"; shift
  echo "=============================================="
  echo "== Running: $name"
  echo "=============================================="
  docker exec python-primekgqa-experiment python -m pipeline.run_evaluation $COMMON \
    --output-dir "$CONTAINER_OUT/pcqa_${name}_v2" "$@" > "$HOST_LOG/gpt41mini_pcqa_${name}_v2.log" 2>&1
  local rc=$?
  echo "  -> exit $rc"
  grep -E "Accuracy|F1|Errors" "$HOST_LOG/gpt41mini_pcqa_${name}_v2.log" | tail -5 || true
  return 0
}

run_cfg B_fixa         --max-correction-rounds 0 --reranker llm
run_cfg D_nkopl3_plain --max-correction-rounds 1 --reranker llm --n-kopl-candidates 3 --plain-scoring
run_cfg E_noreranker   --max-correction-rounds 1 --reranker none
run_cfg F_cir          --max-correction-rounds 1 --reranker llm --cypher-informed-rerank
run_cfg G_nkopl3_enh   --max-correction-rounds 1 --reranker llm --n-kopl-candidates 3
echo "== ALL DONE =="
