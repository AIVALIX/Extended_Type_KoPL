#!/bin/bash
# 3 KG × 5 configs (A/B/C/E/F) ablation with gemma3:27b via LiteLLM proxy.
# Skips D/G (multi-candidate) to avoid proxy saturation; they are already
# known to be neutral/negative from the gpt-4.1-mini sweep.
#
# Estimated: ~8 hours sequential.

LITELLM="http://100.96.246.39:4000/v1"
MODEL="ollama/gemma3:27b"
CONTAINER_OUT="/app/result/ablation_20260415/gemma3_27b"
HOST_LOG="/Users/ooshimakotarou/indigo_primekgqa/Indigo_sigswo_68/result/ablation_20260415/gemma3_27b"
mkdir -p "$HOST_LOG"

run_one() {
  local kg="$1"; shift
  local datasets="$1"; shift
  local name="$1"; shift
  echo "== $kg $name =="
  date
  docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
    --kg "$kg" --pipeline extended_type_kopl \
    --dataset $datasets --num-samples 100 --random --seed 42 --workers 4 \
    --model "$MODEL" --api-base "$LITELLM" \
    --output-dir "$CONTAINER_OUT/$kg/$name" \
    "$@" \
    > "$HOST_LOG/${kg}_${name}.log" 2>&1
  echo "  exit $? at $(date)"
  tr '\r' '\n' < "$HOST_LOG/${kg}_${name}.log" | grep -E "Accuracy|F1:|Errors" | tail -8
  echo ""
}

run_kg() {
  local kg="$1"; shift
  local datasets="$1"; shift
  run_one "$kg" "$datasets" A_vanilla --no-anchor-reorient --max-correction-rounds 0 --reranker llm
  run_one "$kg" "$datasets" B_fixa --max-correction-rounds 0 --reranker llm
  run_one "$kg" "$datasets" C_baseline --max-correction-rounds 1 --reranker llm
  run_one "$kg" "$datasets" E_noreranker --max-correction-rounds 1 --reranker none
  run_one "$kg" "$datasets" F_cir --max-correction-rounds 1 --reranker llm --cypher-informed-rerank
}

run_kg metaqa "1hop 2hop 3hop"
run_kg primekgqa "one_hop two_hop two_intersection three_intersection"
run_kg pcqa "all"

echo "== ALL DONE at $(date) =="
