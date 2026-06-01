"""Resume only the worker-errored samples of gemma3:27b ETK-Full / MetaQA 1-hop.

Original run: scripts/etk_apf_pacir_3llm_20260512.sh
  --kg metaqa --pipeline extended_type_kopl --dataset 1hop
  --num-samples 1000 --random --seed 42 --workers 16 --per-sample-timeout 900
  --model ollama/gemma3:27b --api-base <LiteLLM>
  --reranker llm --max-correction-rounds 1 --anchor-probe --cypher-informed-rerank

32/1000 samples failed with timeout(900.0s) (a single stalled worker over a
contiguous idx block) — an infrastructure artifact, not model behaviour.
This re-runs ONLY those errored idx with identical config and splices the new
records back into the existing jsonl (schema-identical via save_pipeline_outputs).

Run inside the container:
  docker exec python-primekgqa-experiment python /app/scripts/resume_metaqa1hop_gemma27b.py
Optional: RESUME_API_BASE env to override the LiteLLM endpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

API_BASE = os.environ.get("RESUME_API_BASE", "http://100.96.246.39:4000/v1")
MODEL = "ollama/gemma3:27b"
DS = "1hop"
SEED = 42
NSAMP = 1000
JSONL = Path(
    "/app/result/etk_apf_pacir_3llm_20260512/gemma3_27b/"
    "extended_type_kopl/metaqa_1hop/extended_type_kopl_1hop.jsonl"
)

os.environ["LLM_MODEL"] = MODEL
os.environ["LLM_API_BASE"] = API_BASE

from pipeline.run_evaluation import DATASETS_BY_KG, PipelineRunner, load_dataset  # noqa: E402
from pipeline.common.eval_metrics import save_pipeline_outputs  # noqa: E402

DATASETS_METAQA = DATASETS_BY_KG["metaqa"]


def main() -> None:
    assert JSONL.exists(), f"missing {JSONL}"
    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == NSAMP, f"expected {NSAMP} rows, got {len(rows)}"

    err_idx = sorted(r["idx"] for r in rows if r.get("error"))
    print(f"[before] total={len(rows)} errors={len(err_idx)} idx={err_idx}")
    if not err_idx:
        print("no errored samples; nothing to do.")
        return

    # Deterministic identical sampling (random.seed(42); random.sample).
    samples = load_dataset(DS, NSAMP, True, DATASETS_METAQA, seed=SEED)
    assert len(samples) == NSAMP, f"sampling mismatch: {len(samples)}"
    ds_cfg = DATASETS_METAQA[DS]

    # Exact ETK-Full kwargs (verified against run_evaluation CLI mapping).
    pipeline_kwargs = {
        "kg_type": "metaqa",
        "model": MODEL,
        "reranker_type": "llm",
        "reranker_input_k": 10,
        "use_llm_cypher": False,
        "max_correction_rounds": 1,
        "cypher_informed_rerank": True,
        "anchor_probe": True,
    }
    runner = PipelineRunner("extended_type_kopl", pipeline_kwargs)

    new_outputs = []
    t0 = time.time()
    for n, idx in enumerate(err_idx, 1):
        s = samples[idx]
        print(f"[{n}/{len(err_idx)}] idx={idx} q={s.get('question','')[:70]!r}", flush=True)
        out = runner.run_sample(idx, s, ds_cfg)
        new_outputs.append(out)
        print(f"    -> err={out.error!r} pred_n={len(out.predicted_entities or [])}", flush=True)
    print(f"re-ran {len(new_outputs)} samples in {time.time()-t0:.0f}s")

    # Serialize new outputs via the SAME function that wrote the file -> identical schema.
    with tempfile.TemporaryDirectory() as td:
        save_pipeline_outputs(new_outputs, Path(td), "extended_type_kopl", DS)
        tmp_file = Path(td) / "extended_type_kopl_1hop.jsonl"
        new_by_idx = {
            json.loads(l)["idx"]: l
            for l in tmp_file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }

    missing = set(err_idx) - set(new_by_idx)
    assert not missing, f"re-run missing idx: {sorted(missing)}"

    backup = JSONL.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(JSONL, backup)
        print(f"backup -> {backup}")

    merged = []
    for r in rows:
        i = r["idx"]
        merged.append(new_by_idx[i] if i in new_by_idx else json.dumps(r, ensure_ascii=False))
    JSONL.write_text("\n".join(merged) + "\n", encoding="utf-8")

    after = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    after_err = sorted(x["idx"] for x in after if x.get("error"))
    acc = sum(
        1.0
        for x in after
        if {str(g).strip().lower() for g in (x.get("gold_answers") or []) if g}
        == {str(p).strip().lower() for p in (x.get("predicted_entities") or []) if p}
    ) / len(after)
    print(f"[after] total={len(after)} errors={len(after_err)} idx={after_err}")
    print(f"[after] exact-set accuracy = {100*acc:.1f}%  (was 96.4% with 32 worker errors)")
    print("done. Re-run scripts/aggregate_all_20260515.py to refresh tables.")


if __name__ == "__main__":
    main()
