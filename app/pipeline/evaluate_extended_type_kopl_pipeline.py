"""
Extended Type-KoPL Pipeline 評価スクリプト

使用方法:
  python pipeline/evaluate_extended_type_kopl_pipeline.py --num-samples 20
  python pipeline/evaluate_extended_type_kopl_pipeline.py --dataset one_hop two_hop --num-samples 50

出力:
  - result/pipeline_outputs/ ディレクトリにJSONL形式で結果を保存
  - 精度算出は evaluate_all_results.py で一括して行う
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from pipeline.extended_type_kopl_pipeline import ExtendedTypeKoPLPipeline, ExtendedTypeKoPLResult
from pipeline.eval_metrics import PipelineOutput, save_pipeline_outputs


# データセット設定
DATASETS_V2 = {
    "one_hop": {
        "path": "result/dataset_v2/one_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answer_nodes",
    },
    "two_hop": {
        "path": "result/dataset_v2/two_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["rel1", "rel2"],
        "gold_answers_key": "answer_nodes",
    },
    "two_intersection": {
        "path": "result/dataset_v2/two_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel"],
        "gold_answers_key": "answer_nodes",
        "extra_entity_key": "anchor_b_name",
    },
    "three_intersection": {
        "path": "result/dataset_v2/three_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel", "anchor_c_rel"],
        "gold_answers_key": "answer_nodes",
        "extra_entity_keys": ["anchor_b_name", "anchor_c_name"],
    },
}


def extract_relations_from_paths(selected_paths) -> Optional[List[str]]:
    """選択されたパスからリレーションを抽出"""
    if not selected_paths:
        return None
    # 最もスコアの高いパスのリレーションを使用
    best_path = selected_paths[0]
    return list(best_path.relations) if best_path.relations else None


def run_sample(
    idx: int,
    sample: Dict[str, Any],
    pipeline: ExtendedTypeKoPLPipeline,
    dataset_config: Dict[str, Any],
) -> PipelineOutput:
    """単一サンプルを実行"""
    question = sample.get("question", "")
    entity_name = sample.get(dataset_config["entity_key"], "")

    # 正解データ
    answers_key = dataset_config.get("gold_answers_key", "answer_nodes")
    gold_answers = [node["name"] for node in sample.get(answers_key, [])]
    gold_relations = [
        sample.get(k, "") for k in dataset_config["gold_relations_keys"]
        if sample.get(k)
    ]

    output = PipelineOutput(
        idx=idx,
        question=question,
        entity_name=entity_name,
        gold_answers=gold_answers,
        gold_relations=gold_relations,
        predicted_relations=None,
        predicted_entities=[],
    )

    try:
        start = time.time()
        pipeline_result = pipeline.run(question=question, entity_name=entity_name)
        output.latency_ms = (time.time() - start) * 1000

        # 予測リレーション
        output.predicted_relations = extract_relations_from_paths(pipeline_result.selected_paths)

        # 予測エンティティ
        output.predicted_entities = pipeline_result.answer_entities

    except Exception as e:
        output.error = str(e)

    return output


def run_pipeline(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    *,
    num_samples: int = 50,
) -> List[PipelineOutput]:
    """データセットに対してパイプラインを実行"""

    data_path = Path(dataset_config["path"])
    if not data_path.exists():
        print(f"  [SKIP] File not found: {data_path}")
        return []

    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"  Loaded {len(samples)} samples")

    # パイプライン作成
    pipeline = ExtendedTypeKoPLPipeline()

    outputs: List[PipelineOutput] = []

    # 逐次処理
    for i, sample in enumerate(tqdm(samples, desc=f"  {dataset_name}")):
        output = run_sample(i, sample, pipeline, dataset_config)
        outputs.append(output)

    return outputs


def main():
    p = argparse.ArgumentParser(description="Run Extended Type-KoPL Pipeline")
    p.add_argument("--dataset", type=str, nargs="+",
                   choices=list(DATASETS_V2.keys()),
                   help="Specific dataset(s) to run (default: all)")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--output-dir", type=Path, default=Path("result/pipeline_outputs"),
                   help="Output directory for pipeline results")
    args = p.parse_args()

    datasets_to_run = args.dataset if args.dataset else list(DATASETS_V2.keys())

    for ds_name in datasets_to_run:
        if ds_name not in DATASETS_V2:
            print(f"Unknown dataset: {ds_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Running Extended Type-KoPL: {ds_name}")
        print(f"{'='*60}")

        ds_config = DATASETS_V2[ds_name]
        outputs = run_pipeline(
            ds_name,
            ds_config,
            num_samples=args.num_samples,
        )

        if outputs:
            save_pipeline_outputs(
                outputs,
                args.output_dir,
                "extended_type_kopl",
                ds_name,
            )
            errors = sum(1 for o in outputs if o.error)
            print(f"  Saved {len(outputs)} outputs ({errors} errors)")
            print(f"  -> {args.output_dir}/extended_type_kopl_{ds_name}.jsonl")


if __name__ == "__main__":
    main()
