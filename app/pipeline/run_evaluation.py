"""
統一評価パイプライン

データセット入力から精度評価までを一括で行う
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation --kg primekgqa --num-samples 20 --random --workers 16 --pipeline safe --no-schema
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation --kg primekgqa --num-samples 500 --random --workers 4 --pipeline safe kgt etk --no-schema --reranker llm --output /app/result/result_em

  python pipeline/run_evaluation.py --pipeline safe --dataset one_hop two_hop

  # MetaQA
  python pipeline/run_evaluation.py --kg metaqa --pipeline extended_type_kopl --num-samples 20

  # 複数パイプライン（比較）
  python pipeline/run_evaluation.py --pipeline extended_type_kopl safe kgt --num-samples 20

  # 全パイプライン
  python pipeline/run_evaluation.py --all --num-samples 20

対応パイプライン:
  - extended_type_kopl: Extended Type-KoPL Pipeline
  - safe: SAFE Pipeline（one_hop, two_hopのみ対応）
  - kgt: KGT Pipeline

対応KG:
  - primekgqa: PrimeKGQA（バイオメディカル）
  - metaqa: MetaQA（映画ドメイン）


  # 1. MetaQA: SAFE + KGT + ETK (Rerankerなし)
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
  --kg metaqa --num-samples 400 --random --workers 4 --no-schema \
  --pipeline safe kgt extended_type_kopl \
  --output-dir /app/result/eval_em_metaqa \
  --output /app/result/eval_em_metaqa/summary.json

# 2. MetaQA: ETK+Reranker
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
  --kg metaqa --num-samples 400 --random --workers 32 --no-schema \
  --pipeline extended_type_kopl --reranker llm \
  --output-dir /app/result/eval_em_metaqa_reranker \
  --output /app/result/eval_em_metaqa_reranker/summary_2.json


python app/pipeline/run_repeated_eval.py \
  --runs 20 \
  --cmd "docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
    --kg metaqa --num-samples 400 --random --workers 16 --no-schema \
    --pipeline extended_type_kopl --reranker llm \
    --output-dir /app/result/eval_em_metaqa_reranker" \
  --output result/eval_em_metaqa_reranker/aggregated.json

# 3. PrimeKGQA paraphrase: SAFE + KGT + ETK
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
  --kg primekgqa --num-samples 500 --random --workers 4 --no-schema \
  --pipeline safe kgt extended_type_kopl \
  --output-dir /app/result/eval_em_primekgqa \
  --output /app/result/eval_em_primekgqa/summary.json

# 4. PrimeKGQA paraphrase: ETK+Reranker
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
  --kg primekgqa --num-samples 500 --random --workers 8 --no-schema \
  --pipeline extended_type_kopl --reranker llm \
  --output-dir /app/result/eval_em_primekgqa_reranker \
  --output /app/result/eval_em_primekgqa_reranker/summary.json

# 5. PrimeKGQA raw: SAFE + KGT + ETK
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
  --kg primekgqa_raw --num-samples 500 --random --workers 4 --no-schema \
  --pipeline safe kgt extended_type_kopl \
  --output-dir /app/result/eval_em_primekgqa_raw \
  --output /app/result/eval_em_primekgqa_raw/summary.json

# 6. PrimeKGQA raw: ETK+Reranker
docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
  --kg primekgqa_raw --num-samples 500 --random --workers 8 --no-schema \
  --pipeline extended_type_kopl --reranker llm \
  --output-dir /app/result/eval_em_primekgqa_raw_reranker \
  --output /app/result/eval_em_primekgqa_raw_reranker/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
import multiprocessing as mp

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from pipeline.common.eval_metrics import (
    PipelineOutput,
    EvalResult,
    NLEvalResult,
    evaluate_outputs,
    aggregate_metrics,
    aggregate_nl_metrics,
    compute_rouge_l,
    compute_embedding_similarity,
    save_pipeline_outputs,
    analyze_failures,
    save_failure_analysis,
    print_failure_summary,
)
from pipeline.common.kg_config import KGConfig, KGType


# データセット設定（PrimeKGQA - paraphrase版）
DATASETS_PRIMEKGQA = {
    "one_hop": {
        "path": "result/dataset_v4_paraphrase/one_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answer_nodes",
    },
    "two_hop": {
        "path": "result/dataset_v4_paraphrase/two_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["rel1", "rel2"],
        "gold_answers_key": "answer_nodes",
    },
    "two_intersection": {
        "path": "result/dataset_v4_paraphrase/two_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel"],
        "gold_answers_key": "answer_nodes",
        "extra_entity_key": "anchor_b_name",
    },
    "three_intersection": {
        "path": "result/dataset_v4_paraphrase/three_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel", "anchor_c_rel"],
        "gold_answers_key": "answer_nodes",
        "extra_entity_keys": ["anchor_b_name", "anchor_c_name"],
    },
}

# データセット設定（PrimeKGQA - no-paraphrase版）
DATASETS_PRIMEKGQA_RAW = {
    "one_hop": {
        "path": "result/dataset_v4/one_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answer_nodes",
    },
    "two_hop": {
        "path": "result/dataset_v4/two_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["rel1", "rel2"],
        "gold_answers_key": "answer_nodes",
    },
    "two_intersection": {
        "path": "result/dataset_v4/two_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel"],
        "gold_answers_key": "answer_nodes",
        "extra_entity_key": "anchor_b_name",
    },
    "three_intersection": {
        "path": "result/dataset_v4/three_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel", "anchor_c_rel"],
        "gold_answers_key": "answer_nodes",
        "extra_entity_keys": ["anchor_b_name", "anchor_c_name"],
    },
}

# データセット設定（MetaQA）
DATASETS_METAQA = {
    "1hop": {
        "path": "result/metaqa_v2/1hop.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answers",
    },
    "2hop": {
        "path": "result/metaqa_v2/2hop.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation1", "relation2"],
        "gold_answers_key": "answers",
    },
    "3hop": {
        "path": "result/metaqa_v2/3hop.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation1", "relation2", "relation3"],
        "gold_answers_key": "answers",
    },
}

# データセット設定（PcQA - Pan-cancer QA）
DATASETS_PCQA = {
    "all": {
        "path": "data/pcqa/qa/eval_v3.jsonl",  # Cypher-verified dataset with entity/path/filters (352 samples, 100% coverage)
        "entity_key": "entity",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answers",
    },
}

# データセット設定（WebQSP - Freebase subset）
DATASETS_WEBQSP = {
    "test": {
        "path": "data/webqsp/qa/test.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answer_nodes",
    },
}

# データセット設定（KQA Pro - Wikidata subset）
DATASETS_KQAPRO = {
    "val": {
        "path": "data/kqapro/qa/val_entity.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answers",
    },
    "val_all": {
        "path": "data/kqapro/qa/val_all.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answers",
    },
}

# データセット設定（PrimeKGQA Original — 本家ベンチマーク）
DATASETS_PRIMEKGQA_ORIGINAL = {
    "test_entity": {
        "path": "data/primekgqa_original/qa/test_entity.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answer_nodes",
    },
}

# KGごとのデータセット設定
DATASETS_BY_KG = {
    "primekgqa": DATASETS_PRIMEKGQA,
    "primekgqa_raw": DATASETS_PRIMEKGQA_RAW,
    "primekgqa_original": DATASETS_PRIMEKGQA_ORIGINAL,
    "metaqa": DATASETS_METAQA,
    "pcqa": DATASETS_PCQA,
    "webqsp": DATASETS_WEBQSP,
    "kqapro": DATASETS_KQAPRO,
}

# 後方互換性のため
DATASETS_V2 = DATASETS_PRIMEKGQA

# パイプライン設定（PrimeKGQA）
PIPELINE_CONFIGS_PRIMEKGQA = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["one_hop", "two_hop", "two_intersection", "three_intersection"],
    },
    "safe": {
        "name": "SAFE",
        "datasets": ["one_hop", "two_hop"],  # intersectionは非対応
    },
    "kgt": {
        "name": "KGT",
        "datasets": ["one_hop", "two_hop", "two_intersection", "three_intersection"],
    },
}

# パイプライン設定（MetaQA）
PIPELINE_CONFIGS_METAQA = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["1hop", "2hop", "3hop"],
    },
    "safe": {
        "name": "SAFE",
        "datasets": ["1hop", "2hop", "3hop"],
    },
    "kgt": {
        "name": "KGT",
        "datasets": ["1hop", "2hop", "3hop"],
    },
}

# パイプライン設定（PcQA）
PIPELINE_CONFIGS_PCQA = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["all"],
    },
    "safe": {
        "name": "SAFE",
        "datasets": ["all"],
    },
    "kgt": {
        "name": "KGT",
        "datasets": ["all"],
    },
}

# パイプライン設定（WebQSP）
PIPELINE_CONFIGS_WEBQSP = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["test"],
    },
    # WIP: SAFE/KGT require WebQSP schema definitions before producing valid results.
    # Added for fairness — all pipelines should be evaluable on all KGs.
    "safe": {
        "name": "SAFE",
        "datasets": ["test"],
    },
    "kgt": {
        "name": "KGT",
        "datasets": ["test"],
    },
}

# パイプライン設定（KQA Pro）
PIPELINE_CONFIGS_KQAPRO = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["val", "val_all"],
    },
}

# KGごとのパイプライン設定
PIPELINE_CONFIGS_BY_KG = {
    "primekgqa": PIPELINE_CONFIGS_PRIMEKGQA,
    "primekgqa_raw": PIPELINE_CONFIGS_PRIMEKGQA,  # 同じパイプライン設定を共有
    "primekgqa_original": {
        "extended_type_kopl": {
            "name": "Extended Type-KoPL",
            "datasets": ["test_entity"],
        },
    },
    "metaqa": PIPELINE_CONFIGS_METAQA,
    "pcqa": PIPELINE_CONFIGS_PCQA,
    "webqsp": PIPELINE_CONFIGS_WEBQSP,
    "kqapro": PIPELINE_CONFIGS_KQAPRO,
}

# 後方互換性のため
PIPELINE_CONFIGS = PIPELINE_CONFIGS_PRIMEKGQA

# データセット表示名
DATASET_NAMES = {
    # PrimeKGQA
    "one_hop": "1-hop",
    "two_hop": "2-hop",
    "two_intersection": "2-intersection",
    "three_intersection": "3-intersection",
    # MetaQA
    "1hop": "1-hop",
    "2hop": "2-hop",
    "3hop": "3-hop",
    # PcQA
    "all": "all",
    # WebQSP
    "test": "test",
}


class PipelineRunner:
    """パイプライン実行クラス"""

    def __init__(
        self, pipeline_id: str, pipeline_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_kwargs = pipeline_kwargs or {}
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """パイプラインを初期化"""
        if self.pipeline_id == "extended_type_kopl":
            from pipeline.extended_type_kopl import ExtendedTypeKoPLPipeline

            self.pipeline = ExtendedTypeKoPLPipeline(**self.pipeline_kwargs)
        elif self.pipeline_id == "safe":
            from pipeline.safe import SAFEPipeline

            self.pipeline = SAFEPipeline(**self.pipeline_kwargs)
        elif self.pipeline_id == "kgt":
            from pipeline.kgt import KGTPipeline

            self.pipeline = KGTPipeline(**self.pipeline_kwargs)
        else:
            raise ValueError(f"Unknown pipeline: {self.pipeline_id}")

    def _extract_relations(self, result) -> Optional[List[str]]:
        """パイプライン結果からリレーションを抽出"""
        if self.pipeline_id == "extended_type_kopl":
            if not result.selected_paths:
                return None
            best_path = result.selected_paths[0]
            return list(best_path.relations) if best_path.relations else None

        elif self.pipeline_id == "safe":
            if not result.best_query_graph:
                return None
            relations = []
            for schema_edge, _ in result.best_query_graph.edges:
                relations.append(schema_edge.relation)
            return relations if relations else None

        elif self.pipeline_id == "kgt":
            if result.optimal_path:
                return result.optimal_path.relations
            return None

        return None

    def _extract_entities(self, result) -> List[str]:
        """パイプライン結果からエンティティを抽出"""
        return result.answer_entities

    def run_sample(
        self,
        idx: int,
        sample: Dict[str, Any],
        dataset_config: Dict[str, Any],
    ) -> PipelineOutput:
        """単一サンプルを実行"""
        question = sample.get("question", "")
        entity_name = sample.get(dataset_config["entity_key"], "")

        # 正解データ
        answers_key = dataset_config.get("gold_answers_key", "answer_nodes")
        raw_answers = sample.get(answers_key, [])
        if raw_answers and isinstance(raw_answers[0], dict):
            gold_answers = [node["name"] for node in raw_answers]
        else:
            gold_answers = list(raw_answers)
        gold_relations = [
            sample.get(k, "")
            for k in dataset_config["gold_relations_keys"]
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
            assert self.pipeline is not None
            run_kwargs = {}
            if (
                hasattr(self.pipeline, "run")
                and "n_gold" in self.pipeline.run.__code__.co_varnames
            ):
                run_kwargs["n_gold"] = len(gold_answers)
            result = self.pipeline.run(
                question=question, entity_name=entity_name, **run_kwargs
            )
            output.latency_ms = (time.time() - start) * 1000

            output.predicted_relations = self._extract_relations(result)
            output.predicted_entities = self._extract_entities(result)

            # For extended answer types (count, attr, verify, select, relation),
            # the answer is in natural_answer, not answer_entities
            if (
                hasattr(result, "natural_answer")
                and result.natural_answer
                and hasattr(result, "kopl_program")
                and result.kopl_program
                and getattr(result.kopl_program, "answer_type", "entity") != "entity"
            ):
                output.predicted_entities = [result.natural_answer]

            # ステップ詳細を保存
            if hasattr(result, "processing_log"):
                output.processing_log = result.processing_log
            if hasattr(result, "kopl_program") and result.kopl_program:
                kp = result.kopl_program
                output.kopl_program = {
                    "op_type": kp.op_type.value if hasattr(kp.op_type, "value") else str(kp.op_type),
                    "relations": [
                        {"src": r.src_type, "rel": r.relation_hint, "tgt": r.tgt_type}
                        for r in kp.relations
                    ],
                    "anchor": kp.anchor_name,
                    "answer_type": getattr(kp, "answer_type", "entity"),
                }
            if hasattr(result, "candidate_paths"):
                output.candidate_paths = [p.to_text() for p in result.candidate_paths[:20]]
            if hasattr(result, "selected_paths"):
                output.selected_paths = [p.to_text() for p in result.selected_paths]

        except Exception as e:
            output.error = str(e)

        return output


# グローバル変数（ProcessPoolExecutor用）
_worker_runner: Optional[PipelineRunner] = None


def _init_worker(pipeline_id: str, pipeline_kwargs: Optional[Dict[str, Any]] = None):
    """ワーカープロセス初期化"""
    import warnings

    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    global _worker_runner
    _worker_runner = PipelineRunner(pipeline_id, pipeline_kwargs)


def _process_sample_worker(
    args: Tuple[int, Dict[str, Any], Dict[str, Any]],
) -> PipelineOutput:
    """ワーカープロセスでサンプルを処理"""
    global _worker_runner
    assert _worker_runner is not None
    idx, sample, dataset_config = args
    return _worker_runner.run_sample(idx, sample, dataset_config)


def load_dataset(
    dataset_name: str,
    num_samples: int,
    random_sample: bool = False,
    datasets_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """データセットを読み込み

    Args:
        dataset_name: データセット名
        num_samples: 取得するサンプル数
        random_sample: Trueの場合、ランダムにサンプリング
        datasets_config: データセット設定（省略時はDATASETS_V2を使用）
    """
    import random

    if datasets_config is None:
        datasets_config = DATASETS_V2
    config = datasets_config[dataset_name]
    data_path = Path(config["path"])

    if not data_path.exists():
        return []

    # 全サンプルを読み込み
    all_samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    # サンプル数が指定数より少ない場合はそのまま返す
    if len(all_samples) <= num_samples:
        return all_samples

    # ランダムサンプリングまたは先頭から取得
    if random_sample:
        if seed is not None:
            random.seed(seed)
        return random.sample(all_samples, num_samples)
    else:
        return all_samples[:num_samples]


def run_pipeline_evaluation(
    pipeline_id: str,
    datasets: List[str],
    num_samples: int,
    output_dir: Optional[Path] = None,
    random_sample: bool = False,
    num_workers: int = 1,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    kg_type: str = "primekgqa",
    per_sample_timeout_s: float = 300.0,
    watch_interval_s: float = 2.0,
    max_inflight: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    パイプライン評価を実行

    Args:
        pipeline_id: パイプラインID
        datasets: 評価するデータセットのリスト
        num_samples: 各データセットから取得するサンプル数
        output_dir: 結果出力ディレクトリ
        random_sample: Trueの場合、ランダムにサンプリング
        num_workers: 並列ワーカー数（デフォルト: 1）
        pipeline_kwargs: パイプライン初期化パラメータ
        kg_type: Knowledge Graphの種類（primekgqa or metaqa）

    Returns:
        データセット名をキーとした結果辞書
    """
    # KGに応じた設定を取得
    pipeline_configs = PIPELINE_CONFIGS_BY_KG.get(kg_type, PIPELINE_CONFIGS_PRIMEKGQA)
    datasets_config = DATASETS_BY_KG.get(kg_type, DATASETS_PRIMEKGQA)

    config = pipeline_configs[pipeline_id]
    pipeline_name = config["name"]
    available_datasets = config["datasets"]
    pipeline_kwargs = pipeline_kwargs or {}

    # 対応データセットのみに絞る
    datasets = [d for d in datasets if d in available_datasets]

    if not datasets:
        print(f"  [SKIP] No compatible datasets for {pipeline_name}")
        return {}

    # シングルワーカー用のランナー
    runner = None if num_workers > 1 else PipelineRunner(pipeline_id, pipeline_kwargs)

    results = {}

    for dataset_name in datasets:
        print(f"\n  [{dataset_name}]")

        # データ読み込み（ランダムサンプリング対応）
        dataset_path = datasets_config[dataset_name]["path"]
        samples = load_dataset(
            dataset_name, num_samples, random_sample, datasets_config, seed=seed
        )
        if not samples:
            print(f"    [SKIP] No data found")
            continue

        print(f"    Loaded {len(samples)} samples (workers: {num_workers})")
        dataset_config = datasets_config[dataset_name]

        # パイプライン実行
        outputs: List[PipelineOutput] = []

        def _timeout_output(
            idx: int, sample: Dict[str, Any], err: str
        ) -> PipelineOutput:
            question = sample.get("question", "")
            entity_name = sample.get(dataset_config["entity_key"], "")
            answers_key = dataset_config.get("gold_answers_key", "answer_nodes")
            raw_answers = sample.get(answers_key, [])
            if raw_answers and isinstance(raw_answers[0], dict):
                gold_answers = [node["name"] for node in raw_answers]
            else:
                gold_answers = list(raw_answers)
            gold_relations = [
                sample.get(k, "")
                for k in dataset_config["gold_relations_keys"]
                if sample.get(k)
            ]
            out = PipelineOutput(
                idx=idx,
                question=question,
                entity_name=entity_name,
                gold_answers=gold_answers,
                gold_relations=gold_relations,
                predicted_relations=None,
                predicted_entities=[],
            )
            out.error = err
            return out

        if num_workers > 1:
            # マルチプロセス並列実行（ハング対策：spawn + per-sample timeout + 監視ループ）
            # NOTE: fork は外部API/HTTPクライアント利用時にデッドロックしやすいので spawn を使う
            ctx = mp.get_context("spawn")
            inflight_limit = max_inflight or max(1, num_workers * 2)

            indexed_samples = list(enumerate(samples))
            next_submit = 0

            future_to_meta: Dict[Any, Tuple[int, Dict[str, Any]]] = {}
            submit_time: Dict[Any, float] = {}
            pending = set()

            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(pipeline_id, pipeline_kwargs),
                mp_context=ctx,
            ) as ex:
                with tqdm(total=len(samples), desc=f"    Running", leave=False) as pbar:
                    try:
                        while next_submit < len(indexed_samples) or pending:
                            # 追加投入（過剰キューイングを避ける）
                            while (
                                next_submit < len(indexed_samples)
                                and len(pending) < inflight_limit
                            ):
                                i, sample = indexed_samples[next_submit]
                                fut = ex.submit(
                                    _process_sample_worker, (i, sample, dataset_config)
                                )
                                future_to_meta[fut] = (i, sample)
                                submit_time[fut] = time.time()
                                pending.add(fut)
                                next_submit += 1

                            if not pending:
                                continue

                            done, not_done = wait(
                                pending,
                                timeout=watch_interval_s,
                                return_when=FIRST_COMPLETED,
                            )

                            # 完了回収
                            for fut in done:
                                i, _sample = future_to_meta[fut]
                                try:
                                    output = fut.result()
                                except Exception as e:
                                    output = _timeout_output(
                                        i, _sample, f"future_error: {e}"
                                    )
                                outputs.append(output)
                                pbar.update(1)
                                pending.discard(fut)

                            # タイムアウト検出（待たずに結果化して進める）
                            now = time.time()
                            timed_out = []
                            for fut in not_done:
                                if (
                                    now - submit_time.get(fut, now)
                                    > per_sample_timeout_s
                                ):
                                    i, _sample = future_to_meta[fut]
                                    fut.cancel()
                                    outputs.append(
                                        _timeout_output(
                                            i,
                                            _sample,
                                            f"timeout({per_sample_timeout_s}s)",
                                        )
                                    )
                                    pbar.update(1)
                                    timed_out.append(fut)

                            for fut in timed_out:
                                pending.discard(fut)

                    finally:
                        # 重要：残タスクを待たずに閉じる（1件ハングしても全体が止まらない）
                        ex.shutdown(wait=False, cancel_futures=True)

            outputs.sort(key=lambda x: x.idx)
        else:
            # シーケンシャル実行
            for i, sample in enumerate(tqdm(samples, desc=f"    Running", leave=False)):
                assert runner is not None
                output = runner.run_sample(i, sample, dataset_config)
                outputs.append(output)

        # 結果保存
        if output_dir:
            save_pipeline_outputs(outputs, output_dir, pipeline_id, dataset_name)

        # 精度評価
        eval_results = evaluate_outputs(outputs)
        result_dicts = [
            {
                "accuracy": r.accuracy,
                "recall": r.recall,
                "precision": r.precision,
                "f1": r.f1,
                "path_match": r.path_match,
                "error": r.error,
                "latency_ms": r.latency_ms,
            }
            for r in eval_results
        ]
        metrics = aggregate_metrics(result_dicts, len(result_dicts))

        # 失敗分析
        failure_analysis = analyze_failures(eval_results)

        results[dataset_name] = {
            "metrics": metrics,
            "outputs": outputs,
            "eval_results": eval_results,
            "failure_analysis": failure_analysis,
        }

        # 結果表示
        print(f"    Accuracy:  {metrics['accuracy']:.1f}%")
        print(f"    Recall:    {metrics['recall']:.1f}%")
        print(f"    Precision: {metrics['precision']:.1f}%")
        print(f"    F1:        {metrics['f1']:.1f}%")
        print(f"    PathAcc:   {metrics.get('path_accuracy', 0):.1f}%")
        print(f"    Errors:    {metrics['errors']}/{metrics['total']}")
        print(f"    Latency:   {metrics['avg_latency_ms']:.0f}ms")

        # 失敗分析表示
        print_failure_summary(failure_analysis)

        # 失敗分析の保存
        if output_dir:
            fa_path = save_failure_analysis(
                failure_analysis, output_dir, pipeline_id, dataset_name
            )
            print(f"    Failures saved to: {fa_path}")

    return results


def run_nl_evaluation(
    pipeline_id: str,
    num_samples: int,
    random_sample: bool = False,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    kg_type: str = "pcqa",
    seed: Optional[int] = None,
    nl_data: str = "pcqa",
) -> Dict[str, Any]:
    """
    NL（自然言語）評価を実行（KGT論文準拠: ROUGE-L + Embedding Cosine Similarity）

    PcQA.json の gold answer テキストと、パイプラインが生成した NL 回答を比較する。

    nl_data:
      - "pcqa": PcQA.json全405サンプル（デフォルト）
      - "eval_v3": eval_v3.jsonlの241サンプル（属性フィルタ不要な質問のみ）
    """
    import random as _random

    # PcQA.json を読み込み（gold NL answer用）
    pcqa_path = Path("data/pcqa/PcQA.json")
    if not pcqa_path.exists():
        print(f"  [ERROR] {pcqa_path} not found")
        return {}

    with pcqa_path.open("r", encoding="utf-8") as f:
        pcqa_all = json.load(f)

    if nl_data == "eval_v3":
        # eval_v3.jsonlからquestion + entity_nameを取得、PcQA.jsonからgold NL answerを取得
        eval_path = Path("data/pcqa/qa/eval_v3.jsonl")
        if not eval_path.exists():
            print(f"  [ERROR] {eval_path} not found")
            return {}

        eval_samples = []
        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                eval_samples.append(json.loads(line.strip()))

        print(f"  Loaded {len(eval_samples)} samples from eval_v3.jsonl")

        # (orig_idx, {"question": ..., "answer": ..., "entity_name": ...}) のリストを構築
        # original_index は1ベース → 0ベースに変換
        all_samples_with_entity = []
        for es in eval_samples:
            orig_idx = es.get("original_index", 1)
            pcqa_idx = orig_idx - 1  # 1-based → 0-based
            if 0 <= pcqa_idx < len(pcqa_all):
                gold_answer = pcqa_all[pcqa_idx].get("answer", "")
            else:
                gold_answer = ""
            all_samples_with_entity.append(
                (
                    orig_idx,
                    {
                        "question": es["question"],
                        "answer": gold_answer,
                        "entity_name": es.get("entity"),
                    },
                )
            )

        # サンプリング
        if len(all_samples_with_entity) > num_samples:
            if random_sample:
                if seed is not None:
                    _random.seed(seed)
                samples = _random.sample(all_samples_with_entity, num_samples)
            else:
                samples = all_samples_with_entity[:num_samples]
        else:
            samples = all_samples_with_entity
    else:
        print(f"  Loaded {len(pcqa_all)} samples from PcQA.json")

        # サンプリング
        all_samples = pcqa_all
        if len(all_samples) > num_samples:
            if random_sample:
                if seed is not None:
                    _random.seed(seed)
                samples = _random.sample(list(enumerate(all_samples)), num_samples)
            else:
                samples = list(enumerate(all_samples[:num_samples]))
        else:
            samples = list(enumerate(all_samples))

    # KGTPipeline を直接初期化（generate_nl パラメータを使うため）
    from pipeline.kgt import KGTPipeline

    pipeline_kwargs = pipeline_kwargs or {}
    pipeline = KGTPipeline(**pipeline_kwargs)

    # Embedding初期化（コサイン類似度計算用）
    embeddings = pipeline.embeddings  # KGTPipelineが既に持っている

    # 評価実行
    nl_results: List[NLEvalResult] = []

    for orig_idx, sample in tqdm(samples, desc="  Running NL eval", leave=False):
        question = sample["question"]
        gold_answer = sample["answer"]

        # "Output: " プレフィックスを除去して比較
        gold_text = gold_answer
        if gold_text.startswith("Output: "):
            gold_text = gold_text[len("Output: ") :]

        result = NLEvalResult(
            idx=orig_idx,
            question=question,
            gold_answer=gold_text,
            predicted_answer="",
        )

        try:
            start = time.time()
            entity_name = (
                sample.get("entity_name") if isinstance(sample, dict) else None
            )
            pipeline_result = pipeline.run(
                question=question, entity_name=entity_name, generate_nl=True
            )
            result.latency_ms = (time.time() - start) * 1000

            pred_answer = pipeline_result.natural_answer or ""
            # "Output: " プレフィックスを除去
            if pred_answer.startswith("Output: "):
                pred_answer = pred_answer[len("Output: ") :]
            result.predicted_answer = pred_answer

            # ROUGE-L
            rouge_scores = compute_rouge_l(gold_text, pred_answer)
            result.rouge_l_r = rouge_scores["r"]
            result.rouge_l_p = rouge_scores["p"]
            result.rouge_l_f = rouge_scores["f"]

            # Embedding Cosine Similarity
            result.embedding_similarity = compute_embedding_similarity(
                gold_text, pred_answer, embeddings
            )

        except Exception as e:
            result.error = str(e)

        nl_results.append(result)

    # 集計
    metrics = aggregate_nl_metrics(nl_results)

    # 結果表示
    print(f"\n  === NL Evaluation Results ({pipeline_id}) ===")
    print(f"  Total:              {metrics.get('total', 0)}")
    print(f"  Errors:             {metrics.get('errors', 0)}")
    print(f"  ROUGE-L Recall:     {metrics.get('rouge_l_recall', 0):.1f}%")
    print(f"  ROUGE-L Precision:  {metrics.get('rouge_l_precision', 0):.1f}%")
    print(f"  ROUGE-L F1:         {metrics.get('rouge_l_f1', 0):.1f}%")
    print(f"  Embed Similarity:   {metrics.get('embedding_similarity', 0):.1f}%")
    print(f"  Avg Latency:        {metrics.get('avg_latency_ms', 0):.0f}ms")

    # サンプル表示
    print(f"\n  --- Sample Predictions (first 5) ---")
    for r in nl_results[:5]:
        print(f"  [{r.idx}] Q: {r.question[:60]}...")
        print(f"       Gold: {r.gold_answer[:80]}...")
        print(f"       Pred: {r.predicted_answer[:80]}...")
        print(
            f"       ROUGE-L F1: {r.rouge_l_f:.3f}  EmbSim: {r.embedding_similarity:.3f}"
        )
        print()

    return {
        "metrics": metrics,
        "nl_results": nl_results,
    }


def print_comparison_table(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    kg_type: str = "primekgqa",
) -> None:
    """比較テーブルを表示"""

    # KGに応じたデータセットリスト
    datasets_config = DATASETS_BY_KG.get(kg_type, DATASETS_PRIMEKGQA)
    pipeline_configs = PIPELINE_CONFIGS_BY_KG.get(kg_type, PIPELINE_CONFIGS_PRIMEKGQA)
    datasets = list(datasets_config.keys())

    print(f"\n{'='*100}")
    print(f"COMPARISON SUMMARY ({kg_type.upper()})")
    print(f"{'='*100}")

    # データセットごとの比較
    for dataset in datasets:
        pipelines_with_data = [
            (pid, pdata) for pid, pdata in all_results.items() if dataset in pdata
        ]

        if not pipelines_with_data:
            continue

        dataset_display = DATASET_NAMES.get(dataset, dataset)
        print(f"\n--- {dataset_display} ---")
        print(
            f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10} {'Latency':>10}"
        )
        print("-" * 95)

        for pipeline_id, pipeline_data in pipelines_with_data:
            m = pipeline_data[dataset]["metrics"]
            pipeline_name = pipeline_configs[pipeline_id]["name"]
            print(
                f"{pipeline_name:<25} "
                f"{m['accuracy']:>9.1f}% "
                f"{m['recall']:>9.1f}% "
                f"{m['precision']:>9.1f}% "
                f"{m['f1']:>9.1f}% "
                f"{m.get('path_accuracy', 0):>9.1f}% "
                f"{m['avg_latency_ms']:>8.0f}ms"
            )

    # パイプラインごとの平均
    print(f"\n{'='*100}")
    print("PIPELINE AVERAGES (across all evaluated datasets)")
    print(f"{'='*100}")
    print(
        f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10}"
    )
    print("-" * 85)

    for pipeline_id, pipeline_data in all_results.items():
        if not pipeline_data:
            continue

        n = len(pipeline_data)
        avg_acc = sum(d["metrics"]["accuracy"] for d in pipeline_data.values()) / n
        avg_recall = sum(d["metrics"]["recall"] for d in pipeline_data.values()) / n
        avg_prec = sum(d["metrics"]["precision"] for d in pipeline_data.values()) / n
        avg_f1 = sum(d["metrics"]["f1"] for d in pipeline_data.values()) / n
        avg_path = (
            sum(d["metrics"].get("path_accuracy", 0) for d in pipeline_data.values())
            / n
        )

        pipeline_name = pipeline_configs[pipeline_id]["name"]
        print(
            f"{pipeline_name:<25} "
            f"{avg_acc:>9.1f}% "
            f"{avg_recall:>9.1f}% "
            f"{avg_prec:>9.1f}% "
            f"{avg_f1:>9.1f}% "
            f"{avg_path:>9.1f}%"
        )


def main():
    p = argparse.ArgumentParser(
        description="Run evaluation pipeline from dataset to metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pipeline
  python pipeline/run_evaluation.py --pipeline kgt --num-samples 20

  # Multiple pipelines (comparison)
  python pipeline/run_evaluation.py --pipeline extended_type_kopl safe kgt

  # All pipelines
  python pipeline/run_evaluation.py --all --num-samples 20

  # Specific datasets
  python pipeline/run_evaluation.py --pipeline kgt --dataset one_hop two_hop
        """,
    )
    p.add_argument(
        "--kg",
        type=str,
        default="primekgqa",
        choices=["primekgqa", "primekgqa_raw", "primekgqa_original", "metaqa", "pcqa", "webqsp", "kqapro"],
        help="Knowledge Graph to use (default: primekgqa). primekgqa_raw uses no-paraphrase dataset.",
    )
    p.add_argument("--pipeline", type=str, nargs="+", help="Pipeline(s) to evaluate")
    p.add_argument("--all", action="store_true", help="Evaluate all pipelines")
    p.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="Specific dataset(s) to evaluate (default: all compatible)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples per dataset (default: 20)",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample from dataset instead of taking first N",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible sampling"
    )
    p.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for pipeline outputs (optional)",
    )
    p.add_argument(
        "--output", type=Path, default=None, help="Output JSON file for metrics summary"
    )
    p.add_argument(
        "--safe-delta",
        type=int,
        default=1,
        help="SAFE pipeline delta parameter (default: 1)",
    )
    p.add_argument(
        "--safe-k-sim-ent",
        type=int,
        default=3,
        help="SAFE pipeline SimEnt k parameter (default: 3)",
    )
    p.add_argument(
        "--reranker",
        type=str,
        default="none",
        choices=["none", "llm", "hybrid"],
        help="Extended Type-KoPL reranker type (default: none)",
    )
    p.add_argument(
        "--reranker-input-k",
        type=int,
        default=10,
        help="Number of candidates to pass to reranker (default: 10)",
    )
    p.add_argument(
        "--use-llm-cypher",
        action="store_true",
        default=False,
        help="Use LLM-generated Cypher in ETK Phase 4 instead of templates",
    )
    p.add_argument(
        "--n-kopl-candidates",
        type=int,
        default=1,
        help="Number of KoPL candidates to generate (>1 enables multi-candidate selection by schema score, default: 1)",
    )
    p.add_argument(
        "--schema-distill",
        action="store_true",
        default=False,
        help="Enable enhanced schema distillation for Phase 1 prompt (NL forms, frequency hints, relevance sorting)",
    )
    p.add_argument(
        "--max-correction-rounds",
        type=int,
        default=0,
        help="Max Phase 1→2 correction rounds when Phase 2 returns 0 paths (default: 0, disabled)",
    )
    p.add_argument(
        "--per-sample-timeout",
        type=float,
        default=300.0,
        help="Per-sample timeout seconds for parallel execution (default: 300)",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="Max inflight tasks for parallel execution (default: workers*2)",
    )
    p.add_argument(
        "--few-shot-k",
        type=int,
        default=3,
        help="Number of dynamic few-shot examples to select via MMR (default: 3)",
    )
    p.add_argument(
        "--few-shot-pool",
        type=str,
        default=None,
        help="Path to few-shot pool JSON (enables retrieval-based few-shot). Use 'auto' to auto-detect from KG type.",
    )
    p.add_argument(
        "--no-schema",
        action="store_true",
        help="Run without schema relations (only type enumeration, relations from KG)",
    )
    p.add_argument(
        "--cypher-informed-rerank",
        action="store_true",
        default=False,
        help="Enable Cypher-Informed Reranking: trial-execute candidate paths and show example results to Reranker",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: env LLM_MODEL or gpt-4.1-mini). Use 'ollama/gemma3:27b' for local LLM via LiteLLM.",
    )
    p.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="LLM API base URL (default: env LLM_API_BASE). E.g., 'http://100.96.246.39:4000/v1' for LiteLLM proxy.",
    )
    p.add_argument(
        "--eval-mode",
        type=str,
        default="set",
        choices=["set", "nl"],
        help="Evaluation mode: 'set' (entity set matching, default) or 'nl' (natural language ROUGE-L/EmbSim, PCQA only)",
    )
    p.add_argument(
        "--nl-data",
        type=str,
        default="pcqa",
        choices=["pcqa", "eval_v3"],
        help="NL evaluation data source: 'pcqa' (PcQA.json 405 samples) or 'eval_v3' (eval_v3.jsonl 241 samples, no attribute questions)",
    )
    args = p.parse_args()

    # LLMモデル/API base の設定（環境変数経由でパイプラインに伝播）
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.api_base:
        os.environ["LLM_API_BASE"] = args.api_base
    # 環境変数を反映して config を再読込
    import core.config as _cfg
    _cfg.BASEMODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    _cfg.LLM_API_BASE = os.getenv("LLM_API_BASE", "")
    if args.model or args.api_base:
        print(f"LLM:       {_cfg.BASEMODEL}")
        if _cfg.LLM_API_BASE:
            print(f"API Base:  {_cfg.LLM_API_BASE}")

    # KGに応じた設定を取得
    kg_type = args.kg
    pipeline_configs = PIPELINE_CONFIGS_BY_KG.get(kg_type, PIPELINE_CONFIGS_PRIMEKGQA)
    datasets_config = DATASETS_BY_KG.get(kg_type, DATASETS_PRIMEKGQA)

    # パイプライン選択
    if args.all:
        pipelines = list(pipeline_configs.keys())
    elif args.pipeline:
        # パイプライン名を検証
        for p_id in args.pipeline:
            if p_id not in pipeline_configs:
                print(f"Error: Unknown pipeline '{p_id}' for KG '{kg_type}'")
                print(f"Available: {list(pipeline_configs.keys())}")
                return
        pipelines = args.pipeline
    else:
        p.print_help()
        print("\nError: Please specify --pipeline or --all")
        return

    # データセット選択
    if args.dataset:
        # データセット名を検証
        for d_id in args.dataset:
            if d_id not in datasets_config:
                print(f"Error: Unknown dataset '{d_id}' for KG '{kg_type}'")
                print(f"Available: {list(datasets_config.keys())}")
                return
        datasets = args.dataset
    else:
        datasets = list(datasets_config.keys())

    print(f"{'='*60}")
    print("EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"KG:        {kg_type.upper()}")
    print(f"Pipelines: {', '.join(pipelines)}")
    print(f"Datasets:  {', '.join(datasets)}")
    print(f"Samples:   {args.num_samples}" + (" (random)" if args.random else ""))
    print(f"Workers:   {args.workers}")
    print(f"Eval Mode: {args.eval_mode}")
    if args.no_schema:
        print(f"Schema:    NO (relations from KG)")
    if args.output_dir:
        print(f"Output:    {args.output_dir}")

    # NL評価モード（PCQA専用）
    if args.eval_mode == "nl":
        if kg_type != "pcqa":
            print("\nError: --eval-mode nl is only supported for --kg pcqa")
            return
        for pipeline_id in pipelines:
            pipeline_name = pipeline_configs[pipeline_id]["name"]
            nl_data = getattr(args, "nl_data", "pcqa")
            print(f"\n{'='*60}")
            print(f"Pipeline: {pipeline_name} (NL evaluation, data={nl_data})")
            print(f"{'='*60}")
            pipeline_kwargs = {"kg_type": kg_type}
            run_nl_evaluation(
                pipeline_id,
                args.num_samples,
                args.random,
                pipeline_kwargs,
                kg_type,
                seed=args.seed,
                nl_data=nl_data,
            )
        return

    all_results = {}

    for pipeline_id in pipelines:
        pipeline_name = pipeline_configs[pipeline_id]["name"]
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{'='*60}")

        # パイプライン固有のパラメータ
        # primekgqa_raw / primekgqa_original は KG自体は primekgqa（データセットのみ異なる）
        pipeline_kg_type = "primekgqa" if kg_type in ("primekgqa_raw", "primekgqa_original") else kg_type
        pipeline_kwargs = {"kg_type": pipeline_kg_type}

        # スキーマなしモード（Extended Type-KoPL, SAFE のみ対応）
        if args.no_schema and pipeline_id in ["extended_type_kopl", "safe"]:
            pipeline_kwargs["use_schema_relations"] = False
            print(f"  (no-schema mode: relations from KG)")

        if pipeline_id == "safe":
            pipeline_kwargs["delta"] = args.safe_delta
            pipeline_kwargs["k_sim_ent"] = args.safe_k_sim_ent
            if args.safe_delta != 1:
                print(f"  (delta={args.safe_delta})")
            if args.safe_k_sim_ent != 3:
                print(f"  (k_sim_ent={args.safe_k_sim_ent})")
        if pipeline_id == "extended_type_kopl":
            pipeline_kwargs["reranker_type"] = args.reranker
            pipeline_kwargs["reranker_input_k"] = args.reranker_input_k
            pipeline_kwargs["use_llm_cypher"] = args.use_llm_cypher
            if args.reranker != "none":
                print(f"  (reranker={args.reranker}, input_k={args.reranker_input_k})")
            if args.use_llm_cypher:
                print("  (use_llm_cypher=True)")
            if args.n_kopl_candidates > 1:
                pipeline_kwargs["n_kopl_candidates"] = args.n_kopl_candidates
                print(f"  (n_kopl_candidates={args.n_kopl_candidates})")
            if args.schema_distill:
                pipeline_kwargs["schema_distill"] = True
                print("  (schema_distill=True)")
            if args.max_correction_rounds > 0:
                pipeline_kwargs["max_correction_rounds"] = args.max_correction_rounds
            if args.cypher_informed_rerank:
                pipeline_kwargs["cypher_informed_rerank"] = True
                print("  (cypher_informed_rerank=True)")
                print(f"  (max_correction_rounds={args.max_correction_rounds})")
            # Retrieval-based few-shot
            few_shot_pool = args.few_shot_pool
            if few_shot_pool == "auto":
                _POOL_MAP = {
                    "primekgqa": "pipeline/extended_type_kopl/few_shot_pools/primekgqa_pool.json",
                    "metaqa": "pipeline/extended_type_kopl/few_shot_pools/metaqa_pool.json",
                    "pcqa": "pipeline/extended_type_kopl/few_shot_pools/pcqa_pool.json",
                }
                few_shot_pool = _POOL_MAP.get(pipeline_kg_type)
                if not few_shot_pool:
                    print(f"  Warning: no auto pool for kg_type={pipeline_kg_type}")
            if few_shot_pool:
                pipeline_kwargs["few_shot_pool_path"] = few_shot_pool
                pipeline_kwargs["few_shot_k"] = args.few_shot_k
                print(f"  (few_shot_pool={few_shot_pool}, k={args.few_shot_k})")

        results = run_pipeline_evaluation(
            pipeline_id,
            datasets,
            args.num_samples,
            args.output_dir,
            args.random,
            args.workers,
            pipeline_kwargs,
            kg_type,
            per_sample_timeout_s=args.per_sample_timeout,
            max_inflight=args.max_inflight,
            seed=args.seed,
        )
        all_results[pipeline_id] = results

    # 複数パイプラインの場合は比較テーブルを表示
    if len(pipelines) > 1:
        print_comparison_table(all_results, kg_type)

    # メトリクスをJSON出力
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            pipeline_id: {
                dataset_name: data["metrics"]
                for dataset_name, data in pipeline_data.items()
            }
            for pipeline_id, pipeline_data in all_results.items()
        }
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
