"""
共通評価メトリクス

全パイプライン評価スクリプトで使用する共通の評価ロジック
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class PipelineOutput:
    """パイプライン実行結果（精度算出前の生の出力）"""
    idx: int
    question: str
    entity_name: str
    gold_answers: List[str]
    gold_relations: List[str]
    predicted_relations: Optional[List[str]]
    predicted_entities: List[str]
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineOutput":
        return cls(**d)


@dataclass
class EvalResult:
    """評価結果（精度算出後）"""
    idx: int
    question: str
    entity_name: str
    gold_answers: List[str]
    gold_relations: List[str]
    predicted_relations: Optional[List[str]]
    predicted_entities: List[str]
    accuracy: bool = False
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    path_match: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_pipeline_output(cls, output: PipelineOutput) -> "EvalResult":
        """PipelineOutputからEvalResultを作成（精度は後で計算）"""
        return cls(
            idx=output.idx,
            question=output.question,
            entity_name=output.entity_name,
            gold_answers=output.gold_answers,
            gold_relations=output.gold_relations,
            predicted_relations=output.predicted_relations,
            predicted_entities=output.predicted_entities,
            error=output.error,
            latency_ms=output.latency_ms,
        )


@dataclass
class SetMetrics:
    """集合ベースの評価メトリクス"""
    accuracy: bool = False    # 正解が全て含まれるか
    recall: float = 0.0       # 正解のうち何割が含まれるか
    precision: float = 0.0    # 予測のうち何割が正解か
    f1: float = 0.0           # F1スコア


def compute_set_metrics(gold_set: Set[str], pred_set: Set[str]) -> SetMetrics:
    """
    集合ベースのメトリクスを計算

    Args:
        gold_set: 正解エンティティの集合
        pred_set: 予測エンティティの集合

    Returns:
        SetMetrics: 計算されたメトリクス
    """
    metrics = SetMetrics()

    if not gold_set:
        return metrics

    overlap = gold_set & pred_set
    metrics.accuracy = gold_set <= pred_set  # 正解が全て予測に含まれるか
    metrics.recall = len(overlap) / len(gold_set)
    metrics.precision = len(overlap) / len(pred_set) if pred_set else 0.0

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

    return metrics


def check_path_match(
    predicted_relations: Optional[List[str]],
    gold_relations: List[str],
) -> bool:
    """
    予測されたリレーションパスが正解と完全一致するかチェック

    Args:
        predicted_relations: 予測されたリレーションのリスト（順序付き）
        gold_relations: 正解リレーションのリスト（順序付き）

    Returns:
        bool: 完全一致すればTrue
    """
    if not predicted_relations or not gold_relations:
        return False

    # 順序を含めた完全一致チェック
    if len(predicted_relations) != len(gold_relations):
        return False

    return predicted_relations == gold_relations


def aggregate_metrics(
    results: List[dict],
    total: int,
) -> dict:
    """
    結果リストからメトリクスを集計

    Args:
        results: 各サンプルの結果辞書のリスト
        total: 総サンプル数

    Returns:
        dict: 集計されたメトリクス
    """
    if total == 0:
        return {}

    errors = sum(1 for r in results if r.get("error"))
    success = total - errors

    metrics = {
        "total": total,
        "success": success,
        "errors": errors,
        "accuracy": sum(r.get("accuracy", False) for r in results) / total * 100,
        "recall": sum(r.get("recall", 0.0) for r in results) / total * 100,
        "precision": sum(r.get("precision", 0.0) for r in results) / total * 100,
        "f1": sum(r.get("f1", 0.0) for r in results) / total * 100,
        "avg_latency_ms": sum(r.get("latency_ms", 0.0) for r in results) / total,
    }

    # path_matchがある場合のみ集計
    if any("path_match" in r for r in results):
        metrics["path_accuracy"] = sum(r.get("path_match", False) for r in results) / total * 100

    return metrics


def evaluate_output(output: PipelineOutput) -> EvalResult:
    """
    PipelineOutputから精度を計算してEvalResultを返す

    Args:
        output: パイプラインの生の出力

    Returns:
        EvalResult: 精度算出後の評価結果
    """
    result = EvalResult.from_pipeline_output(output)

    if output.error:
        return result

    gold_set = set(output.gold_answers)
    pred_set = set(output.predicted_entities)

    metrics = compute_set_metrics(gold_set, pred_set)
    result.accuracy = metrics.accuracy
    result.recall = metrics.recall
    result.precision = metrics.precision
    result.f1 = metrics.f1

    result.path_match = check_path_match(
        output.predicted_relations, output.gold_relations
    )

    return result


def save_pipeline_outputs(
    outputs: List[PipelineOutput],
    output_path: Path,
    pipeline_name: str,
    dataset_name: str,
) -> None:
    """
    パイプライン出力をJSONLファイルに保存

    Args:
        outputs: PipelineOutputのリスト
        output_path: 出力ディレクトリ
        pipeline_name: パイプライン名（例: "extended_type_kopl", "safe", "kgt"）
        dataset_name: データセット名（例: "one_hop", "two_hop"）
    """
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{pipeline_name}_{dataset_name}.jsonl"

    with file_path.open("w", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(output.to_dict(), ensure_ascii=False) + "\n")


def load_pipeline_outputs(file_path: Path) -> List[PipelineOutput]:
    """
    JSONLファイルからパイプライン出力を読み込む

    Args:
        file_path: 入力ファイルパス

    Returns:
        List[PipelineOutput]: 読み込んだ出力のリスト
    """
    outputs = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(PipelineOutput.from_dict(json.loads(line)))
    return outputs


def evaluate_outputs(outputs: List[PipelineOutput]) -> List[EvalResult]:
    """
    複数のPipelineOutputをまとめて評価

    Args:
        outputs: PipelineOutputのリスト

    Returns:
        List[EvalResult]: 評価結果のリスト
    """
    return [evaluate_output(output) for output in outputs]
