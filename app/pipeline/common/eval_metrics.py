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
    # ステップ詳細（オプション）
    entity_type: Optional[str] = None
    target_type: Optional[str] = None
    kopl_program: Optional[Dict[str, Any]] = None  # Phase 1: 生成されたKoPL
    candidate_paths: Optional[List[str]] = None     # Phase 2: 候補パス一覧
    selected_paths: Optional[List[str]] = None      # Phase 3/3.5: 選択されたパス
    processing_log: Optional[List[str]] = None      # 全ステップの処理ログ

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
    accuracy: bool = False    # Exact Match: 正解と予測が完全一致か
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
    metrics.accuracy = gold_set == pred_set  # Exact Match: 正解と予測が完全一致
    metrics.recall = len(overlap) / len(gold_set)
    metrics.precision = len(overlap) / len(pred_set) if pred_set else 0.0

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

    return metrics


# リレーション名のエイリアスマッピング（KG名 -> 正規化名）
# 逆方向リレーションも順方向に正規化
RELATION_ALIASES = {
    # associated_disease / associated_with は同じ
    "associated_with": "associated_disease",
    "associated with": "associated_disease",
    # 逆方向リレーションを順方向に正規化
    "targeted_by": "target",
    "treated_by": "indication",
    "caused_by_drug": "side_effect",
    "interacted_by": "interacts_with",
    "expressed_gene": "expression_present",
    "expressed gene": "expression_present",
    "absent_gene": "expression_absent",
    "absent gene": "expression_absent",
}


def normalize_relation(rel: str) -> str:
    """
    リレーション名を正規化（スペースとアンダースコアを統一、エイリアス解決）

    Args:
        rel: リレーション名

    Returns:
        str: 正規化されたリレーション名（小文字、スペース→アンダースコア）
    """
    normalized = rel.lower().replace(" ", "_").replace("-", "_")
    # エイリアス解決
    return RELATION_ALIASES.get(normalized, normalized)


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

    # 正規化して比較（スペース/アンダースコア/ハイフンの違いを吸収）
    normalized_pred = [normalize_relation(r) for r in predicted_relations]
    normalized_gold = [normalize_relation(r) for r in gold_relations]

    return normalized_pred == normalized_gold


class FailureCategory:
    """失敗カテゴリ定数"""
    SUCCESS = "success"
    ERROR = "error"                # パイプライン実行エラー
    PATH_MISMATCH = "path_mismatch"  # リレーションパスが不一致
    NO_MATCH = "no_match"          # 予測エンティティが正解と完全不一致
    PARTIAL_MATCH = "partial_match"  # 一部一致（recall > 0 だが accuracy = False）
    OVER_PREDICTION = "over_prediction"  # 正解は全て含むが余分な予測あり


def classify_failure(result: EvalResult) -> str:
    """EvalResultの失敗カテゴリを分類"""
    if result.error:
        return FailureCategory.ERROR
    if result.accuracy:
        return FailureCategory.SUCCESS
    if result.recall == 0.0:
        return FailureCategory.NO_MATCH
    if result.recall == 1.0 and result.precision < 1.0:
        return FailureCategory.OVER_PREDICTION
    return FailureCategory.PARTIAL_MATCH


def analyze_failures(
    eval_results: List[EvalResult],
) -> Dict[str, Any]:
    """
    失敗分析を実行

    Args:
        eval_results: 評価結果のリスト

    Returns:
        dict: カテゴリ別の集計と失敗サンプル詳細
    """
    categories: Dict[str, List[Dict[str, Any]]] = {
        FailureCategory.SUCCESS: [],
        FailureCategory.ERROR: [],
        FailureCategory.PATH_MISMATCH: [],
        FailureCategory.NO_MATCH: [],
        FailureCategory.PARTIAL_MATCH: [],
        FailureCategory.OVER_PREDICTION: [],
    }

    for r in eval_results:
        cat = classify_failure(r)
        detail: Dict[str, Any] = {
            "idx": r.idx,
            "question": r.question,
            "entity_name": r.entity_name,
            "gold_answers": r.gold_answers,
            "predicted_entities": r.predicted_entities,
            "gold_relations": r.gold_relations,
            "predicted_relations": r.predicted_relations,
            "recall": r.recall,
            "precision": r.precision,
            "f1": r.f1,
            "path_match": r.path_match,
        }
        if r.error:
            detail["error"] = r.error
        categories[cat].append(detail)

        # path_mismatch は accuracy とは独立に追跡
        if not r.path_match and r.predicted_relations and r.gold_relations:
            if cat != FailureCategory.ERROR:
                categories[FailureCategory.PATH_MISMATCH].append(detail)

    total = len(eval_results)
    summary = {
        "total": total,
        "counts": {k: len(v) for k, v in categories.items()},
        "rates": {
            k: len(v) / total * 100 if total > 0 else 0.0
            for k, v in categories.items()
        },
        "details": categories,
    }
    return summary


def save_failure_analysis(
    analysis: Dict[str, Any],
    output_path: Path,
    pipeline_name: str,
    dataset_name: str,
) -> Path:
    """
    失敗分析をJSONファイルに保存

    Returns:
        Path: 保存先ファイルパス
    """
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{pipeline_name}_{dataset_name}_failures.json"

    # detailsは失敗カテゴリのみ保存（successは除外して軽量化）
    save_data = {
        "total": analysis["total"],
        "counts": analysis["counts"],
        "rates": analysis["rates"],
        "failures": {
            k: v for k, v in analysis["details"].items()
            if k != FailureCategory.SUCCESS
        },
    }

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    return file_path


def print_failure_summary(analysis: Dict[str, Any]) -> None:
    """失敗分析サマリーを表示"""
    counts = analysis["counts"]
    rates = analysis["rates"]
    total = analysis["total"]

    print(f"    --- Failure Analysis ---")
    print(f"    Success:         {counts['success']:>4} ({rates['success']:.1f}%)")
    print(f"    Error:           {counts['error']:>4} ({rates['error']:.1f}%)")
    print(f"    No Match:        {counts['no_match']:>4} ({rates['no_match']:.1f}%)")
    print(f"    Partial Match:   {counts['partial_match']:>4} ({rates['partial_match']:.1f}%)")
    print(f"    Over Prediction: {counts['over_prediction']:>4} ({rates['over_prediction']:.1f}%)")
    print(f"    Path Mismatch:   {counts['path_mismatch']:>4} ({rates['path_mismatch']:.1f}%)")

    # 失敗サンプルを最大3件表示
    for cat in [FailureCategory.ERROR, FailureCategory.NO_MATCH, FailureCategory.PARTIAL_MATCH]:
        details = analysis["details"][cat]
        if not details:
            continue
        print(f"    --- {cat} samples (up to 3) ---")
        for d in details[:3]:
            print(f"      [{d['idx']}] Q: {d['question'][:70]}")
            print(f"           Gold:  {d['gold_answers'][:5]}")
            print(f"           Pred:  {d['predicted_entities'][:5]}")
            if d.get("error"):
                print(f"           Error: {d['error'][:80]}")
            print()


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

    # Case-insensitive matching (PcQA等でエンティティ名の大文字小文字が不一致)
    gold_set = {a.lower() for a in output.gold_answers}
    pred_set = {e.lower() for e in output.predicted_entities}

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


# =============================================================================
# NL (Natural Language) Evaluation Metrics
# =============================================================================

@dataclass
class NLEvalResult:
    """NL評価結果（ROUGE-L + Embedding Cosine Similarity）"""
    idx: int
    question: str
    gold_answer: str
    predicted_answer: str
    rouge_l_r: float = 0.0
    rouge_l_p: float = 0.0
    rouge_l_f: float = 0.0
    embedding_similarity: float = 0.0
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "rouge_l_r": self.rouge_l_r,
            "rouge_l_p": self.rouge_l_p,
            "rouge_l_f": self.rouge_l_f,
            "embedding_similarity": self.embedding_similarity,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


def _rouge_l_fallback(gold: str, predicted: str) -> Dict[str, float]:
    """Pure-Python ROUGE-L using LCS."""
    gold_tokens = gold.strip().split()
    pred_tokens = predicted.strip().split()
    if not gold_tokens or not pred_tokens:
        return {"r": 0.0, "p": 0.0, "f": 0.0}
    # LCS length via DP
    m, n = len(gold_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gold_tokens[i - 1].lower() == pred_tokens[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    r = lcs_len / m if m > 0 else 0.0
    p = lcs_len / n if n > 0 else 0.0
    f = (2 * r * p / (r + p)) if (r + p) > 0 else 0.0
    return {"r": r, "p": p, "f": f}


def compute_rouge_l(gold: str, predicted: str) -> Dict[str, float]:
    """
    ROUGE-Lスコアを計算（KGT論文準拠: rouge Pythonライブラリ使用）

    Args:
        gold: 正解テキスト
        predicted: 予測テキスト

    Returns:
        dict: {"r": recall, "p": precision, "f": f1}
    """
    # 空文字列の場合は0を返す
    if not gold or not predicted or not gold.strip() or not predicted.strip():
        return {"r": 0.0, "p": 0.0, "f": 0.0}

    try:
        from rouge import Rouge

        rouge = Rouge()
        scores = rouge.get_scores(predicted, gold)
        return scores[0]["rouge-l"]
    except ImportError:
        # Fallback: pure-Python ROUGE-L (LCS-based)
        return _rouge_l_fallback(gold, predicted)


def compute_embedding_similarity(
    gold: str,
    predicted: str,
    embeddings,
) -> float:
    """
    Embedding コサイン類似度を計算
    （KGT論文のBERTScore=CLS-tokenコサイン類似度に相当）

    Args:
        gold: 正解テキスト
        predicted: 予測テキスト
        embeddings: OpenAI Embeddingsインスタンス

    Returns:
        float: コサイン類似度 (0-1)
    """
    import numpy as np

    if not gold or not predicted or not gold.strip() or not predicted.strip():
        return 0.0

    try:
        vecs = embeddings.embed_documents([gold, predicted])
        v1, v2 = np.array(vecs[0]), np.array(vecs[1])
        cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return max(0.0, cos_sim)
    except Exception:
        return 0.0


def aggregate_nl_metrics(results: List[NLEvalResult]) -> Dict[str, float]:
    """
    NL評価結果を集計

    Args:
        results: NLEvalResultのリスト

    Returns:
        dict: 集計されたメトリクス
    """
    if not results:
        return {}

    total = len(results)
    errors = sum(1 for r in results if r.error)

    return {
        "total": total,
        "errors": errors,
        "rouge_l_recall": sum(r.rouge_l_r for r in results) / total * 100,
        "rouge_l_precision": sum(r.rouge_l_p for r in results) / total * 100,
        "rouge_l_f1": sum(r.rouge_l_f for r in results) / total * 100,
        "embedding_similarity": sum(r.embedding_similarity for r in results) / total * 100,
        "avg_latency_ms": sum(r.latency_ms for r in results) / total,
    }
