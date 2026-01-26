"""
共通モジュール

パイプライン間で共有されるベースクラス、評価メトリクス、デバッグユーティリティ等
"""

from pipeline.common.base import BasePipeline, PipelineResult
from pipeline.common.kg_config import (
    KGType,
    Neo4jConfig,
    DatasetConfig,
    KGConfig,
    get_default_config,
)
from pipeline.common.eval_metrics import (
    PipelineOutput,
    EvalResult,
    SetMetrics,
    compute_set_metrics,
    check_path_match,
    aggregate_metrics,
    evaluate_output,
    evaluate_outputs,
    save_pipeline_outputs,
    load_pipeline_outputs,
)
from pipeline.common.debug import (
    load_samples,
    get_gold_info,
    compute_metrics,
    print_header,
    print_metrics,
)

__all__ = [
    # Base
    "BasePipeline",
    "PipelineResult",
    # KG Config
    "KGType",
    "Neo4jConfig",
    "DatasetConfig",
    "KGConfig",
    "get_default_config",
    # Eval metrics
    "PipelineOutput",
    "EvalResult",
    "SetMetrics",
    "compute_set_metrics",
    "check_path_match",
    "aggregate_metrics",
    "evaluate_output",
    "evaluate_outputs",
    "save_pipeline_outputs",
    "load_pipeline_outputs",
    # Debug
    "load_samples",
    "get_gold_info",
    "compute_metrics",
    "print_header",
    "print_metrics",
]
