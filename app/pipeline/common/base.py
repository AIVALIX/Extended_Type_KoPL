"""
パイプライン基底クラス

全パイプラインが継承する抽象基底クラス
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineResult:
    """パイプライン実行結果の基底クラス"""
    question: str
    entity_name: Optional[str]
    answer_entities: List[str] = field(default_factory=list)
    processing_log: List[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        """ログメッセージを追加"""
        self.processing_log.append(message)


class BasePipeline(ABC):
    """パイプライン基底クラス"""

    @abstractmethod
    def run(
        self,
        question: str,
        entity_name: Optional[str] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        パイプラインを実行

        Args:
            question: 自然言語の質問
            entity_name: アンカーエンティティ名（オプション）
            **kwargs: パイプライン固有の追加引数

        Returns:
            PipelineResult: 実行結果
        """
        pass

    @abstractmethod
    def extract_relations(self, result: PipelineResult) -> Optional[List[str]]:
        """
        結果からリレーションパスを抽出

        Args:
            result: パイプライン実行結果

        Returns:
            Optional[List[str]]: リレーションのリスト（順序付き）
        """
        pass

    def extract_entities(self, result: PipelineResult) -> List[str]:
        """
        結果からエンティティを抽出

        Args:
            result: パイプライン実行結果

        Returns:
            List[str]: 回答エンティティのリスト
        """
        return result.answer_entities
