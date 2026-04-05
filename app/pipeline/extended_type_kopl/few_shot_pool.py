"""
Retrieval-based Few-shot Example Pool

質問分布に応じてFew-shot例示を自動選択するretrieval-based prompting。
MMR (Maximal Marginal Relevance) で類似かつ多様な例を選択する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FewShotExample:
    """Few-shot例"""

    question: str
    operations: List[Dict[str, Any]]  # [{src_type, tgt_type, relation, anchor_name?}]
    final_operation: str  # "relate" or "intersection"
    entity_name: str = ""
    entity_type: str = ""
    target_type: str = ""
    hop_count: int = 1
    operation_type: str = "relate"  # "relate" or "intersection"
    notes: str = ""  # domain-specific notes to include after the example
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_prompt_text(self, index: int) -> str:
        """プロンプト用テキスト形式に変換"""
        # Describe hop type
        if self.operation_type == "intersection":
            hop_desc = "Intersection"
        else:
            hop_desc = f"{self.hop_count}-hop"

        ops_str = json.dumps(self.operations, ensure_ascii=False)
        text = f'{index}. {hop_desc}: "{self.question}"\n'
        text += f"operations: {ops_str}\n"
        text += f'final_operation: "{self.final_operation}"'
        return text


class FewShotPool:
    """Few-shot例プール（MMRベース選択）

    質問をベクトル化し、コサイン類似度 + MMRで多様な例を選択する。
    プールは小規模（~20-50例）なのでFAISSなしで十分。
    """

    def __init__(self, kg_type: str, embedding_model: str = "text-embedding-3-small"):
        self.kg_type = kg_type
        self.embedding_model_name = embedding_model
        self._embeddings = None  # lazy init
        self.examples: List[FewShotExample] = []
        self._embeddings_built = False

    def _get_embeddings(self):
        """Lazy init for OpenAIEmbeddings"""
        if self._embeddings is None:
            import os

            from core.config import get_settings
            from langchain_openai import OpenAIEmbeddings

            if not os.getenv("OPENAI_API_KEY"):
                settings = get_settings()
                os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

            self._embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        return self._embeddings

    def load_pool(self, pool_path: str) -> None:
        """JSONファイルからプールを読み込み"""
        path = Path(pool_path)
        if not path.exists():
            raise FileNotFoundError(f"Few-shot pool not found: {pool_path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        examples_data = data if isinstance(data, list) else data.get("examples", [])
        self.examples = []
        for item in examples_data:
            self.examples.append(
                FewShotExample(
                    question=item["question"],
                    operations=item["operations"],
                    final_operation=item.get("final_operation", "relate"),
                    entity_name=item.get("entity_name", ""),
                    entity_type=item.get("entity_type", ""),
                    target_type=item.get("target_type", ""),
                    hop_count=item.get("hop_count", 1),
                    operation_type=item.get("operation_type", "relate"),
                    notes=item.get("notes", ""),
                )
            )
        self._embeddings_built = False

    def build_embeddings(self) -> None:
        """全例の質問をベクトル化"""
        if not self.examples:
            return
        if self._embeddings_built:
            return

        embeddings = self._get_embeddings()
        questions = [ex.question for ex in self.examples]
        vecs = embeddings.embed_documents(questions)
        for i, ex in enumerate(self.examples):
            ex.embedding = np.array(vecs[i], dtype=np.float32)

        self._embeddings_built = True

    def select_examples(
        self,
        query: str,
        k: int = 3,
        diversity_lambda: float = 0.3,
    ) -> List[FewShotExample]:
        """MMRベースで多様な類似例を選択

        Args:
            query: 入力質問
            k: 選択する例数
            diversity_lambda: 多様性パラメータ (0=max diversity, 1=max similarity)

        Returns:
            選択されたFewShotExampleのリスト
        """
        if not self.examples:
            return []

        self.build_embeddings()

        embeddings = self._get_embeddings()
        query_vec = np.array(
            embeddings.embed_query(query), dtype=np.float32
        )

        candidate_vecs = np.array(
            [ex.embedding for ex in self.examples], dtype=np.float32
        )

        selected_indices = self._mmr_select(
            query_vec, candidate_vecs, min(k, len(self.examples)), diversity_lambda
        )

        return [self.examples[i] for i in selected_indices]

    def _mmr_select(
        self,
        query_vec: np.ndarray,
        candidate_vecs: np.ndarray,
        k: int,
        lambda_param: float,
    ) -> List[int]:
        """Maximal Marginal Relevance selection

        score = lambda * sim(query, doc) - (1-lambda) * max_sim(doc, selected_docs)

        Uses cosine similarity.
        """
        # Compute cosine similarities between query and all candidates
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-10
        candidate_norms = candidate_vecs / norms
        query_sims = candidate_norms @ query_norm  # (N,)

        selected: List[int] = []
        remaining = set(range(len(candidate_vecs)))

        for _ in range(k):
            if not remaining:
                break

            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                # Relevance term
                relevance = query_sims[idx]

                # Diversity term: max similarity to already selected docs
                if selected:
                    selected_vecs = candidate_norms[selected]  # (S, dim)
                    sims_to_selected = selected_vecs @ candidate_norms[idx]  # (S,)
                    max_sim_to_selected = float(np.max(sims_to_selected))
                else:
                    max_sim_to_selected = 0.0

                score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected

    def format_examples_prompt(
        self,
        query: str,
        k: int = 3,
        diversity_lambda: float = 0.3,
        domain_label: str = "",
    ) -> str:
        """クエリに基づいてFew-shot例をプロンプト文字列として返す

        Args:
            query: 入力質問
            k: 選択する例数
            diversity_lambda: 多様性パラメータ
            domain_label: ドメイン名 (e.g., "PrimeKGQA (Biomedical domain)")

        Returns:
            プロンプト用テキスト
        """
        examples = self.select_examples(query, k, diversity_lambda)
        if not examples:
            return ""

        header = f"Examples for {domain_label}:" if domain_label else "Examples:"
        parts = [header, ""]

        for i, ex in enumerate(examples, 1):
            parts.append(ex.to_prompt_text(i))
            parts.append("")

        # Collect unique notes from selected examples
        notes = []
        for ex in examples:
            if ex.notes and ex.notes not in notes:
                notes.append(ex.notes)

        if notes:
            parts.append("IMPORTANT:")
            for note in notes:
                parts.append(f"- {note}")

        return "\n".join(parts)
