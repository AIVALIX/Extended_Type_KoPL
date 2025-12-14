import os
from typing import Any, List, Dict

from langchain.chat_models import init_chat_model
from models.model import (
    RerankerResponse,
)
from core.config import get_settings


class PathReranker:
    """パスの再ランキングを行うクラス"""

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, queries: List[str], paths: List[List[str]]) -> Any:
        """
        クエリリストとパスリストを受け取って再ランキングを実行

        Parameters
        ----------
        queries : List[str]
            質問リスト（サブクエリまたは元の質問）
        paths : List[List[str]]
            候補パスのリスト

        Returns
        -------
        RerankerResponse
            再ランキング結果
        """
        # クエリリストをテキスト形式に変換
        if len(queries) == 1:
            query_text = queries[0]
            query_context = f"Question: {query_text}"
        else:
            query_text = " | ".join(queries)
            query_context = f"Multi-step questions: {query_text}"

        # パスリストをテキスト形式に変換
        path_strings = []
        for i, path in enumerate(paths, 1):
            path_str = " -> ".join(path)
            path_strings.append(f"{i}. {path_str}")

        paths_text = "\n".join(path_strings)

        # プロンプトの構築
        prompt = f"""
Given the following question(s) and candidate paths, select the best path that can answer the question(s).

{query_context}

Candidate Paths:
{paths_text}

Please select the path number (1-{len(paths)}) that best answers the question(s).
Consider the semantic relevance and logical flow of each path.

Response format: Return only the number of the selected path.
"""

        try:
            # LLMに送信
            response = self.llm.invoke(prompt)

            # レスポンスから番号を抽出
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # 数字を抽出
            import re

            numbers = re.findall(r"\d+", response_text)

            if numbers:
                selected_index = int(numbers[0]) - 1  # 1-basedから0-basedに変換
                if 0 <= selected_index < len(paths):
                    selected_path = paths[selected_index]

                    # RerankerResponseオブジェクトを作成
                    class RerankerResponse:
                        def __init__(self, reranked_combos):
                            self.reranked_combos = reranked_combos

                    return RerankerResponse(reranked_combos=selected_path)

            # デフォルトで最初のパスを返す
            class RerankerResponse:
                def __init__(self, reranked_combos):
                    self.reranked_combos = reranked_combos

            return RerankerResponse(reranked_combos=paths[0] if paths else [])

        except Exception as e:
            print(f"Error in path reranking: {e}")

            # エラー時はデフォルトで最初のパスを返す
            class RerankerResponse:
                def __init__(self, reranked_combos):
                    self.reranked_combos = reranked_combos

            return RerankerResponse(reranked_combos=paths[0] if paths else [])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) API キーをセット
    if not os.getenv("OPENAI_API_KEY"):
        settings = get_settings()
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    # 2) LLM 初期化
    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)

    # 3) Reranker 初期化
    reranker = PathReranker(llm)

    # 4) サンプルデータ（リレーションのみ）
    question = "映画『新世紀エヴァンゲリオン劇場版 Air/まごころを、君に』の前作を制作した会社はどこでしょうか"
    candidate_relation_paths = [
        ["directed_by"],
        ["previous_series", "production_company"],
        ["production_company"],
        ["sequel_of", "production_company"],
        ["genre", "movies", "production_company"],
    ]

    # 5) リランキング実行
    result = reranker.invoke(question, candidate_relation_paths)
    print("Question:", question)
    print("\nCandidate relation paths:")
    for i, relations in enumerate(candidate_relation_paths, 1):
        print(f"{i}. {relations}")

    print(f"\nSelected relation sequence: {result.reranked_combos}")


# python llm_process/path_reranker.py
