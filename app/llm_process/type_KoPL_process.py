from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from langchain.chat_models import init_chat_model

from core.config import BASEMODEL, get_settings
from prompts.type_KoPL_fewshot import FEW_SHOT_MetaQA, FEW_SHOT_PrimeKGQA


SYSTEM_PROMPT = """
あなたは自然言語の質問を、型ベースの論理プログラム「Type KoPL」に変換するエキスパートです。
KGのスキーマ構造（型と関係）に基づき、論理クエリを組み立ててください。

【関数定義】
- Findanchor(anchor_type: str, entity_name: str) -> entities: 
  指定された型(anchor_type)と名前(entity_name)を持つ実体を起点として取得。
  ※ entity_nameは、質問文中の異なる実体を区別するために必須。
  
- FindTypeRelate(expression, anchor_type: str, target_type: str, relation_name: str) -> entities: 
  現在の実体集合から、指定した関係を通じて別の型へ遷移する。

- And(exp1, exp2) -> entities: 2つの集合の共通部分(Intersection)。
- Or(exp1, exp2) -> entities: 2つの集合の和集合(Union)。
- Stop(expression) -> result: プログラムを終了し回答を出力。

【厳守ルール】
1. 質問文に登場する具体的な実体名（例: "Silver-Russell syndrome"）を必ず Findanchor の entity_name に入れてください。
2. 同じ型でも名前が異なる実体は、必ず別の変数（exp1, exp2...）として定義してください。
3. ターゲットとなる型が異なる場合は Or() を、条件の絞り込みには And() を使用してください。
4. コードを生成する前に、まず「どの型から開始し、どの関係を辿り、どの型に着地するか」という思考プロセス（Thinking Process）を簡潔に記述してください。
5. FindTypeRelateの第1引数(expression)には、必ず直前のステップで定義した変数を渡して
""".strip()


def _ensure_openai_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    settings = get_settings()
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def generate_type_kopl(
    *,
    question: str,
    fewshot: str,
    model: str = BASEMODEL,
    temperature: float = 0.0,
) -> str:
    """Generate Type KoPL code from a question.

    This function MUST take `question` and `fewshot` as arguments.
    """

    q = (question or "").strip()
    if not q:
        return ""

    _ensure_openai_api_key()

    llm = init_chat_model(model, model_provider="openai", temperature=temperature)
    prompt = f"""
{SYSTEM_PROMPT}

{fewshot}

Question: 「{q}"
Code:
""".strip()
    res = llm.invoke(prompt)

    content = getattr(res, "content", "")
    return (content or "").strip()


def run_type_kopl_trial(
    file_path: str | Path,
    *,
    num_samples: int = 8,
    fewshot: str = FEW_SHOT_MetaQA,
    model: str = BASEMODEL,
    temperature: float = 0.0,
) -> None:
    path = Path(file_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break

                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                question = str(data.get("question", "") or "")

                print(f"\n--- 試行 {i + 1} ---")
                print(f"質問: {question}")

                generated_code = generate_type_kopl(
                    question=question,
                    fewshot=fewshot,
                    model=model,
                    temperature=temperature,
                )
                print(f"生成された Type KoPL:\n{generated_code}")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {path}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    result = generate_type_kopl(
        question=" the films that share directors with the film [Black Snake Moan] were in which genres",
        fewshot=FEW_SHOT_MetaQA,
    )
    print(result)
