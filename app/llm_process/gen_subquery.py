import os
from typing import Any, List, get_args, Optional

from langchain.chat_models import init_chat_model
from models.model import (
    MetaQAType,
    MetaSubQueryResponse,
    SubQueryResponse,
)
from core.config import get_settings
from prompts.shex_prompt import shex_prompt


class SubQueryRunnable:
    """
    Runnable that splits a Japanese NL question into minimal sub-queries.
    If sub-queries depend on previous answers, insert {answer[n]} placeholders.
    Returns a `Split_Response` pydantic model.
    """

    def __init__(self, chat_model: Any):
        self.model = chat_model.with_structured_output(SubQueryResponse)

    def invoke(self, sentence: str, hop: Optional[int] = None) -> SubQueryResponse:
        """Return Split_Response for *sentence* with placeholders where needed."""

        # hop制約を条件付きで追加
        hop_constraint = ""
        if hop is not None:
            hop_constraint = (
                f"• You MUST produce exactly **{hop}** sub-queries, no more, no less.\n"
            )
        else:
            hop_constraint = ""

        prompt = f"""
You are an expert assistant that decomposes complex questions into a series of minimal, logical sub‑queries. Follow these rules strictly:

Few‑shot examples:

Question: In which year was the film "Inception" released?
Number of sub-queries to produce: 1
{{"sub_queries":[
  {{"query":"In which year was the film \\"Inception\\" released?"}}
]}}

Question: Which actor played the character "Neo" in "The Matrix", and in which year was that actor born?  
Number of sub-queries to produce: 2  
{{"sub_queries":[
  {{"query":"Who played the character \"Neo\" in the film \"The Matrix\"?"}},
  {{"query":"In what year was {{answer[0]}} born?"}},
]}}

Question: Which actor played the character "Neo" in "The Matrix", in which year was that actor born, and which other actor in "The Matrix" was also born in that same year?
Number of sub-queries to produce: 3
{{"sub_queries":[
{{"query":"Who played the character \"Neo\" in the film \"The Matrix\"?"}},
{{"query":"In what year was {{answer[0]}} born?"}},
{{"query":"Which actor in the film \"The Matrix\" was born in {{answer[1]}}?"}}
]}}

{{"query":"{sentence}"}}

Important rules:
{hop_constraint}• Each sub‑query must be independently answerable.  
• Use placeholders `{{answer[n]}}` to reference previous answers.  
• Avoid pronouns like "it" or "that"; make all references explicit.  
• Produce the **minimal** number of sub-queries required by the question.  
• Each sub‑query must express exactly one condition.  
• Final output must be valid JSON, e.g.:  
  {{\"sub_queries\":[{{\"query\":\"...\"}}, …]}}
"""

        return self.model.invoke(prompt)


class MetaSubQueryRunnable:
    """
    Meta runnable that handles the sub-query splitting process.
    """

    def __init__(self, chat_model: Any):
        self.model = chat_model.with_structured_output(MetaSubQueryResponse)

    def invoke(self, sentence: str, hop: Optional[int] = None) -> MetaSubQueryResponse:
        """Return SubQuery_With_Meta for *sentence*."""

        # Literalの値を安全に取得する方法
        allowed_types = get_args(MetaQAType)

        # hop制約を条件付きで追加
        hop_constraint = ""
        if hop is not None:
            hop_constraint = (
                f"• You MUST produce exactly **{hop}** sub-queries, no more, no less.\n"
            )
        else:
            hop_constraint = ""

        prompt = f"""
You are an expert assistant that decomposes complex questions into a series of minimal, logical sub‑queries. Follow these rules strictly:

Few‑shot examples:
Question: In which year was the film "Inception" released?
Number of sub-queries to produce: 1
{{"sub_queries":[
  {{"query":"In which year was the film \\"Inception\\" released?","answer_type":"Date"}}
]}}

Question: Which actor played the character "Neo" in "The Matrix", and in which year was that actor born?  
Number of sub-queries to produce: 2  
{{"sub_queries":[
  {{"query":"Who played the character \"Neo\" in the film \"The Matrix\"?","answer_type":"Person"}},
  {{"query":"In what year was {{answer[0]}} born?","answer_type":"Date"}},
]}}

Question: Which actor played the character "Neo" in "The Matrix", in which year was that actor born, and which other actor in "The Matrix" was also born in that same year?
Number of sub-queries to produce: 3
{{"sub_queries":[
{{"query":"Who played the character \"Neo\" in the film \"The Matrix\"?","answer_type":"Person"}},
{{"query":"In what year was {{answer[0]}} born?","answer_type":"Date"}},
{{"query":"Which actor in the film \"The Matrix\" was born in {{answer[1]}}?","answer_type":"Person"}}
]}}

{{"query":"{sentence}","answer_type":""}}
There is shape expression defined as below:
{shex_prompt}
Important rules:
{hop_constraint}• Each sub‑query must be independently answerable.  
• Use placeholders `{{answer[n]}}` to reference previous answers.  
• Avoid pronouns like "it" or "that"; make all references explicit.  
• Produce the **minimal** number of sub-queries required by the question.   
• Each sub‑query must express exactly one condition.  
• answer_type must be one of: {', '.join(allowed_types)}.
• Final output must be valid JSON, e.g.:  
  {{\"sub_queries\":[{{\"query\":\"...\",\"answer_type\":\"...\"}}, …]}}
"""

        return self.model.invoke(prompt)


class TypePathMetaSubQuery:
    """
    type pathを入力として、サブクエリ分割を行う
    指定されたtype pathの順序に従って、各ステップの期待される回答タイプでサブクエリを生成する
    """

    def __init__(self, chat_model: Any):
        self.model = chat_model.with_structured_output(SubQueryResponse)

    def build_prompt_typepath_minimal(
        self, sentence: str, type_path: list[str], hop: int | None = None
    ) -> str:
        hop = len(type_path) if hop is None else hop
        type_path_arrow = " → ".join(type_path)
        step_lines = "\n".join(
            [f"- Step {i+1}: expects {t}" for i, t in enumerate(type_path)]
        )

        hop_constraint = (
            f"• You MUST produce exactly **{hop}** sub-queries, no more, no less.\n"
        )

        prompt = f"""
    You are an expert assistant that decomposes complex questions into a series of minimal, logical sub-queries. Follow these rules strictly:

    Few-shot examples:

    Question: Who directed "Spirited Away" and in which year was their first film released?
    Required type path (hop=2): Person → Date
    {{"sub_queries":[
    {{"query":"Who directed the film \\"Spirited Away\\"?"}},
    {{"query":"In which year was {{answer[0]}}'s first directed film released?"}}
    ]}}

    Question: List books written by authors born after 1980 that have more than 500 pages.
    Required type path (hop=3): Person → CreativeWork → CreativeWork
    {{"sub_queries":[
    {{"query":"Which authors were born after 1980?"}},
    {{"query":"Which books were written by {{answer[0]}}?"}},
    {{"query":"Which of {{answer[1]}} have more than 500 pages?"}}
    ]}}

    Question: Who directed the movies starred by John Krasinski?
    Required type path (hop=2): CreativeWork → Person
    {{"sub_queries":[
    {{"query":"What movies did John Krasinski star in?"}},
    {{"query":"Who directed {{answer[0]}}?"}}
    ]}}

    Question: Describe “Radioland Murders” in a few words.
    Required type path (hop=1): Text
    {{"sub_queries":[
    {{"query":"What is the plot summary of Radioland Murders?"}}
    ]}}

    Now decompose the following question:
    {{"query":"{sentence}"}}

    Required type path (hop={hop}): {type_path_arrow}

    ⚠️ Important rules:
    {hop_constraint}• **Follow the required type path in order.** At step *i*, phrase the sub-query so its **answer is an instance of** `type_path[i]`. Do not add, drop, or reorder steps.
    • Each sub-query must be independently answerable.  
    • Use placeholders `{{{{answer[n]}}}}` to reference previous answers.  
    • Avoid pronouns like "it" or "that"; make all references explicit.  
    • Each sub-query must express exactly one condition.  
    • Final output must be valid JSON:
    {{ "sub_queries":[{{"query":"..."}}, …] }}

    Type-path step reminder:
    {step_lines}
    """.strip()
        return prompt

    def invoke(
        self, type_path: List[str], sentence: str, hop: Optional[int] = None
    ) -> SubQueryResponse:
        """Return SubQuery_With_Meta for *type_path*.

        Parameters
        ----------
        type_path : List[str]
            期待される回答タイプの順序 (例: ["Person", "CreativeWork", "Date"])
        sentence : str
            分割する質問文
        hop : Optional[int]
            hop数の指定。Noneの場合はtype_pathの長さを使用
        """
        prompt = self.build_prompt_typepath_minimal(sentence, type_path, hop)
        return self.model.invoke(prompt)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) API キーをセット
    if not os.getenv("OPENAI_API_KEY"):
        settings = get_settings()
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    # 2) LLM 初期化
    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)

    # 3) SubQueryRunnable 実行（hop指定あり）
    splitter = SubQueryRunnable(llm)
    text = "which person directed the movies starred by John Krasinski"
    response_with_hop = splitter.invoke(text, hop=2)
    print("SubQueryRunnable Response (with hop=2):")
    print(response_with_hop)

    # 4) SubQueryRunnable 実行（hop指定なし）
    response_no_hop = splitter.invoke(text)
    print("\nSubQueryRunnable Response (no hop constraint):")
    print(response_no_hop)

    # 5) MetaSubQueryRunnable 実行（hop指定あり）
    meta_splitter = MetaSubQueryRunnable(llm)
    meta_response_with_hop = meta_splitter.invoke(text, hop=2)
    print("\nMetaSubQueryRunnable Response (with hop=2):")
    print(meta_response_with_hop)

    # 6) MetaSubQueryRunnable 実行（hop指定なし）
    meta_response_no_hop = meta_splitter.invoke(text)
    print("\nMetaSubQueryRunnable Response (no hop constraint):")
    print(meta_response_no_hop)

    # 7) TypePathMetaSubQuery 実行
    print("\n" + "=" * 80)
    print("TypePathMetaSubQuery Tests")
    print("=" * 80)

    type_path_splitter = TypePathMetaSubQuery(llm)

    # テストケース1: Person → CreativeWork
    type_path_1 = ["CreativeWork", "Person"]
    response_1 = type_path_splitter.invoke(type_path_1, text)
    print(f"\nTypePathMetaSubQuery Response (type_path: {type_path_1}):")
    print(response_1)

    # テストケース2: Person → CreativeWork → Date
    text_2 = "when were the films directed by actors who worked with Tom Hanks released"
    type_path_2 = ["Person", "CreativeWork", "Date"]
    response_2 = type_path_splitter.invoke(type_path_2, text_2)
    print(f"\nTypePathMetaSubQuery Response (type_path: {type_path_2}):")
    print(response_2)

    # テストケース3: CreativeWork → Person → CreativeWork
    text_3 = "what other movies were made by the director of Inception"
    type_path_3 = ["Person", "CreativeWork"]
    response_3 = type_path_splitter.invoke(type_path_3, text_3)
    print(f"\nTypePathMetaSubQuery Response (type_path: {type_path_3}):")
    print(response_3)

    # テストケース4: 複雑なtype path
    text_4 = "which actors starred in movies directed by people born in the same year as Steven Spielberg"
    type_path_4 = ["Person", "Date", "Person", "CreativeWork", "Person"]
    response_4 = type_path_splitter.invoke(type_path_4, text_4)
    print(f"\nTypePathMetaSubQuery Response (type_path: {type_path_4}):")
    print(response_4)

# python llm_process/gen_subquery.py
