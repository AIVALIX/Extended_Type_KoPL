"""
PcQA-specific entity resolution: CancerCell compound name matching

PcQA KGのCancerCellノードは "GENE-MUTATION-CANCER" のような複合名を持つ。
質問文に遺伝子名+癌種名が含まれる場合、CONTAINS検索でCancerCellノードを特定し、
正しいスキーマパス（CancerCell → Drug等）を選択できるようにする。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class PcQAEntityPairResponse(BaseModel):
    """LLMによる2エンティティ抽出結果"""
    entity_names: List[str] = Field(
        description=(
            "Entity names extracted from the question (1 or 2). "
            "If the question mentions BOTH a gene/mutation AND a specific cancer type, "
            "extract BOTH names. Otherwise extract only the main entity."
        )
    )


# CancerCell関連キーワード（1エンティティでもCancerCell解決を試みるトリガー）
CANCERCELL_KEYWORDS = {
    "resistant", "resistance",
    "sensitive", "sensitivity", "sensible",
    "efficacy",
}


def resolve_pcqa_compound_entity(
    llm,
    finder,
    question: str,
    entity_name: str,
    entity_type: Optional[str] = None,
) -> Optional[Tuple[str, str, List[str]]]:
    """PcQA複合エンティティ解決

    質問文から2つのエンティティ（遺伝子+癌種）を抽出し、
    CancerCellノードのCONTAINS検索でマッチングする。

    Args:
        llm: LLMインスタンス
        finder: GraphPathFinder
        question: 質問文
        entity_name: 既知エンティティ名（データセットから）
        entity_type: エンティティタイプ（"Genesymbol", "Fusion"等）

    Returns:
        (compound_name, entity_type, all_compound_names) or None
        compound_name: 最初のマッチした複合名
        entity_type: タイプ（例: "CancerCell"）
        all_compound_names: マッチした全複合名リスト
    """
    # Genesymbol/Fusion以外は対象外
    if entity_type and entity_type not in ("Genesymbol", "Fusion"):
        return None

    # Step 1: LLMで2エンティティ抽出を試みる
    two_entities = _extract_entity_pair(llm, question, entity_name)

    if two_entities and len(two_entities) >= 2:
        # 2エンティティ抽出成功 → CONTAINS検索
        compound = _search_compound(finder, two_entities)
        if compound:
            return compound

    # Step 2: キーワードベースのフォールバック
    # resistance/sensitivity/efficacy キーワードがある場合、
    # エンティティ名のみでCancerCell CONTAINS検索
    q_lower = question.lower()
    if any(kw in q_lower for kw in CANCERCELL_KEYWORDS):
        compound = _search_compound(finder, [entity_name], label="CancerCell")
        if compound:
            return compound

    # Step 3: "treat [cancer] with [gene]" パターン検出
    # "How to treat X with Y" where X is cancer and Y is gene
    if "treat" in q_lower:
        compound = _search_compound(finder, [entity_name], label="CancerCell")
        if compound:
            return compound

    return None


def _extract_entity_pair(
    llm, question: str, known_entity: str
) -> Optional[List[str]]:
    """LLMで質問文から2エンティティを抽出"""
    prompt = f"""Extract entity names from this biomedical question.

Question: {question}
Known entity: {known_entity}

Instructions:
- If the question mentions BOTH a gene/mutation AND a specific cancer/tumor type, return BOTH names.
- If the question mentions only ONE entity or a generic term like "cancers", return only that entity.
- Return the exact names as they appear in the question.

Examples:
- "How to treat non-small cell lung cancer with EGFR?" → ["EGFR", "non-small cell lung cancer"]
- "Which drugs are ALK in giant cell carcinoma of the lung resistant to?" → ["ALK", "giant cell carcinoma of the lung"]
- "Which drugs are KRAS in colon cancer sensitive to?" → ["KRAS", "colon cancer"]
- "What drugs can treat cancers with TERT mutations?" → ["TERT"]
- "Which genetic mutations may be resistant to sorafenib?" → ["sorafenib"]
- "How to treat melanoma with NRAS?" → ["NRAS", "melanoma"]

Return only the entity names list."""

    try:
        llm_with_output = llm.with_structured_output(PcQAEntityPairResponse)
        result = llm_with_output.invoke(prompt)
        if result and result.entity_names:
            return result.entity_names
    except Exception:
        pass
    return None


def _search_compound(
    finder,
    terms: List[str],
    label: Optional[str] = None,
) -> Optional[Tuple[str, str, List[str]]]:
    """CONTAINS検索でCancerCell複合名を探す"""

    # 2エンティティの場合: Cancer→CancerCell経由で検索（英語Cancer名→中国語CancerCell名）
    if len(terms) >= 2:
        # 遺伝子名（短い方）とCancer名（長い方）を推定
        sorted_terms = sorted(terms, key=len)
        gene_term = sorted_terms[0]  # 通常短い方が遺伝子名
        cancer_term = sorted_terms[-1]  # 通常長い方がCancer名
        results = finder.find_compound_entity_via_cancer(gene_term, cancer_term)
        if results:
            cancercell_names = [name for name, labels in results if "CancerCell" in labels]
            if cancercell_names:
                return (cancercell_names[0], "CancerCell", cancercell_names)

    # フォールバック: 直接CONTAINS検索
    results = finder.find_compound_entity(terms, label=label or "CancerCell")

    if not results and label:
        results = finder.find_compound_entity(terms)

    if not results:
        return None

    # CancerCellを優先
    cancercell_names = []
    for name, labels in results:
        if "CancerCell" in labels:
            cancercell_names.append(name)

    if cancercell_names:
        return (cancercell_names[0], "CancerCell", cancercell_names)

    # CancerCell以外の場合
    first_name, first_labels = results[0]
    filtered = [l for l in first_labels if l not in ("_Entity", "Entity")]
    etype = filtered[0] if filtered else "Unknown"
    return (first_name, etype, [first_name])
