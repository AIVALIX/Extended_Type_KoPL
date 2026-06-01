"""
Direct QA Pipeline

KGを使わず、LLMに直接質問するベースライン。
KG使用の効果を示すための対照実験。
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from core.config import BASEMODEL, LLM_API_BASE, get_settings
from pipeline.direct_qa.models import DirectQAResult


DOMAIN_HINTS = {
    "primekgqa": (
        "This is a biomedical question about drugs, diseases, genes/proteins, "
        "biological processes, pathways, anatomical structures, or phenotypes."
    ),
    "metaqa": (
        "This is a question about movies, including actors, directors, writers, "
        "genres, languages, release years, and production companies."
    ),
    "pcqa": (
        "This is a question about pan-cancer genomics, including cancers, drugs, "
        "genes, mutations (SNV/Fusion), and clinical trials."
    ),
}


def _parse_entity_list(text: str) -> List[str]:
    """LLM出力からエンティティ名リストをパース。{entity} 形式を優先。"""
    entities = []
    # {entity1, entity2} 形式
    braced = re.findall(r"\{([^}]+)\}", text)
    if braced:
        for item in braced:
            for ent in item.split(","):
                ent = ent.strip().strip("'").strip('"')
                if ent:
                    entities.append(ent)
        return entities

    # 改行区切りフォールバック
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line.lower().startswith(("based on", "the answer", "therefore")):
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", line)
            entities.extend(quoted)
            continue
        line = line.strip().strip('"').strip("'")
        if line:
            entities.append(line)
    return entities


class DirectQAPipeline:
    """Direct QA パイプライン（KGなしベースライン）"""

    def __init__(
        self,
        model: str = BASEMODEL,
        kg_type: str = "primekgqa",
        **kwargs,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model

        llm_kwargs = {"model_provider": "openai", "temperature": 0, "max_tokens": 4096}
        api_base = os.getenv("LLM_API_BASE", "") or LLM_API_BASE or None
        if api_base:
            llm_kwargs["base_url"] = api_base
            llm_kwargs["api_key"] = "sk-local"
        self.llm = init_chat_model(model, **llm_kwargs)
        self.kg_type = kg_type

    def run(
        self, question: str, entity_name: Optional[str] = None, **kwargs
    ) -> DirectQAResult:
        log = []
        log.append(f"Question: {question}")
        if entity_name:
            log.append(f"Entity hint: {entity_name}")

        domain_hint = DOMAIN_HINTS.get(self.kg_type, "")
        prompt = (
            f"{domain_hint}\n\n"
            f"Answer the following question. "
            f"List the answer entity names inside curly braces like {{entity1, entity2}}.\n\n"
            f"Q: {question}\n"
            f"A:"
        )

        try:
            response = self.llm.invoke(prompt)
            raw_text = response.content.strip()
            log.append(f"LLM raw: {raw_text[:200]}")
            answer_entities = _parse_entity_list(raw_text)
            log.append(f"Parsed {len(answer_entities)} entities")
        except Exception as e:
            log.append(f"LLM error: {e}")
            answer_entities = []

        return DirectQAResult(
            question=question,
            entity_name=entity_name,
            answer_entities=answer_entities,
            processing_log=log,
        )
