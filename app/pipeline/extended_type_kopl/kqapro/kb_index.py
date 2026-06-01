"""KB-aware retrieval for KQA-Pro JSONL KoPL generation.

The JSONL Phase 1 PoC showed that the LLM's main failure mode is
hallucinating entity / concept / attribute / predicate names that do not
match the KQA-Pro KB verbatim. ``KBIndex`` precomputes lookup tables
from ``kb.json`` and ``retrieve_hints`` returns a short list of the
actual KB strings that are plausibly referenced by a question. The
caller injects those hints into the Phase 1 prompt so the LLM only has
to copy them.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Extremely common English stop words we want to keep OUT of the token
# overlap scoring. Removing them is critical: without this every hit for
# attribute "the" or predicate "of" would drown the real signals.
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "its", "many",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "will",
    "with", "you", "your",
})

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(s)]


def _content_tokens(s: str) -> List[str]:
    return [t for t in _tokenize(s) if t not in _STOP_WORDS and len(t) > 1]


@dataclass
class KBIndex:
    """Precomputed lookups over a KQA-Pro kb.json."""

    entity_id_to_name: Dict[str, str] = field(default_factory=dict)
    concept_id_to_name: Dict[str, str] = field(default_factory=dict)
    # Canonical name (case-preserving) keyed by lowercase
    entity_names: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    concept_names: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    attribute_keys: Set[str] = field(default_factory=set)
    predicates: Set[str] = field(default_factory=set)
    qualifier_keys: Set[str] = field(default_factory=set)
    # Reverse token index: token → set of canonical names containing it.
    _entity_tok_idx: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _concept_tok_idx: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _attr_tok_idx: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _pred_tok_idx: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def from_kb_json(cls, kb_path: str = "/app/data/kqapro/kb.json") -> "KBIndex":
        logger.info("loading %s ...", kb_path)
        kb = json.load(open(kb_path))
        idx = cls()

        for eid, ent in kb["entities"].items():
            name = " ".join(ent["name"].split())  # normalize whitespace
            idx.entity_id_to_name[eid] = name
            idx.entity_names[name.lower()].append(name)
            for tok in _content_tokens(name):
                idx._entity_tok_idx[tok].add(name)
            for attr in ent.get("attributes", []):
                idx.attribute_keys.add(attr["key"])
                for tok in _content_tokens(attr["key"]):
                    idx._attr_tok_idx[tok].add(attr["key"])
                for qk in attr.get("qualifiers", {}):
                    idx.qualifier_keys.add(qk)
            for rel in ent.get("relations", []):
                idx.predicates.add(rel["predicate"])
                for tok in _content_tokens(rel["predicate"]):
                    idx._pred_tok_idx[tok].add(rel["predicate"])
                for qk in rel.get("qualifiers", {}):
                    idx.qualifier_keys.add(qk)

        for cid, con in kb["concepts"].items():
            name = " ".join(con["name"].split())
            idx.concept_id_to_name[cid] = name
            idx.concept_names[name.lower()].append(name)
            for tok in _content_tokens(name):
                idx._concept_tok_idx[tok].add(name)

        logger.info(
            "KBIndex built: %d entities, %d concepts, %d attr keys, %d predicates, %d qual keys",
            sum(len(v) for v in idx.entity_names.values()),
            sum(len(v) for v in idx.concept_names.values()),
            len(idx.attribute_keys),
            len(idx.predicates),
            len(idx.qualifier_keys),
        )
        return idx

    # ------------------------------------------------------------------
    # Per-question retrieval
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        question_tokens: Set[str],
        candidate_names: Set[str],
        limit: int,
    ) -> List[Tuple[str, float]]:
        """Score candidate strings by token-overlap ratio."""
        scored: List[Tuple[str, float]] = []
        for name in candidate_names:
            name_tokens = set(_content_tokens(name))
            if not name_tokens:
                continue
            overlap = len(question_tokens & name_tokens)
            if overlap == 0:
                continue
            # Ratio of the candidate's tokens that appear in the question —
            # prefers names whose ALL tokens are mentioned (e.g. "cast
            # member" when q has both "cast" and "member"), not names with
            # incidental single-word overlap (e.g. "cast" alone).
            score = overlap / len(name_tokens)
            scored.append((name, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def _exact_substring_entities(self, question: str, limit: int) -> List[str]:
        """Entities whose lowercase name occurs verbatim in the question."""
        q = question.lower()
        hits: List[Tuple[str, int]] = []
        # Bounded scan: only entities that share at least one content token
        # with the question.
        q_tokens = set(_content_tokens(question))
        candidates: Set[str] = set()
        for tok in q_tokens:
            candidates.update(self._entity_tok_idx.get(tok, ()))
        for name in candidates:
            low = name.lower()
            if low in q:
                hits.append((name, len(low)))
        hits.sort(key=lambda x: -x[1])  # longest match wins
        return [n for n, _ in hits[:limit]]

    def retrieve_hints(self, question: str, limit: int = 10) -> Dict[str, List[str]]:
        """Return a short list of KB strings that plausibly match the question."""
        q_tokens = set(_content_tokens(question))

        entities = self._exact_substring_entities(question, limit)

        concept_candidates: Set[str] = set()
        for tok in q_tokens:
            concept_candidates.update(self._concept_tok_idx.get(tok, ()))
        concepts = [n for n, _ in self._score_candidates(q_tokens, concept_candidates, limit)]

        attr_candidates: Set[str] = set()
        for tok in q_tokens:
            attr_candidates.update(self._attr_tok_idx.get(tok, ()))
        attr_keys = [n for n, _ in self._score_candidates(q_tokens, attr_candidates, limit * 2)]

        pred_candidates: Set[str] = set()
        for tok in q_tokens:
            pred_candidates.update(self._pred_tok_idx.get(tok, ()))
        predicates = [n for n, _ in self._score_candidates(q_tokens, pred_candidates, limit * 2)]

        return {
            "entities": entities,
            "concepts": concepts,
            "attr_keys": attr_keys,
            "predicates": predicates,
        }


_index_singleton: Optional[KBIndex] = None


def get_kb_index(kb_path: str = "/app/data/kqapro/kb.json") -> KBIndex:
    global _index_singleton
    if _index_singleton is None:
        _index_singleton = KBIndex.from_kb_json(kb_path)
    return _index_singleton
