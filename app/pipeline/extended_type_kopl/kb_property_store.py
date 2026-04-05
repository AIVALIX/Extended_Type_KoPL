"""
KQA-Pro KB Property Store

kb.jsonをインメモリにロードし、エンティティ属性の検索・フィルタを提供する。
Neo4jにはname/labelのみ格納されているため、attribute/qualifier操作はこのストアを経由する。

Usage:
    store = KBPropertyStore("data/kqapro/kb.json")
    store.query_attr("Fred Zinnemann", "ISNI")  # -> "0000 0001 0869 4236"
    store.get_available_properties("Fred Zinnemann")  # -> [("ISNI", "0000...", "string"), ...]
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValueClass — ported from KQAPro_Baselines/utils/value_class.py
# ---------------------------------------------------------------------------

class ValueClass:
    """Typed value with comparison semantics for KQA-Pro KB."""

    def __init__(self, vtype: str, value: Any, unit: Optional[str] = None):
        self.type = vtype
        self.value = value
        self.unit = unit

    def is_time(self) -> bool:
        return self.type in ("year", "date")

    def can_compare(self, other: "ValueClass") -> bool:
        if self.type == "string":
            return other.type == "string"
        if self.type == "quantity":
            return other.type == "quantity" and other.unit == self.unit
        # year/date can compare with each other
        return other.type in ("year", "date")

    def contains(self, other: "ValueClass") -> bool:
        if self.type == "year":
            other_val = other.value if other.type == "year" else other.value.year
            return self.value == other_val
        if self.type == "date":
            return other.type == "date" and self.value == other.value
        raise ValueError(f"contains not supported for type: {self.type}")

    def __eq__(self, other):
        if not isinstance(other, ValueClass):
            return NotImplemented
        if not self.can_compare(other):
            return False
        return self.type == other.type and self.value == other.value

    def __lt__(self, other):
        if self.type == "string":
            raise ValueError("cannot compare strings with <")
        if self.type == "quantity":
            return self.value < other.value
        if self.type == "year":
            ov = other.value if other.type == "year" else other.value.year
            return self.value < ov
        if self.type == "date":
            return self.value.year < other.value if other.type == "year" else self.value < other.value

    def __gt__(self, other):
        if self.type == "string":
            raise ValueError("cannot compare strings with >")
        if self.type == "quantity":
            return self.value > other.value
        if self.type == "year":
            ov = other.value if other.type == "year" else other.value.year
            return self.value > ov
        if self.type == "date":
            return self.value.year > other.value if other.type == "year" else self.value > other.value

    def __str__(self):
        if self.type == "string":
            return self.value
        if self.type == "quantity":
            v = int(self.value) if self.value - int(self.value) < 1e-5 else self.value
            return f"{v} {self.unit}" if self.unit and self.unit != "1" else str(v)
        if self.type == "year":
            return str(self.value)
        if self.type == "date":
            return self.value.isoformat()
        return str(self.value)


def comp(a: ValueClass, b: ValueClass, op: str) -> bool:
    """Compare two ValueClass objects with KQA-Pro time semantics."""
    if b.is_time():
        if op == "=":
            return b.contains(a)
        if op == "!=":
            return not b.contains(a)
    if op == "=":
        return a == b
    if op == "<":
        return a < b
    if op == ">":
        return a > b
    if op == "!=":
        return a != b
    return False


# ---------------------------------------------------------------------------
# KBPropertyStore
# ---------------------------------------------------------------------------

class KBPropertyStore:
    """In-memory property store for KQA-Pro KB (kb.json)."""

    def __init__(self, kb_path: str):
        logger.info("Loading KQA-Pro KB from %s ...", kb_path)
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        self.entities: Dict[str, dict] = kb.get("entities", {})
        self.concepts: Dict[str, dict] = kb.get("concepts", {})

        # name → [entity_id, ...] mapping (lowercase for case-insensitive lookup)
        self.name_to_ids: Dict[str, List[str]] = defaultdict(list)
        for eid, ent in self.entities.items():
            self.name_to_ids[ent["name"].lower()].append(eid)
        for cid, con in self.concepts.items():
            self.name_to_ids[con["name"].lower()].append(cid)

        # Build key_type index and parse values
        self.key_type: Dict[str, str] = {}
        self._build_key_type_index()
        self._parse_all_values()

        # Fuzzy key matching: normalized_key → original_key
        self._key_norm_map: Dict[str, str] = {}
        for k in self.key_type:
            self._key_norm_map[self._normalize_key(k)] = k

        logger.info(
            "KBPropertyStore loaded: %d entities, %d concepts, %d attribute keys",
            len(self.entities), len(self.concepts), len(self.key_type),
        )

    # ─── Initialization helpers ───────────────────────────────────

    def _build_key_type_index(self):
        for ent in self.entities.values():
            for attr in ent.get("attributes", []):
                self.key_type[attr["key"]] = attr["value"]["type"]
                for qk, qvs in attr.get("qualifiers", {}).items():
                    for qv in qvs:
                        self.key_type[qk] = qv["type"]
        for ent in self.entities.values():
            for rel in ent.get("relations", []):
                for qk, qvs in rel.get("qualifiers", {}).items():
                    for qv in qvs:
                        self.key_type[qk] = qv["type"]
        # Normalize: year → date (key may have both year and date values)
        self.key_type = {k: ("date" if v == "year" else v) for k, v in self.key_type.items()}

    def _parse_all_values(self):
        for ent in self.entities.values():
            for attr in ent.get("attributes", []):
                attr["value"] = self._parse_value(attr["value"])
                for qk, qvs in attr.get("qualifiers", {}).items():
                    attr["qualifiers"][qk] = [self._parse_value(qv) for qv in qvs]
        for ent in self.entities.values():
            for rel in ent.get("relations", []):
                for qk, qvs in rel.get("qualifiers", {}).items():
                    rel["qualifiers"][qk] = [self._parse_value(qv) for qv in qvs]

    @staticmethod
    def _parse_value(raw: dict) -> ValueClass:
        vtype = raw["type"]
        val = raw["value"]
        if vtype == "date":
            if "/" in str(val):
                p1, p2 = str(val).find("/"), str(val).rfind("/")
                y, m, d = int(val[:p1]), int(val[p1 + 1:p2]), int(val[p2 + 1:])
            else:
                # ISO format fallback
                parts = str(val).split("-")
                y, m, d = int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, int(parts[2]) if len(parts) > 2 else 1
            return ValueClass("date", date(y, m, d))
        if vtype == "year":
            return ValueClass("year", int(val))
        if vtype == "string":
            return ValueClass("string", str(val))
        if vtype == "quantity":
            return ValueClass("quantity", float(val), raw.get("unit", "1"))
        raise ValueError(f"unsupported value type: {vtype}")

    # ─── Key normalization / fuzzy matching ────────────────────────

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize a key for fuzzy matching: lowercase, remove underscores/spaces."""
        return re.sub(r"[\s_\-]+", "", key.lower())

    def resolve_key(self, candidate_key: str) -> Optional[str]:
        """Resolve a candidate key to an exact KB key via fuzzy matching."""
        # Exact match
        if candidate_key in self.key_type:
            return candidate_key
        # Normalized match
        norm = self._normalize_key(candidate_key)
        if norm in self._key_norm_map:
            return self._key_norm_map[norm]
        return None

    # ─── Entity ID resolution ─────────────────────────────────────

    def _get_entity_ids(self, name: str) -> List[str]:
        """Get entity IDs by name (case-insensitive)."""
        ids = self.name_to_ids.get(name.lower(), [])
        # Filter to actual entities (not concepts) first
        entity_ids = [eid for eid in ids if eid in self.entities]
        return entity_ids if entity_ids else ids

    def _get_entity(self, name: str) -> Optional[dict]:
        """Get the first matching entity by name."""
        ids = self._get_entity_ids(name)
        for eid in ids:
            if eid in self.entities:
                return self.entities[eid]
        for eid in ids:
            if eid in self.concepts:
                return self.concepts[eid]
        return None

    # ─── Public API ───────────────────────────────────────────────

    def query_attr(self, name: str, key: str) -> Optional[str]:
        """Query an attribute value by entity name and attribute key.

        Returns the string representation of the value, or None if not found.
        """
        resolved_key = self.resolve_key(key)

        ids = self._get_entity_ids(name)
        for eid in ids:
            ent = self.entities.get(eid) or self.concepts.get(eid)
            if not ent:
                continue
            for attr in ent.get("attributes", []):
                if attr["key"] == (resolved_key or key):
                    return str(attr["value"])
                # Try original key if resolved differs
                if resolved_key and attr["key"] == key:
                    return str(attr["value"])
        return None

    def query_attr_value(self, name: str, key: str) -> Optional[ValueClass]:
        """Query an attribute value as ValueClass (for comparison)."""
        resolved_key = self.resolve_key(key)
        target_key = resolved_key or key

        ids = self._get_entity_ids(name)
        for eid in ids:
            ent = self.entities.get(eid) or self.concepts.get(eid)
            if not ent:
                continue
            for attr in ent.get("attributes", []):
                if attr["key"] == target_key:
                    return attr["value"]
        return None

    def parse_value_for_key(self, key: str, raw_value: str) -> ValueClass:
        """Parse a raw string value using the known type for a key."""
        resolved_key = self.resolve_key(key)
        vtype = self.key_type.get(resolved_key or key, "string")

        if vtype == "date":
            # Try to parse as year or date
            raw = raw_value.strip()
            if re.match(r"^\d{4}$", raw):
                return ValueClass("year", int(raw))
            if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", raw):
                parts = raw.split("-")
                return ValueClass("date", date(int(parts[0]), int(parts[1]), int(parts[2])))
            return ValueClass("string", raw)
        if vtype == "quantity":
            try:
                return ValueClass("quantity", float(raw_value), "1")
            except ValueError:
                return ValueClass("string", raw_value)
        return ValueClass("string", raw_value)

    def filter_by_attr(
        self, names: List[str], key: str, value: str, op: str = "="
    ) -> List[str]:
        """Filter entity names by attribute value."""
        resolved_key = self.resolve_key(key)
        target_key = resolved_key or key
        target_value = self.parse_value_for_key(target_key, value)

        result = []
        for name in names:
            ids = self._get_entity_ids(name)
            for eid in ids:
                ent = self.entities.get(eid) or self.concepts.get(eid)
                if not ent:
                    continue
                for attr in ent.get("attributes", []):
                    if attr["key"] != target_key:
                        continue
                    av = attr["value"]
                    if av.can_compare(target_value):
                        try:
                            if comp(av, target_value, op):
                                result.append(name)
                                break
                        except Exception:
                            pass
                else:
                    continue
                break  # already matched this name
        return result

    def get_available_properties(self, name: str) -> List[Tuple[str, str, str]]:
        """Get available properties for an entity.

        Returns list of (key, value_preview, type) tuples.
        """
        ent = self._get_entity(name)
        if not ent:
            return []

        props = []
        seen_keys: Set[str] = set()
        for attr in ent.get("attributes", []):
            key = attr["key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            val_str = str(attr["value"])
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            props.append((key, val_str, attr["value"].type))
        return props

    def format_properties_for_prompt(self, name: str, max_props: int = 30) -> str:
        """Format entity properties for LLM prompt injection."""
        props = self.get_available_properties(name)
        if not props:
            return ""
        lines = [f"Available properties for '{name}':"]
        for key, val_preview, vtype in props[:max_props]:
            lines.append(f"  - {key} ({vtype}): {val_preview}")
        if len(props) > max_props:
            lines.append(f"  ... and {len(props) - max_props} more")
        return "\n".join(lines)
