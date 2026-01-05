from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

from tqdm import tqdm

RAW_TYPE_TO_CLEAN_TYPE: Mapping[str, str] = {
    "anatomy": "Anatomy",
    "gene/protein": "Gene",
    "biological_process": "BiologicalProcess",
    "cellular_component": "CellularComponent",
    "disease": "Disease",
    "drug": "Drug",
    "effect/phenotype": "Phenotype",
    "exposure": "Exposure",
    "molecular_function": "MolecularFunction",
    "pathway": "Pathway",
    "unknown": "Entity",
}


def _clean_type(raw: Optional[str]) -> str:
    if not raw:
        return "Entity"
    return RAW_TYPE_TO_CLEAN_TYPE.get(str(raw), "Entity")


def _first_of_list(value: Any) -> Optional[Any]:
    if isinstance(value, list) and value:
        return value[0]
    return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").strip()
            if not raw:
                continue
            row = json.loads(raw)
            if isinstance(row, dict):
                yield row


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _first_present(row: Mapping[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return None


def _serialize_kopl_func_style(program: List[Dict[str, Any]]) -> str:
    """Serialize KoPL list into the "<func>" / "<arg>" style string."""

    if not program:
        return ""

    parts: List[str] = []
    for step in program:
        func = step.get("function")
        inputs = step.get("inputs") or []
        args = "".join([f" <arg> {str(inp)}" for inp in inputs])
        parts.append(f"{func}{args}")
    return " <func> ".join(parts)


@dataclass(frozen=True)
class TransitionMap:
    # (relation, src_clean_type) -> tgt_clean_type
    mapping: Mapping[Tuple[str, str], str]

    def next_type(self, *, current_type: str, relation: str) -> str:
        if not relation:
            return "Entity"
        return self.mapping.get((relation, current_type), "Entity")


def build_transition_map_from_kg(
    *, nodes_csv: Path, rels_csv: Path, max_edges: int = 0
) -> TransitionMap:
    """Infer (relation, source_type)->target_type from KG CSV exports.

    nodes_csv: expects columns [id:ID, :LABEL, name]
    rels_csv: expects columns [:START_ID, :END_ID, :TYPE]

    If multiple target types are seen for the same (relation, source), choose the most frequent.
    """

    id_to_clean_type: Dict[str, str] = {}

    with nodes_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row.get("id:ID")
            raw_label = row.get(":LABEL")
            if node_id is None:
                continue
            id_to_clean_type[str(node_id)] = _clean_type(raw_label)

    counts: DefaultDict[Tuple[str, str], Counter[str]] = defaultdict(Counter)

    with rels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            if max_edges and i > max_edges:
                break
            rel = row.get(":TYPE")
            if not rel:
                continue
            start_id = row.get(":START_ID")
            end_id = row.get(":END_ID")
            if start_id is None or end_id is None:
                continue
            src = id_to_clean_type.get(str(start_id), "Entity")
            tgt = id_to_clean_type.get(str(end_id), "Entity")
            counts[(rel, src)][tgt] += 1

    mapping: Dict[Tuple[str, str], str] = {}
    for key, c in counts.items():
        mapping[key] = c.most_common(1)[0][0]

    return TransitionMap(mapping=mapping)


def _strict_kopl_struct(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """ "厳密KoPL" (具体エンティティ名 + 関係名) を生成する."""

    q_type = row.get("query_type") or row.get("question_type") or row.get("type")
    DEFAULT_DIRECTION = "forward"

    if q_type == "one_hop_chain":
        anchor = _first_present(
            row, ["anchor_name", "anchor", "head_name", "start_name"]
        )
        rel = _first_present(row, ["rel_type", "relation", "edge_type", "rel"])
        if not anchor or not rel:
            return []
        return [
            {"function": "Find", "inputs": [anchor]},
            {"function": "Relate", "inputs": [rel, DEFAULT_DIRECTION]},
            {"function": "QueryName", "inputs": []},
        ]

    if q_type == "two_hop_chain":
        anchor = _first_present(
            row, ["anchor_name", "anchor", "head_name", "start_name"]
        )
        rel1 = _first_present(row, ["rel1", "relation1", "edge1", "rel_1"])
        rel2 = _first_present(row, ["rel2", "relation2", "edge2", "rel_2"])
        if not anchor or not rel1 or not rel2:
            return []
        return [
            {"function": "Find", "inputs": [anchor]},
            {"function": "Relate", "inputs": [rel1, DEFAULT_DIRECTION]},
            {"function": "Relate", "inputs": [rel2, DEFAULT_DIRECTION]},
            {"function": "QueryName", "inputs": []},
        ]

    if q_type == "two_anchor_intersection":
        a = _first_present(
            row,
            ["anchorA_name", "anchor_a_name", "anchor1_name", "anchorA", "anchor_1"],
        )
        b = _first_present(
            row,
            ["anchorB_name", "anchor_b_name", "anchor2_name", "anchorB", "anchor_2"],
        )
        rel_a = _first_present(
            row,
            [
                "anchorA_edge_type",
                "anchorA_rel",
                "rel1",
                "edge_type_1",
                "anchor1_edge_type",
            ],
        )
        rel_b = _first_present(
            row,
            [
                "anchorB_edge_type",
                "anchorB_rel",
                "rel2",
                "edge_type_2",
                "anchor2_edge_type",
            ],
        )
        if not a or not b or not rel_a or not rel_b:
            return []
        return [
            {"function": "Find", "inputs": [a]},
            {"function": "Relate", "inputs": [rel_a, DEFAULT_DIRECTION]},
            {"function": "Find", "inputs": [b]},
            {"function": "Relate", "inputs": [rel_b, DEFAULT_DIRECTION]},
            {"function": "And", "inputs": []},
            {"function": "QueryName", "inputs": []},
        ]

    if q_type == "three_anchor_intersection":
        a = _first_present(
            row,
            ["anchorA_name", "anchor_a_name", "anchor1_name", "anchorA", "anchor_1"],
        )
        b = _first_present(
            row,
            ["anchorB_name", "anchor_b_name", "anchor2_name", "anchorB", "anchor_2"],
        )
        c = _first_present(
            row,
            ["anchorC_name", "anchor_c_name", "anchor3_name", "anchorC", "anchor_3"],
        )
        rel_a = _first_present(
            row,
            [
                "anchorA_edge_type",
                "anchorA_rel",
                "rel1",
                "edge_type_1",
                "anchor1_edge_type",
            ],
        )
        rel_b = _first_present(
            row,
            [
                "anchorB_edge_type",
                "anchorB_rel",
                "rel2",
                "edge_type_2",
                "anchor2_edge_type",
            ],
        )
        rel_c = _first_present(
            row,
            [
                "anchorC_edge_type",
                "anchorC_rel",
                "rel3",
                "edge_type_3",
                "anchor3_edge_type",
            ],
        )
        if not a or not b or not c or not rel_a or not rel_b or not rel_c:
            return []
        return [
            {"function": "Find", "inputs": [a]},
            {"function": "Relate", "inputs": [rel_a, DEFAULT_DIRECTION]},
            {"function": "Find", "inputs": [b]},
            {"function": "Relate", "inputs": [rel_b, DEFAULT_DIRECTION]},
            {"function": "And", "inputs": []},
            {"function": "Find", "inputs": [c]},
            {"function": "Relate", "inputs": [rel_c, DEFAULT_DIRECTION]},
            {"function": "And", "inputs": []},
            {"function": "QueryName", "inputs": []},
        ]

    return []


def _type_kopl_struct(
    row: Mapping[str, Any], *, transitions: TransitionMap
) -> List[Dict[str, Any]]:
    """ "type KoPL" (Type抽象 + relationで遷移推定) を生成する."""

    q_type = row.get("query_type") or row.get("question_type") or row.get("type")
    DEFAULT_DIRECTION = "forward"

    def infer_anchor_clean_type(types_field: Any) -> str:
        return _clean_type(_first_of_list(types_field) or "unknown")

    def infer_answer_clean_type() -> Optional[str]:
        ans = row.get("answer_nodes_sample")
        if isinstance(ans, list) and ans:
            ans0 = ans[0]
            if isinstance(ans0, dict):
                t = _first_of_list(ans0.get("types"))
                if t is not None:
                    return _clean_type(str(t))
        return None

    if q_type == "one_hop_chain":
        curr = infer_anchor_clean_type(
            _first_present(row, ["anchor_types", "key_type", "head_type", "start_type"])
        )
        rel = _first_present(row, ["rel_type", "relation", "edge_type", "rel"])
        if not rel:
            return []
        nxt = transitions.next_type(current_type=curr, relation=str(rel))
        out_type = infer_answer_clean_type() or nxt

        return [
            {"function": "Find", "inputs": [f"Type:{curr}"]},
            {"function": "Relate", "inputs": [f"Type:{nxt}", DEFAULT_DIRECTION]},
            {"function": "FilterConcept", "inputs": [f"Type:{out_type}"]},
        ]

    if q_type == "two_hop_chain":
        curr = infer_anchor_clean_type(
            _first_present(row, ["anchor_types", "key_type", "head_type", "start_type"])
        )
        rel1 = _first_present(row, ["rel1", "relation1", "edge1", "rel_1"])
        rel2 = _first_present(row, ["rel2", "relation2", "edge2", "rel_2"])
        if not rel1 or not rel2:
            return []
        t1 = transitions.next_type(current_type=curr, relation=str(rel1))
        t2 = transitions.next_type(current_type=t1, relation=str(rel2))
        out_type = infer_answer_clean_type() or t2

        return [
            {"function": "Find", "inputs": [f"Type:{curr}"]},
            {"function": "Relate", "inputs": [f"Type:{t1}", DEFAULT_DIRECTION]},
            {"function": "Relate", "inputs": [f"Type:{t2}", DEFAULT_DIRECTION]},
            {"function": "FilterConcept", "inputs": [f"Type:{out_type}"]},
        ]

    if q_type == "two_anchor_intersection":
        type_a = infer_anchor_clean_type(
            _first_present(
                row, ["anchorA_types", "key_type_1", "type1", "anchor1_types"]
            )
        )
        type_b = infer_anchor_clean_type(
            _first_present(
                row, ["anchorB_types", "key_type_2", "type2", "anchor2_types"]
            )
        )
        rel_a = _first_present(
            row,
            [
                "anchorA_edge_type",
                "anchorA_rel",
                "rel1",
                "edge_type_1",
                "anchor1_edge_type",
            ],
        )
        rel_b = _first_present(
            row,
            [
                "anchorB_edge_type",
                "anchorB_rel",
                "rel2",
                "edge_type_2",
                "anchor2_edge_type",
            ],
        )
        if not rel_a or not rel_b:
            return []
        tgt_a = transitions.next_type(current_type=type_a, relation=str(rel_a))
        tgt_b = transitions.next_type(current_type=type_b, relation=str(rel_b))
        intersect_type = tgt_a if tgt_a == tgt_b else "Entity"
        out_type = infer_answer_clean_type() or intersect_type

        return [
            {"function": "Find", "inputs": [f"Type:{type_a}"]},
            {"function": "Relate", "inputs": [f"Type:{tgt_a}", DEFAULT_DIRECTION]},
            {"function": "Find", "inputs": [f"Type:{type_b}"]},
            {"function": "Relate", "inputs": [f"Type:{tgt_b}", DEFAULT_DIRECTION]},
            {"function": "And", "inputs": []},
            {"function": "FilterConcept", "inputs": [f"Type:{out_type}"]},
        ]

    if q_type == "three_anchor_intersection":
        type_a = infer_anchor_clean_type(
            _first_present(
                row, ["anchorA_types", "key_type_1", "type1", "anchor1_types"]
            )
        )
        type_b = infer_anchor_clean_type(
            _first_present(
                row, ["anchorB_types", "key_type_2", "type2", "anchor2_types"]
            )
        )
        type_c = infer_anchor_clean_type(
            _first_present(
                row, ["anchorC_types", "key_type_3", "type3", "anchor3_types"]
            )
        )
        rel_a = _first_present(
            row,
            [
                "anchorA_edge_type",
                "anchorA_rel",
                "rel1",
                "edge_type_1",
                "anchor1_edge_type",
            ],
        )
        rel_b = _first_present(
            row,
            [
                "anchorB_edge_type",
                "anchorB_rel",
                "rel2",
                "edge_type_2",
                "anchor2_edge_type",
            ],
        )
        rel_c = _first_present(
            row,
            [
                "anchorC_edge_type",
                "anchorC_rel",
                "rel3",
                "edge_type_3",
                "anchor3_edge_type",
            ],
        )
        if not rel_a or not rel_b or not rel_c:
            return []

        tgt_a = transitions.next_type(current_type=type_a, relation=str(rel_a))
        tgt_b = transitions.next_type(current_type=type_b, relation=str(rel_b))
        tgt_ab = tgt_a if tgt_a == tgt_b else "Entity"

        tgt_c = transitions.next_type(current_type=type_c, relation=str(rel_c))
        tgt_abc = tgt_ab if tgt_ab == tgt_c else "Entity"
        out_type = infer_answer_clean_type() or tgt_abc

        return [
            {"function": "Find", "inputs": [f"Type:{type_a}"]},
            {"function": "Relate", "inputs": [f"Type:{tgt_a}", DEFAULT_DIRECTION]},
            {"function": "Find", "inputs": [f"Type:{type_b}"]},
            {"function": "Relate", "inputs": [f"Type:{tgt_b}", DEFAULT_DIRECTION]},
            {"function": "And", "inputs": []},
            {"function": "Find", "inputs": [f"Type:{type_c}"]},
            {"function": "Relate", "inputs": [f"Type:{tgt_c}", DEFAULT_DIRECTION]},
            {"function": "And", "inputs": []},
            {"function": "FilterConcept", "inputs": [f"Type:{out_type}"]},
        ]

    return []


def add_kopl_fields(
    row: MutableMapping[str, Any],
    *,
    transitions: TransitionMap,
    strict_key: str = "kopl_strict",
    strict_struct_key: str = "kopl_strict_struct",
    type_key: str = "kopl_type",
    type_struct_key: str = "kopl_type_struct",
) -> MutableMapping[str, Any]:
    strict_struct = _strict_kopl_struct(row)
    type_struct = _type_kopl_struct(row, transitions=transitions)

    row[strict_struct_key] = strict_struct
    row[strict_key] = _serialize_kopl_func_style(strict_struct)

    row[type_struct_key] = type_struct
    # type KoPL は「JSON文字列」が欲しいケースが多いので、まずはjson.dumpsで統一
    row[type_key] = json.dumps(type_struct, ensure_ascii=False)

    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Add KoPL fields to PrimeKGQA JSONL rows. "
            "Generates two variants: strict KoPL (entity names) and type KoPL (type-abstracted)."
        )
    )
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("result/dataset_construction/all_with_questions_v3.jsonl"),
        help="Input JSONL (default: result/dataset_construction/all_with_questions_v3.jsonl)",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path(
            "result/dataset_construction/all_with_questions_v3_with_kopl.jsonl"
        ),
        help="Output JSONL path",
    )
    p.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("data/kg/nodes.csv"),
        help="KG nodes CSV for inferring type transitions (default: data/kg/nodes.csv)",
    )
    p.add_argument(
        "--rels-csv",
        type=Path,
        default=Path("data/kg/relationships.csv"),
        help="KG relationships CSV for inferring type transitions (default: data/kg/relationships.csv)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, process at most this many rows",
    )
    p.add_argument(
        "--max-edges",
        type=int,
        default=0,
        help="If >0, use at most this many KG edges to infer transitions (faster, less accurate)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if exists",
    )

    args = p.parse_args()

    if args.out_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists (use --overwrite): {args.out_path}")

    if not args.in_path.exists():
        raise SystemExit(f"input not found: {args.in_path}")

    if not args.nodes_csv.exists() or not args.rels_csv.exists():
        raise SystemExit(
            f"KG CSVs not found: nodes={args.nodes_csv} rels={args.rels_csv}"
        )

    print("Inferring transition map from KG CSVs...")
    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv,
        rels_csv=args.rels_csv,
        max_edges=int(args.max_edges or 0),
    )
    print(f"Transition rules: {len(transitions.mapping)}")

    out_rows: List[Dict[str, Any]] = []

    processed = 0
    for row in tqdm(_iter_jsonl(args.in_path), desc="Adding KoPL"):
        processed += 1
        if args.max_rows and processed > args.max_rows:
            break

        new_row = dict(row)
        add_kopl_fields(new_row, transitions=transitions)
        # 空KoPLは学習用に微妙なので、最低限 strict/type のどちらかが非空のものだけ残す
        if new_row.get("kopl_strict") or new_row.get("kopl_type") not in {"[]", ""}:
            out_rows.append(new_row)

    _write_jsonl(args.out_path, out_rows)
    print(f"Wrote: {args.out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
