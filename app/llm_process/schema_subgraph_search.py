"""
Type KoPL出力をパースし、スキーマグラフを探索してサブグラフ候補を取得するモジュール。

主な機能:
1. LLM生成のType KoPLテキストをパースして構造化
2. スキーマグラフ（TransitionMap）から有効なrelationパスを列挙
3. 組み合わせによるサブグラフ候補の生成
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from dataset_construction.add_kopl import TransitionMap, build_transition_map_from_kg


# ────────────────────────────────────────────────────────────────
#  Type KoPL パーサー
# ────────────────────────────────────────────────────────────────
@dataclass
class FindanchorStep:
    """Findanchor関数の情報"""

    var_name: str
    anchor_type: str
    entity_name: str


@dataclass
class FindTypeRelateStep:
    """FindTypeRelate関数の情報"""

    var_name: str
    input_var: str
    anchor_type: str
    target_type: str
    relation_name: str
    delta: int = 1


@dataclass
class AndStep:
    """And関数の情報"""

    var_name: str
    exp1: str
    exp2: str


@dataclass
class OrStep:
    """Or関数の情報"""

    var_name: str
    exp1: str
    exp2: str


@dataclass
class StopStep:
    """Stop関数の情報"""

    var_name: str
    input_var: str


TypeKoPLStep = FindanchorStep | FindTypeRelateStep | AndStep | OrStep | StopStep


@dataclass
class ParsedTypeKoPL:
    """パースされたType KoPL全体"""

    steps: List[TypeKoPLStep] = field(default_factory=list)
    anchors: List[FindanchorStep] = field(default_factory=list)
    transitions: List[FindTypeRelateStep] = field(default_factory=list)
    and_ops: List[AndStep] = field(default_factory=list)
    or_ops: List[OrStep] = field(default_factory=list)
    stop: Optional[StopStep] = None

    def get_type_path(self) -> List[Tuple[str, str, str]]:
        """
        (anchor_type, target_type, relation_name)のリストを返す。
        チェーン型のクエリで使用。
        """
        return [
            (t.anchor_type, t.target_type, t.relation_name) for t in self.transitions
        ]


def parse_type_kopl(kopl_text: str) -> ParsedTypeKoPL:
    """
    Type KoPLテキストをパースして構造化する。

    入力例:
    ```
    exp1 = Findanchor(anchor_type='Person', entity_name='Bob Denver')
    films = FindTypeRelate(exp1, anchor_type="Person", target_type="Movie", relation_name="starred_in", delta=1)
    directors = FindTypeRelate(films, anchor_type="Movie", target_type="Person", relation_name="directed_by", delta=1)
    final = Stop(directors)
    ```

    Returns:
        ParsedTypeKoPL: パースされた構造体
    """
    result = ParsedTypeKoPL()

    # Findanchor パターン
    findanchor_pattern = re.compile(
        r"(\w+)\s*=\s*Findanchor\s*\(\s*"
        r"anchor_type\s*=\s*['\"](\w+)['\"]"
        r"\s*,\s*entity_name\s*=\s*['\"]([^'\"]+)['\"]"
        r"\s*\)"
    )

    # FindTypeRelate パターン
    findtyperelate_pattern = re.compile(
        r"(\w+)\s*=\s*FindTypeRelate\s*\(\s*"
        r"(\w+)\s*,\s*"
        r"anchor_type\s*=\s*['\"](\w+)['\"]"
        r"\s*,\s*target_type\s*=\s*['\"](\w+)['\"]"
        r"\s*,\s*relation_name\s*=\s*['\"]([^'\"]+)['\"]"
        r"(?:\s*,\s*delta\s*=\s*(\d+))?"
        r"\s*\)"
    )

    # And パターン（代入形式）
    and_pattern = re.compile(r"(\w+)\s*=\s*And\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")

    # Or パターン（代入形式）
    or_pattern = re.compile(r"(\w+)\s*=\s*Or\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")

    # Stop パターン（引数に式が入る場合も対応）
    stop_pattern = re.compile(r"(\w+)\s*=\s*Stop\s*\(\s*(.+?)\s*\)")

    # Stop内のAnd/Orパターン（例: Stop(And(exp1, exp2))）
    stop_and_pattern = re.compile(r"Stop\s*\(\s*And\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\)")
    stop_or_pattern = re.compile(r"Stop\s*\(\s*Or\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\)")

    for line in kopl_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Findanchor
        match = findanchor_pattern.search(line)
        if match:
            step = FindanchorStep(
                var_name=match.group(1),
                anchor_type=match.group(2),
                entity_name=match.group(3),
            )
            result.steps.append(step)
            result.anchors.append(step)
            continue

        # FindTypeRelate
        match = findtyperelate_pattern.search(line)
        if match:
            delta = int(match.group(6)) if match.group(6) else 1
            step = FindTypeRelateStep(
                var_name=match.group(1),
                input_var=match.group(2),
                anchor_type=match.group(3),
                target_type=match.group(4),
                relation_name=match.group(5),
                delta=delta,
            )
            result.steps.append(step)
            result.transitions.append(step)
            continue

        # And
        match = and_pattern.search(line)
        if match:
            step = AndStep(
                var_name=match.group(1),
                exp1=match.group(2),
                exp2=match.group(3),
            )
            result.steps.append(step)
            result.and_ops.append(step)
            continue

        # Or
        match = or_pattern.search(line)
        if match:
            step = OrStep(
                var_name=match.group(1),
                exp1=match.group(2),
                exp2=match.group(3),
            )
            result.steps.append(step)
            result.or_ops.append(step)
            continue

        # Stop（And/Or内包のチェック）
        match = stop_pattern.search(line)
        if match:
            input_expr = match.group(2).strip()

            # Stop内のAndをチェック
            and_match = stop_and_pattern.search(line)
            if and_match:
                and_step = AndStep(
                    var_name="_implicit_and",
                    exp1=and_match.group(1),
                    exp2=and_match.group(2),
                )
                result.steps.append(and_step)
                result.and_ops.append(and_step)
                input_expr = "_implicit_and"

            # Stop内のOrをチェック
            or_match = stop_or_pattern.search(line)
            if or_match:
                or_step = OrStep(
                    var_name="_implicit_or",
                    exp1=or_match.group(1),
                    exp2=or_match.group(2),
                )
                result.steps.append(or_step)
                result.or_ops.append(or_step)
                input_expr = "_implicit_or"

            step = StopStep(
                var_name=match.group(1),
                input_var=input_expr,
            )
            result.steps.append(step)
            result.stop = step
            continue

    return result


# ────────────────────────────────────────────────────────────────
#  スキーマグラフ探索
# ────────────────────────────────────────────────────────────────
@dataclass
class SchemaGraph:
    """
    TransitionMapを拡張し、スキーマグラフ探索機能を提供する。

    - 型遷移 (src_type, tgt_type) に対応するrelationのリストを取得
    - Type KoPLに基づくサブグラフ候補（relationパス）の列挙
    """

    transitions: TransitionMap
    # 逆引きマップ: (src_type, tgt_type) -> frozenset of relations
    _type_pair_to_relations: Dict[Tuple[str, str], FrozenSet[str]] = field(
        default_factory=dict
    )
    # src_type -> {(relation, tgt_type), ...}
    _outgoing: Dict[str, Set[Tuple[str, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._build_indexes()

    def _build_indexes(self) -> None:
        """TransitionMapから逆引きインデックスを構築"""
        type_pair_rels: Dict[Tuple[str, str], Set[str]] = {}
        outgoing: Dict[str, Set[Tuple[str, str]]] = {}

        for (relation, src_type), tgt_type in self.transitions.mapping.items():
            # (src, tgt) -> relations
            key = (src_type, tgt_type)
            if key not in type_pair_rels:
                type_pair_rels[key] = set()
            type_pair_rels[key].add(relation)

            # src -> {(rel, tgt), ...}
            if src_type not in outgoing:
                outgoing[src_type] = set()
            outgoing[src_type].add((relation, tgt_type))

        self._type_pair_to_relations = {
            k: frozenset(v) for k, v in type_pair_rels.items()
        }
        self._outgoing = outgoing

    def get_relations_for_transition(
        self, src_type: str, tgt_type: str
    ) -> FrozenSet[str]:
        """
        指定された型遷移(src_type → tgt_type)に対応するrelation集合を返す。
        """
        return self._type_pair_to_relations.get((src_type, tgt_type), frozenset())

    def get_outgoing_edges(self, src_type: str) -> Set[Tuple[str, str]]:
        """
        指定された型から出るエッジ（relation, tgt_type）の集合を返す。
        """
        return self._outgoing.get(src_type, set())

    def get_all_types(self) -> Set[str]:
        """スキーマグラフに含まれる全ての型を返す"""
        types: Set[str] = set()
        for (relation, src_type), tgt_type in self.transitions.mapping.items():
            types.add(src_type)
            types.add(tgt_type)
        return types

    def get_all_relations(self) -> Set[str]:
        """スキーマグラフに含まれる全てのrelationを返す"""
        return {rel for (rel, _) in self.transitions.mapping.keys()}


# ────────────────────────────────────────────────────────────────
#  サブグラフ候補探索
# ────────────────────────────────────────────────────────────────
@dataclass
class SubgraphCandidate:
    """サブグラフ候補"""

    relation_path: List[str]  # relationのシーケンス
    type_path: List[str]  # 型のシーケンス (anchor → ... → target)
    query_structure: str  # "chain", "intersection", "union"
    branches: Optional[List["SubgraphCandidate"]] = None  # intersection/union用


def search_chain_subgraphs(
    schema: SchemaGraph,
    parsed: ParsedTypeKoPL,
    *,
    max_candidates: int = 100,
) -> List[SubgraphCandidate]:
    """
    チェーン型クエリ (1-hop, 2-hop chain) のサブグラフ候補を探索する。

    Args:
        schema: SchemaGraph
        parsed: パースされたType KoPL
        max_candidates: 最大候補数

    Returns:
        サブグラフ候補のリスト
    """
    if not parsed.transitions:
        return []

    # 各遷移に対応するrelation候補を収集
    relation_options: List[List[str]] = []
    type_sequence: List[str] = []

    first_anchor = parsed.anchors[0] if parsed.anchors else None
    if first_anchor:
        type_sequence.append(first_anchor.anchor_type)

    for trans in parsed.transitions:
        rels = schema.get_relations_for_transition(trans.anchor_type, trans.target_type)
        if not rels:
            # スキーマに該当する遷移がない場合、LLMが指定したrelationを使用
            relation_options.append([trans.relation_name])
        else:
            relation_options.append(list(rels))
        type_sequence.append(trans.target_type)

    # 全組み合わせを生成
    candidates: List[SubgraphCandidate] = []
    for combo in product(*relation_options):
        if len(candidates) >= max_candidates:
            break
        candidates.append(
            SubgraphCandidate(
                relation_path=list(combo),
                type_path=type_sequence.copy(),
                query_structure="chain",
            )
        )

    return candidates


def search_intersection_subgraphs(
    schema: SchemaGraph,
    parsed: ParsedTypeKoPL,
    *,
    max_candidates: int = 100,
) -> List[SubgraphCandidate]:
    """
    Intersection型クエリ (2-anchor, 3-anchor intersection) のサブグラフ候補を探索する。

    Args:
        schema: SchemaGraph
        parsed: パースされたType KoPL
        max_candidates: 最大候補数

    Returns:
        サブグラフ候補のリスト
    """
    if not parsed.and_ops:
        return []

    # アンカーごとにブランチを構築
    # 簡易実装: 各アンカー→最終的なAndまでのパスを個別に処理
    branches: List[List[SubgraphCandidate]] = []

    # 変数名→ステップのマッピング
    var_to_step: Dict[str, TypeKoPLStep] = {}
    for step in parsed.steps:
        if hasattr(step, "var_name"):
            var_to_step[step.var_name] = step

    # 各アンカーからの遷移を追跡
    for anchor in parsed.anchors:
        branch_transitions: List[FindTypeRelateStep] = []

        # アンカーから始まる遷移チェーンを追跡
        current_var = anchor.var_name
        for trans in parsed.transitions:
            if trans.input_var == current_var:
                branch_transitions.append(trans)
                current_var = trans.var_name

        if not branch_transitions:
            continue

        # このブランチのrelation候補を収集
        relation_options: List[List[str]] = []
        type_sequence = [anchor.anchor_type]

        for trans in branch_transitions:
            rels = schema.get_relations_for_transition(
                trans.anchor_type, trans.target_type
            )
            if not rels:
                relation_options.append([trans.relation_name])
            else:
                relation_options.append(list(rels))
            type_sequence.append(trans.target_type)

        # ブランチ候補を生成
        branch_candidates: List[SubgraphCandidate] = []
        for combo in product(*relation_options):
            branch_candidates.append(
                SubgraphCandidate(
                    relation_path=list(combo),
                    type_path=type_sequence.copy(),
                    query_structure="branch",
                )
            )
        branches.append(branch_candidates)

    if not branches:
        return []

    # 全ブランチの組み合わせ
    candidates: List[SubgraphCandidate] = []
    for branch_combo in product(*branches):
        if len(candidates) >= max_candidates:
            break
        all_relations = []
        all_types = []
        for branch in branch_combo:
            all_relations.extend(branch.relation_path)
            if not all_types:
                all_types = branch.type_path.copy()

        candidates.append(
            SubgraphCandidate(
                relation_path=all_relations,
                type_path=all_types,
                query_structure="intersection",
                branches=list(branch_combo),
            )
        )

    return candidates


def search_subgraph_candidates(
    schema: SchemaGraph,
    parsed: ParsedTypeKoPL,
    *,
    max_candidates: int = 100,
) -> List[SubgraphCandidate]:
    """
    Type KoPLに基づいてサブグラフ候補を探索する統合関数。

    Args:
        schema: SchemaGraph
        parsed: パースされたType KoPL
        max_candidates: 最大候補数

    Returns:
        サブグラフ候補のリスト
    """
    if parsed.and_ops:
        return search_intersection_subgraphs(
            schema, parsed, max_candidates=max_candidates
        )
    elif parsed.or_ops:
        # Union型は将来拡張（現時点ではchain型として処理）
        return search_chain_subgraphs(schema, parsed, max_candidates=max_candidates)
    else:
        return search_chain_subgraphs(schema, parsed, max_candidates=max_candidates)


# ────────────────────────────────────────────────────────────────
#  スキーマベースの型パス探索（Type KoPLなしで使用可能）
# ────────────────────────────────────────────────────────────────
def enumerate_type_paths(
    schema: SchemaGraph,
    start_type: str,
    max_hops: int = 3,
    *,
    target_type: Optional[str] = None,
) -> Iterator[List[Tuple[str, str, str]]]:
    """
    スキーマグラフ上で、指定された型から到達可能な全ての型パスを列挙する。

    Args:
        schema: SchemaGraph
        start_type: 開始型
        max_hops: 最大ホップ数
        target_type: 目標型（指定時はこの型で終わるパスのみ）

    Yields:
        [(src_type, relation, tgt_type), ...] のリスト
    """

    def dfs(
        current_type: str,
        current_path: List[Tuple[str, str, str]],
        visited_types: Set[str],
    ) -> Iterator[List[Tuple[str, str, str]]]:
        if len(current_path) > 0:
            if target_type is None or current_type == target_type:
                yield current_path.copy()

        if len(current_path) >= max_hops:
            return

        for relation, next_type in schema.get_outgoing_edges(current_type):
            if next_type in visited_types:
                continue

            current_path.append((current_type, relation, next_type))
            visited_types.add(next_type)

            yield from dfs(next_type, current_path, visited_types)

            current_path.pop()
            visited_types.discard(next_type)

    yield from dfs(start_type, [], {start_type})


def get_relation_paths_for_type_path(
    schema: SchemaGraph,
    type_path: List[str],
    *,
    max_candidates: int = 100,
) -> List[List[str]]:
    """
    型パス [Type1, Type2, Type3, ...] に対応するrelationパスの候補を返す。

    Args:
        schema: SchemaGraph
        type_path: 型のシーケンス
        max_candidates: 最大候補数

    Returns:
        relationパスのリスト
    """
    if len(type_path) < 2:
        return [[]]

    relation_options: List[List[str]] = []
    for i in range(len(type_path) - 1):
        src, tgt = type_path[i], type_path[i + 1]
        rels = schema.get_relations_for_transition(src, tgt)
        if not rels:
            return []  # この遷移がスキーマに存在しない
        relation_options.append(list(rels))

    candidates: List[List[str]] = []
    for combo in product(*relation_options):
        if len(candidates) >= max_candidates:
            break
        candidates.append(list(combo))

    return candidates


# ────────────────────────────────────────────────────────────────
#  便利関数: Type KoPLテキストからサブグラフ候補を直接取得
# ────────────────────────────────────────────────────────────────
def get_subgraph_candidates_from_type_kopl(
    kopl_text: str,
    transitions: TransitionMap,
    *,
    max_candidates: int = 100,
) -> List[SubgraphCandidate]:
    """
    Type KoPLテキストからサブグラフ候補を取得する便利関数。

    Args:
        kopl_text: LLM生成のType KoPLテキスト
        transitions: TransitionMap
        max_candidates: 最大候補数

    Returns:
        サブグラフ候補のリスト
    """
    parsed = parse_type_kopl(kopl_text)
    schema = SchemaGraph(transitions=transitions)
    return search_subgraph_candidates(schema, parsed, max_candidates=max_candidates)


# ────────────────────────────────────────────────────────────────
#  メイン（テスト用）
# ────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Type KoPLからサブグラフ候補を探索するテスト"
    )
    p.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("data/kg/nodes.csv"),
        help="KG nodes CSV",
    )
    p.add_argument(
        "--rels-csv",
        type=Path,
        default=Path("data/kg/relationships.csv"),
        help="KG relationships CSV",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="最大候補数",
    )

    args = p.parse_args()

    # TransitionMapの構築
    print("Building TransitionMap from KG CSVs...")
    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv,
        rels_csv=args.rels_csv,
    )
    print(f"Transition rules: {len(transitions.mapping)}")

    schema = SchemaGraph(transitions=transitions)
    print(f"Types: {schema.get_all_types()}")
    print(f"Relations: {schema.get_all_relations()}")

    # テスト用Type KoPL（チェーン型）
    test_kopl_chain = """
exp1 = Findanchor(anchor_type='Disease', entity_name='Silver-Russell syndrome')
genes = FindTypeRelate(exp1, anchor_type="Disease", target_type="Gene", relation_name="has_phenotype", delta=1)
drugs = FindTypeRelate(genes, anchor_type="Gene", target_type="Drug", relation_name="targets", delta=1)
final = Stop(drugs)
"""

    print("\n" + "=" * 60)
    print("Test: Chain Type KoPL")
    print("=" * 60)
    print(test_kopl_chain)

    parsed = parse_type_kopl(test_kopl_chain)
    print(f"\nParsed anchors: {len(parsed.anchors)}")
    print(f"Parsed transitions: {len(parsed.transitions)}")

    candidates = search_subgraph_candidates(
        schema, parsed, max_candidates=args.max_candidates
    )
    print(f"\nSubgraph candidates ({len(candidates)}):")
    for i, cand in enumerate(candidates[:10]):
        print(f"  [{i+1}] relations: {cand.relation_path}, types: {cand.type_path}")

    # テスト用Type KoPL（Intersection型）
    test_kopl_intersection = """
exp1 = Findanchor(anchor_type='Disease', entity_name='Silver-Russell syndrome')
srs_genes = FindTypeRelate(exp1, anchor_type="Disease", target_type="Gene", relation_name="has_phenotype", delta=1)

exp2 = Findanchor(anchor_type='Disease', entity_name='benign mesothelioma')
bm_genes = FindTypeRelate(exp2, anchor_type="Disease", target_type="Gene", relation_name="related_to", delta=1)

final = Stop(And(srs_genes, bm_genes))
"""

    print("\n" + "=" * 60)
    print("Test: Intersection Type KoPL")
    print("=" * 60)
    print(test_kopl_intersection)

    parsed = parse_type_kopl(test_kopl_intersection)
    print(f"\nParsed anchors: {len(parsed.anchors)}")
    print(f"Parsed transitions: {len(parsed.transitions)}")
    print(f"Parsed And operations: {len(parsed.and_ops)}")

    candidates = search_subgraph_candidates(
        schema, parsed, max_candidates=args.max_candidates
    )
    print(f"\nSubgraph candidates ({len(candidates)}):")
    for i, cand in enumerate(candidates[:10]):
        print(f"  [{i+1}] relations: {cand.relation_path}")
        if cand.branches:
            for j, branch in enumerate(cand.branches):
                print(f"       branch {j+1}: {branch.relation_path} ({branch.type_path})")


if __name__ == "__main__":
    main()
