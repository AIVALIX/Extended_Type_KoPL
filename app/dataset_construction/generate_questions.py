from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from pydantic import BaseModel, Field

from tqdm import tqdm

from langchain.chat_models import init_chat_model

from core.config import BASEMODEL, get_settings


class QuestionResponse(BaseModel):
    question: str = Field(..., description="A single natural-language question")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _generate_questions_parallel(
    *,
    gen: "QuestionGenerator",
    rows: List[Dict[str, Any]],
    max_workers: int,
    max_inflight: int,
    per_task_timeout: float,
    watch_interval: float,
) -> List[Dict[str, Any]]:
    """Generate questions in parallel while keeping output order.

    - Limits concurrent LLM calls via a semaphore (max_inflight)
    - Produces per-row errors instead of crashing the whole run
    - Uses a timeout watcher loop to avoid hanging forever on slow calls

        Note:
        - We intentionally DO NOT submit all rows at once.
            Otherwise, queued tasks can exceed per_task_timeout before they even start.
    """

    if not rows:
        return []

    max_workers = max(1, int(max_workers))
    max_inflight = max(1, int(max_inflight))
    per_task_timeout = float(per_task_timeout)
    watch_interval = float(watch_interval)

    sem = threading.Semaphore(max_inflight)

    def task(idx: int, row: Dict[str, Any]):
        with sem:
            q = gen.generate(row)
            return idx, q

    results_by_idx: Dict[int, Dict[str, Any]] = {}
    completed = 0
    total = len(rows)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    start_time = time.time()
    last_log = start_time
    pbar = tqdm(total=total, desc="Generating questions", unit="row")

    try:
        future_to_idx: Dict[Any, int] = {}
        submit_time: Dict[Any, float] = {}

        row_iter = iter(list(enumerate(rows)))

        def submit_next() -> bool:
            try:
                idx, row = next(row_iter)
            except StopIteration:
                return False
            fut = executor.submit(task, idx, row)
            future_to_idx[fut] = idx
            submit_time[fut] = time.time()
            pending.add(fut)
            return True

        pending: set[Any] = set()
        # Keep only a bounded number of in-flight tasks.
        initial = min(max_inflight, total)
        for _ in range(initial):
            if not submit_next():
                break

        while pending:
            done, not_done = wait(
                pending, timeout=watch_interval, return_when=FIRST_COMPLETED
            )

            newly_completed = 0

            for fut in done:
                idx = future_to_idx[fut]
                try:
                    _idx, q = fut.result()
                    results_by_idx[idx] = {**rows[idx], "question": str(q).strip()}
                except Exception as e:
                    results_by_idx[idx] = {
                        **rows[idx],
                        "question": "",
                        "llm_error": f"future_error: {e}",
                    }
                completed += 1
                newly_completed += 1

            now = time.time()
            timed_out: List[Any] = []
            if per_task_timeout > 0:
                for fut in not_done:
                    if now - submit_time[fut] > per_task_timeout:
                        idx = future_to_idx[fut]
                        timed_out.append(fut)
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        results_by_idx[idx] = {
                            **rows[idx],
                            "question": "",
                            "llm_error": f"timeout({per_task_timeout}s)",
                        }
                        completed += 1
                        newly_completed += 1

            if newly_completed:
                pbar.update(newly_completed)

            for fut in done:
                pending.discard(fut)
            for fut in timed_out:
                pending.discard(fut)

            # Refill up to max_inflight
            while len(pending) < max_inflight and completed + len(pending) < total:
                if not submit_next():
                    break

            if now - last_log >= 5 or completed == total:
                elapsed = now - start_time
                avg = (elapsed / completed) if completed else 0.0
                eta = max(0.0, avg * (total - completed))
                pbar.set_postfix(
                    {
                        "elapsed_s": f"{elapsed:.1f}",
                        "eta_s": f"{eta:.1f}",
                    }
                )
                last_log = now

            if completed >= total:
                break

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        pbar.close()

    # Preserve input order
    return [results_by_idx[i] for i in range(total)]


def _unique_answer_types_hint(row: Dict[str, Any]) -> List[str]:
    nodes = row.get("answer_nodes_sample")
    if not isinstance(nodes, list):
        return []
    types: List[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        ts = n.get("types")
        if isinstance(ts, list):
            for t in ts:
                if isinstance(t, str) and t not in types:
                    types.append(t)
    return types


def _pattern_description(row: Dict[str, Any]) -> str:
    qt = row.get("query_type")
    ans_types = _unique_answer_types_hint(row)
    ans_types_text = ", ".join(ans_types) if ans_types else "(unknown)"

    if qt == "one_hop_chain":
        return (
            "Pattern: one-hop. Find entities x such that (anchor)-[rel_type]-(x).\n"
            f"anchor_name: {row.get('anchor_name')}\n"
            f"anchor_types: {row.get('anchor_types')}\n"
            f"rel_type: {row.get('rel_type')}\n"
            f"answer_types_hint: {ans_types_text}\n"
            f"answer_count: {row.get('answer_count')}"
        )

    if qt == "two_hop_chain":
        # 中間ノード情報を抽出
        mid_nodes = row.get("mid_nodes_sample", [])
        if mid_nodes and isinstance(mid_nodes, list) and len(mid_nodes) > 0:
            mid_node = mid_nodes[0]
            mid_name = mid_node.get("name", "unknown")
            mid_types = mid_node.get("types", [])
            mid_types_text = ", ".join(mid_types) if isinstance(mid_types, list) else str(mid_types)
        else:
            mid_name = "(unknown intermediate)"
            mid_types_text = "(unknown)"

        anchor_name = row.get('anchor_name', '')
        rel1 = row.get('rel1', '')
        rel2 = row.get('rel2', '')

        # rel2に基づいて適切な質問形式を決定
        if rel2 == "parent-child":
            question_verb = "subcategories" if mid_types_text in ["pathway", "biological_process", "molecular_function"] else "subtypes"
        else:
            question_verb = f"{rel2} entities"

        return (
            "Pattern: TWO-HOP CHAIN\n\n"
            "SIMPLE QUESTION FORMAT REQUIRED:\n"
            f"  'What are the {question_verb} of {mid_name}, a {rel1} of {anchor_name}?'\n\n"
            "MUST INCLUDE:\n"
            f"  1. \"{anchor_name}\" (the starting point)\n"
            f"  2. \"{mid_name}\" (the intermediate entity)\n"
            f"  3. Ask for {ans_types_text} (the answers we want)\n\n"
            "Graph Path:\n"
            f"  {anchor_name} --({rel1})--> {mid_name} --({rel2})--> [ANSWER]\n\n"
            "Details:\n"
            f"  - Anchor: {anchor_name} ({row.get('anchor_types')})\n"
            f"  - Intermediate: {mid_name} ({mid_types_text})\n"
            f"  - Answer type: {ans_types_text}\n"
            f"  - Relations: {rel1}, {rel2}\n\n"
            "EXAMPLE QUESTIONS (pick a style):\n"
            f"  1. 'What {ans_types_text} are {rel2} of {mid_name}, itself a {rel1} of {anchor_name}?'\n"
            f"  2. 'Which {ans_types_text} belong to {mid_name} (a subcategory of {anchor_name})?'\n"
            f"  3. '{mid_name} is a {rel1} of {anchor_name}. What are its {rel2}?'\n\n"
            "DO NOT ASK:\n"
            f"  - 'How does X relate to Y?' or 'What is the relationship between X and Y?'\n"
            f"  - 'What role does X play in Y?' or 'What is the significance of X?'"
        )

    if qt == "two_anchor_intersection":
        return (
            "Pattern: two-anchor intersection. Find entities y such that "
            "(anchorA)-[anchorA_edge_type]-(y) AND (anchorB)-[anchorB_edge_type]-(y).\n"
            f"anchorA_name: {row.get('anchorA_name')}\n"
            f"anchorA_types: {row.get('anchorA_types')}\n"
            f"anchorA_edge_type: {row.get('anchorA_edge_type')}\n"
            f"anchorB_name: {row.get('anchorB_name')}\n"
            f"anchorB_types: {row.get('anchorB_types')}\n"
            f"anchorB_edge_type: {row.get('anchorB_edge_type')}\n"
            f"answer_types_hint: {ans_types_text}\n"
            f"answer_count: {row.get('answer_count')}"
        )

    if qt == "three_anchor_intersection":
        return (
            "Pattern: three-anchor intersection. Find entities y such that "
            "(anchorA)-[anchorA_edge_type]-(y) AND (anchorB)-[anchorB_edge_type]-(y) AND (anchorC)-[anchorC_edge_type]-(y).\n"
            f"anchorA_name: {row.get('anchorA_name')}\n"
            f"anchorA_types: {row.get('anchorA_types')}\n"
            f"anchorA_edge_type: {row.get('anchorA_edge_type')}\n"
            f"anchorB_name: {row.get('anchorB_name')}\n"
            f"anchorB_types: {row.get('anchorB_types')}\n"
            f"anchorB_edge_type: {row.get('anchorB_edge_type')}\n"
            f"anchorC_name: {row.get('anchorC_name')}\n"
            f"anchorC_types: {row.get('anchorC_types')}\n"
            f"anchorC_edge_type: {row.get('anchorC_edge_type')}\n"
            f"answer_types_hint: {ans_types_text}\n"
            f"answer_count: {row.get('answer_count')}"
        )

    return (
        "Pattern: unknown. Use the provided JSON row fields to generate a question.\n"
        + json.dumps(row, ensure_ascii=False)
    )


class QuestionGenerator:
    def __init__(self, *, model: str, temperature: float, lang: str):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        # Temperatureを少し上げる(0.2 -> 0.4~0.7)と、表現の多様性が増します
        llm = init_chat_model(model, model_provider="openai", temperature=0.5)
        self._model = llm.with_structured_output(QuestionResponse)
        self._lang = lang

    def _validate_two_hop_question(self, question: str, row: Dict[str, Any]) -> bool:
        """two_hop_chainの質問が両方のエンティティを含み、正しい形式かチェック"""
        # query_typeを推測: rel1/rel2があり、mid_nodes_sampleがあればtwo_hop_chain
        is_two_hop = (
            "rel1" in row and "rel2" in row and
            row.get("mid_nodes_sample") and
            len(row.get("mid_nodes_sample", [])) > 0
        )

        if not is_two_hop:
            return True

        anchor_name = row.get("anchor_name", "")
        mid_nodes = row.get("mid_nodes_sample", [])
        mid_name = mid_nodes[0].get("name", "") if mid_nodes else ""

        if not mid_name:
            return True

        q_lower = question.lower()
        anchor_in_q = anchor_name.lower() in q_lower if anchor_name else True
        mid_in_q = mid_name.lower() in q_lower

        # 両方のエンティティを含むかチェック
        if not (anchor_in_q and mid_in_q):
            return False

        # A-B関係を聞く質問や曖昧な質問はリジェクト
        bad_patterns = [
            "how does", "how is", "how are",
            "relate to", "related to", "relationship between",
            "connection between", "link between",
            "what role", "what is the role", "what is the significance",
            "what is the importance", "what is the function"
        ]
        for pattern in bad_patterns:
            if pattern in q_lower:
                return False

        # 良いパターンを含むかチェック（少なくとも1つは含むべき）
        good_patterns = [
            "what are", "which are", "what ", "which ",
            "subcategories", "subtypes", "subclasses",
            "belong to", "part of", "types of"
        ]
        has_good_pattern = any(pattern in q_lower for pattern in good_patterns)

        return has_good_pattern

    def generate(self, row: Dict[str, Any]) -> str:
        desc = _pattern_description(row)

        # 言語設定
        if self._lang.lower().startswith("ja"):
            lang_instruction = "Output Language: Japanese (Keep technical terms/entity names in English if appropriate for the domain)."
        else:
            lang_instruction = "Output Language: English"

        # プロンプトの構築
        prompt = f"""
You are an expert researcher in the biomedical domain (or the specific domain of the provided entities).
Your task is to convert a structured Knowledge Graph query pattern into a natural, fluent question.

### Input Data
The user provides a "Row description" describing a query path on a graph.
- Anchor: The starting entity.
- Relations: The edges traversed.
- Target: The type of answer we are looking for.

### Guidelines for Natural Generation
1. **Avoid Graph Jargon**: Do NOT use words like "anchor", "hop", "edge", "node", "starting from", "via relation", or "two levels of".
2. **Use Domain Verbs**:
   - Instead of "related to", use specific verbs based on the entity types (e.g., "treats", "targets", "expresses", "is involved in", "causes").
   - If the relation is "parent-child", use "is a type of", "belongs to the category of", or "is a subclass of".
3. **Hide the Structure**:
   - For 2-hop chains (A->B->C), phrase it as a single cohesive question (e.g., "What drug targets proteins associated with Disease X?").
   - For Intersections (A & B -> C), use "both" or "and" naturally (e.g., "Which gene is targeted by Drug A and also associated with Disease B?").
4. **Be Concise but Specific**: The question should be answerable by a domain expert without seeing the graph.
5. **Context-Aware Verbs (Crucial)**:
   - If the relation is "parent-child" but the types are DIFFERENT (e.g., Disease -> Gene), do NOT use "subtype". Use "associated with" or "implicated in".
   - If asking about Gene-Gene expression, use "**co-expressed with**" instead of "expressed in".
   - "Expressed in" should only be used for Anatomy/Tissues (e.g., "expressed in the Liver").

6. **Logical Consistency**:
   - A Gene cannot be a "subtype" of a Disease.
   - A Drug cannot "interact" with a Disease (it "treats" or "causes" it).
### Examples (English)
[Bad]: Which exposure is related through two levels of parent-child relationships starting from X?
[Good]: What represents a broader category of the exposure X? (or "What is the grandparent class of X?")

[Bad]: Which gene is connected to Drug A via target and Disease B via association?
[Good]: Which gene is targeted by Sunitinib and is also implicated in Renal Cell Carcinoma?

[Bad]: What is the result of the intersection of A and B?
[Good]: What biological process is shared between A and B?

### Task
{lang_instruction}

Row description:
{desc}

Return JSON with a single key "question".
""".strip()

        # two_hop_chainの場合はリトライ付きで生成
        max_retries = 3
        for attempt in range(max_retries):
            res = self._model.invoke(prompt)
            question = res.question.strip()

            is_valid = self._validate_two_hop_question(question, row)
            if is_valid:
                print(res, flush=True)
                return question

            # リトライプロンプト：より強調
            if attempt < max_retries - 1:
                anchor_name = row.get("anchor_name", "")
                mid_nodes = row.get("mid_nodes_sample", [])
                mid_name = mid_nodes[0].get("name", "") if mid_nodes else ""
                prompt = f"""
Your previous answer was REJECTED because it did not contain BOTH required entity names.

REQUIRED ENTITY NAMES (must appear EXACTLY in your question):
1. "{anchor_name}"
2. "{mid_name}"

Previous rejected question: "{question}"

Generate a NEW question that includes BOTH entity names above. Copy and paste them directly.

{desc}

{lang_instruction}
Return JSON with a single key "question".
""".strip()

        # 最後のリトライでも失敗した場合は最後の結果を返す
        print(f"Warning: Could not generate valid question after {max_retries} attempts", flush=True)
        print(res, flush=True)
        return question


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate natural-language questions from dataset_construction JSONL rows using an LLM. "
            "Adds a 'question' field to each row."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="in_path", type=Path, help="Input JSONL file")
    src.add_argument(
        "--json",
        dest="json_str",
        type=str,
        help="Single JSON object string (one row)",
    )
    src.add_argument(
        "--stdin",
        action="store_true",
        help="Read a single JSON object from stdin",
    )

    p.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    p.add_argument("--model", type=str, default=BASEMODEL)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "ja"],
        help="Question language (default: en)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, process at most this many rows",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Generate questions in parallel (ThreadPool).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max worker threads when --parallel (default: 8)",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=16,
        help="Max concurrent LLM calls when --parallel (default: 16)",
    )
    p.add_argument(
        "--per-task-timeout",
        type=float,
        default=300.0,
        help="Per-row timeout seconds when --parallel (default: 300)",
    )
    p.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        help="Watcher loop interval seconds when --parallel (default: 5)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if exists",
    )

    args = p.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"output exists (use --overwrite): {args.out}")

    gen = QuestionGenerator(
        model=args.model, temperature=args.temperature, lang=args.lang
    )

    if args.in_path is not None:
        input_rows: List[Dict[str, Any]] = []
        for i, row in enumerate(_iter_jsonl(args.in_path), 1):
            if args.max_rows and i > args.max_rows:
                break
            input_rows.append(row)

        if args.parallel:
            rows = _generate_questions_parallel(
                gen=gen,
                rows=input_rows,
                max_workers=args.max_workers,
                max_inflight=args.max_inflight,
                per_task_timeout=args.per_task_timeout,
                watch_interval=args.watch_interval,
            )
        else:
            rows = []
            for row in tqdm(input_rows, desc="Generating questions", unit="row"):
                q = gen.generate(row)
                rows.append({**row, "question": q})

    elif args.json_str is not None:
        row = json.loads(args.json_str)
        q = gen.generate(row)
        rows = [{**row, "question": q}]

    else:
        raw = (sys.stdin.read() or "").strip()
        if not raw:
            raise SystemExit("stdin is empty")
        row = json.loads(raw)
        q = gen.generate(row)
        rows = [{**row, "question": q}]

    _write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()

"""
docker compose run --rm app python dataset_construction/generate_questions.py  --in /app/result/dataset_splits/all/train.jsonl --out /app/result/dataset_construction/all_with_questions_v2.jsonl --lang en --parallel --max-workers 32 --max-inflight 64 --per-task-timeout 300

"""
