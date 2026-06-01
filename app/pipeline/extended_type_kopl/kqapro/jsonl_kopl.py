"""JSONL-KoPL driver for KQA-Pro.

The KQA-Pro Baselines RuleExecutor expects programs in the shape
``[{"function": str, "dependencies": [int, ...], "inputs": [str, ...]}, ...]``.

Our Phase 1 prompts the LLM to emit each program step as a compact JSON
line (``{"fn": "...", "args": [...], "deps": [...]}``). This module:

1. parses that LLM output into a program list,
2. infers missing ``deps`` from the function type (same logic the
   executor's ``forward`` uses),
3. executes the program via the vendored ``RuleExecutor``.

The executor returns a terminal value for query functions (``What``,
``Count``, ``SelectBetween``, ``SelectAmong``, ``QueryAttr``,
``QueryAttrUnderCondition``, ``VerifyStr``/``Num``/``Year``/``Date``,
``QueryRelation``, ``QueryAttrQualifier``, ``QueryRelationQualifier``)
so the caller can compare directly against the gold answer string.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pipeline.extended_type_kopl.kqapro.executor_rule import RuleExecutor

logger = logging.getLogger(__name__)


# Terminal functions whose output is the final answer.
TERMINAL_FUNCTIONS = {
    "What",
    "Count",
    "SelectBetween",
    "SelectAmong",
    "QueryAttr",
    "QueryAttrUnderCondition",
    "VerifyStr",
    "VerifyNum",
    "VerifyYear",
    "VerifyDate",
    "QueryRelation",
    "QueryAttrQualifier",
    "QueryRelationQualifier",
}


def parse_jsonl_kopl(text: str) -> List[Dict[str, Any]]:
    """Parse a JSONL-KoPL blob into the executor's list-of-dicts format.

    - Accepts lines that are complete JSON objects (``{"fn": "...", "args": [...]}``).
    - Skips markdown code fences (```jsonl, ```json, ```).
    - Infers missing ``deps`` using the same branch-stack logic as
      ``RuleExecutor.forward`` so the LLM can be lenient about dep indices.
    """
    program: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("```") or line.endswith("```"):
            continue
        # Some LLMs prefix lines like "Step 1:" or "- ". Strip those off.
        if line[:1] in ("-", "*"):
            line = line[1:].strip()
        if ":" in line and not line.startswith("{"):
            # "1: {...}" or "Step 1: {...}"
            _, _, rest = line.partition(":")
            line = rest.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        fn = obj.get("fn") or obj.get("function")
        if not fn:
            continue
        args = obj.get("args") or obj.get("inputs") or []
        deps = obj.get("deps") or obj.get("dependencies")
        program.append({
            "function": fn,
            "inputs": [str(a) for a in args],
            "dependencies": list(deps) if deps is not None else None,
        })

    # Fill missing deps using the executor's branch-stack heuristic.
    branch_stack: List[int] = []
    for i, step in enumerate(program):
        if step["dependencies"] is not None:
            # Still feed the stack so subsequent steps can use it.
            fn = step["function"]
            if fn in {"FindAll", "Find"}:
                branch_stack.append(i - 1)
            elif fn in {"And", "Or", "SelectBetween", "QueryRelation", "QueryRelationQualifier"}:
                if branch_stack:
                    branch_stack.pop()
            continue
        fn = step["function"]
        if fn in {"FindAll", "Find"}:
            step["dependencies"] = []
            branch_stack.append(i - 1)
        elif fn in {"And", "Or", "SelectBetween", "QueryRelation", "QueryRelationQualifier"}:
            left = branch_stack[-1] if branch_stack else i - 1
            right = i - 1
            step["dependencies"] = [left, right]
            if branch_stack:
                branch_stack.pop()
        else:
            step["dependencies"] = [i - 1] if i > 0 else []
    return program


def execute_program(executor: RuleExecutor, program: List[Dict[str, Any]]) -> Tuple[Any, Optional[str]]:
    """Run a parsed program through the RuleExecutor.

    Returns (result, error). ``result`` is whatever the final step
    produced — a string for query functions, a (entity_ids, facts) tuple
    for locating-only programs (useful for debugging). ``error`` is None
    on success, otherwise a short diagnostic string.
    """
    if not program:
        return None, "empty program"

    memory: List[Any] = []
    for i, step in enumerate(program):
        fn = step["function"]
        method = getattr(executor, fn, None)
        if method is None:
            return None, f"unknown function at step {i}: {fn!r}"
        dep_idxs = step.get("dependencies") or []
        try:
            dep_vals = [memory[d] for d in dep_idxs]
        except IndexError:
            return None, f"bad dep at step {i}: deps={dep_idxs}, memory_size={len(memory)}"
        try:
            out = method(dep_vals, step.get("inputs", []))
        except Exception as e:
            return None, f"exec error at step {i} ({fn}): {type(e).__name__}: {e}"
        memory.append(out)

    return memory[-1], None


def result_to_string(result: Any) -> str:
    """Normalise an executor result to a comparable answer string."""
    if result is None:
        return ""
    if isinstance(result, tuple):
        # locating step left at end — no scalar answer
        return ""
    return str(result)


_executor_singleton: Optional[RuleExecutor] = None


def get_executor(kb_path: str = "/app/data/kqapro/kb.json") -> RuleExecutor:
    """Lazily build a shared RuleExecutor (kb.json load is ~5s)."""
    global _executor_singleton
    if _executor_singleton is None:
        _executor_singleton = RuleExecutor(vocab={}, kb_json=kb_path)
    return _executor_singleton
