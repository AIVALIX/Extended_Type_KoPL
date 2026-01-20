#!/usr/bin/env python
import json
import sys

for line in sys.stdin:
    if not line.strip():
        continue
    d = json.loads(line)
    anchor = d.get("anchor_name", "")
    mid_nodes = d.get("mid_nodes_sample", [])
    mid = mid_nodes[0].get("name") if mid_nodes else "N/A"
    q = d.get("question", "")

    anchor_in_q = anchor.lower() in q.lower() if anchor else False
    mid_in_q = mid.lower() in q.lower() if mid and mid != "N/A" else False

    print(f"Anchor: {anchor}")
    print(f"  -> In question: {'YES' if anchor_in_q else 'NO'}")
    print(f"Mid: {mid}")
    print(f"  -> In question: {'YES' if mid_in_q else 'NO'}")
    print(f"Question: {q}")
    print(f"VALID: {'YES' if (anchor_in_q and mid_in_q) else 'NO'}")
    print("-" * 60)
