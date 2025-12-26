# Cypher query for 1-hop chain
one_hop_chain_cypher = """
CALL {
  WITH 2 AS degMin, 200 AS degMax, 0.02 AS anchorKeepProb
  MATCH (a)
  WITH a, COUNT{ (a)--() } AS deg
  WHERE degMin <= deg AND deg <= degMax
    AND rand() < anchorKeepProb
  RETURN a
  LIMIT 500
}

MATCH (a)-[r]-(x)
WITH a,
     type(r) AS rel,
     count(DISTINCT x) AS answer_count,
  collect(DISTINCT x)[0..50] AS answers
WHERE answer_count >= 1 AND answer_count <= 50

RETURN
  id(a) AS anchor_id,
  rel   AS rel_type,
  answer_count,
  [n IN answers | id(n)] AS answer_ids_sample
ORDER BY rand()
LIMIT 100;
"""


def build_one_hop_chain_cypher(
    *,
    node_label: str | None = "Entity",
    anchor_keep_prob: float = 0.02,
    deg_min: int = 2,
    deg_max: int = 200,
    anchor_pool: int = 500,
    min_answers: int = 1,
    max_answers: int = 50,
    limit: int = 100,
) -> str:
    lbl = f":{node_label}" if node_label else ""
    return f"""
CALL {{
  MATCH (a{lbl})
  WITH a, COUNT {{ (a)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
    AND rand() < {anchor_keep_prob}
  RETURN a
  LIMIT {anchor_pool}
}}

MATCH (a)-[r]-(x{lbl})
WITH a,
     type(r) AS rel,
     count(DISTINCT x) AS answer_count,
     collect(DISTINCT x)[0..{max_answers}] AS answers
WHERE answer_count >= {min_answers} AND answer_count <= {max_answers}

RETURN
  elementId(a) AS anchor_id,
  a.name       AS anchor_name,
  coalesce(a.type, labels(a)) AS anchor_types,
  rel          AS rel_type,
  answer_count,
  [n IN answers | elementId(n)] AS answer_ids_sample,
  [n IN answers | {{id: elementId(n), name: n.name, types: coalesce(n.type, labels(n))}}] AS answer_nodes_sample
LIMIT {limit};
"""


two_hop_chain_cypher = """
CALL {
  WITH 2 AS degMin, 200 AS degMax, 0.02 AS anchorKeepProb
  MATCH (a)
  WITH a, COUNT{ (a)--() } AS deg
  WHERE degMin <= deg AND deg <= degMax
    AND rand() < anchorKeepProb
  RETURN a
  LIMIT 500
}

MATCH (a)-[r1]-(z)-[r2]-(x)
WHERE a <> z AND z <> x AND a <> x

WITH a,
     type(r1) AS rel1,
     type(r2) AS rel2,
     count(DISTINCT x) AS answer_count,
  collect(DISTINCT x)[0..50] AS answers
WHERE answer_count >= 1 AND answer_count <= 50

RETURN
  id(a) AS anchor_id,
  rel1, rel2,
  answer_count,
  [n IN answers | id(n)] AS answer_ids_sample
ORDER BY rand()
LIMIT 100;
"""


def build_two_hop_chain_cypher(
    *,
    node_label: str | None = "Entity",
    anchor_keep_prob: float = 0.02,
    deg_min: int = 2,
    deg_max: int = 200,
    anchor_pool: int = 500,
    min_answers: int = 1,
    max_answers: int = 50,
    limit: int = 100,
) -> str:
    lbl = f":{node_label}" if node_label else ""
    return f"""
CALL {{
  MATCH (a{lbl})
  WITH a, COUNT {{ (a)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
    AND rand() < {anchor_keep_prob}
  RETURN a
  LIMIT {anchor_pool}
}}

MATCH (a)-[r1]-(z{lbl})-[r2]-(x{lbl})
WHERE a <> z AND z <> x AND a <> x

WITH a,
     type(r1) AS rel1,
     type(r2) AS rel2,
     count(DISTINCT x) AS answer_count,
     collect(DISTINCT x)[0..{max_answers}] AS answers
WHERE answer_count >= {min_answers} AND answer_count <= {max_answers}

RETURN
  elementId(a) AS anchor_id,
  a.name       AS anchor_name,
  coalesce(a.type, labels(a)) AS anchor_types,
  rel1, rel2,
  answer_count,
  [n IN answers | elementId(n)] AS answer_ids_sample,
  [n IN answers | {{id: elementId(n), name: n.name, types: coalesce(n.type, labels(n))}}] AS answer_nodes_sample
LIMIT {limit};
"""


two_anchor_intersection_cypher = """
// 共通点になりうるノード x を先に集める（degree帯で絞る）
CALL {
  WITH 2 AS degMin, 200 AS degMax, 0.03 AS anchorKeepProb
  MATCH (x)
  WITH x, COUNT{ (x)--() } AS deg
  WHERE degMin <= deg AND deg <= degMax
    AND rand() < anchorKeepProb
  RETURN collect(x)[0..800] AS xs
}

WITH xs, 5000 AS numTries, 1 AS minAnswers, 50 AS maxAnswers
WHERE size(xs) >= 1

UNWIND range(1, numTries) AS i
WITH xs, minAnswers, maxAnswers,
     xs[toInteger(rand() * size(xs))] AS x
WHERE x IS NOT NULL

// x の近傍からランダムに2つのアンカー a,b を引く（関係タイプは問わない）
CALL {
  WITH x
  MATCH (x)-[ra]-(a)
  RETURN a, ra
  ORDER BY rand()
  LIMIT 2
}
WITH x, collect({a:a, ra:ra}) AS picks, minAnswers, maxAnswers
WHERE size(picks) = 2 AND id(picks[0].a) <> id(picks[1].a)

WITH picks[0].a AS a, picks[1].a AS b,
     type(picks[0].ra) AS relA,
     type(picks[1].ra) AS relB,
     minAnswers, maxAnswers

// a と b の共通近傍集合（ラベル・関係タイプともに自由）
MATCH (a)-[rA]-(y)
MATCH (b)-[rB]-(y)

WITH a, b, relA, relB,
     count(DISTINCT y) AS answer_count,
  collect(DISTINCT y)[0..50] AS answers
WHERE answer_count >= 1 AND answer_count <= 50

RETURN
  id(a) AS anchorA_id,
  id(b) AS anchorB_id,
  relA  AS anchorA_edge_type,
  relB  AS anchorB_edge_type,
  answer_count,
  [n IN answers | id(n)] AS answer_ids_sample
LIMIT 100;
"""


def build_two_anchor_intersection_cypher(
    *,
    node_label: str | None = "Entity",
    anchor_keep_prob: float = 0.03,
    deg_min: int = 2,
    deg_max: int = 200,
    anchor_pool: int = 800,
    num_tries: int = 5000,
    min_answers: int = 1,
    max_answers: int = 50,
    limit: int = 100,
) -> str:
    lbl = f":{node_label}" if node_label else ""
    return f"""
// 共通点になりうるノード x を先に集める（degree帯で絞る）
CALL {{
  MATCH (x{lbl})
  WITH x, COUNT {{ (x)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
    AND rand() < {anchor_keep_prob}
  RETURN x
  LIMIT {anchor_pool}
}}

WITH collect(x) AS xs
WHERE size(xs) >= 1

UNWIND range(1, {num_tries}) AS i
WITH xs,
     xs[toInteger(rand() * size(xs))] AS x
WHERE x IS NOT NULL

// x の近傍からランダムに2つのアンカー a,b を引く（関係タイプは問わない）
CALL {{
  WITH x
  MATCH (x)-[ra]-(a{lbl})
  RETURN a, ra
  ORDER BY rand()
  LIMIT 2
}}
WITH x, collect({{a:a, ra:ra}}) AS picks
WHERE size(picks) = 2 AND elementId(picks[0].a) <> elementId(picks[1].a)

WITH picks[0].a AS a, picks[1].a AS b,
     type(picks[0].ra) AS relA,
     type(picks[1].ra) AS relB

// a と b の共通近傍集合（ラベル・関係タイプともに自由）
MATCH (a)-[rA]-(y{lbl})
MATCH (b)-[rB]-(y{lbl})

WITH a, b, relA, relB,
     count(DISTINCT y) AS answer_count,
     collect(DISTINCT y)[0..{max_answers}] AS answers
WHERE answer_count >= {min_answers} AND answer_count <= {max_answers}

RETURN
  elementId(a) AS anchorA_id,
  a.name       AS anchorA_name,
  coalesce(a.type, labels(a)) AS anchorA_types,
  elementId(b) AS anchorB_id,
  b.name       AS anchorB_name,
  coalesce(b.type, labels(b)) AS anchorB_types,
  relA         AS anchorA_edge_type,
  relB         AS anchorB_edge_type,
  answer_count,
  [n IN answers | elementId(n)] AS answer_ids_sample,
  [n IN answers | {{id: elementId(n), name: n.name, types: coalesce(n.type, labels(n))}}] AS answer_nodes_sample
LIMIT {limit};
"""


theree_anchor_intersection_cypher = """
CALL {
  WITH 2 AS degMin, 200 AS degMax, 0.03 AS anchorKeepProb
  MATCH (x)
  WITH x, COUNT{ (x)--() } AS deg
  WHERE degMin <= deg AND deg <= degMax
    AND rand() < anchorKeepProb
  RETURN collect(x)[0..800] AS xs
}

WITH xs, 8000 AS numTries, 1 AS minAnswers, 50 AS maxAnswers
WHERE size(xs) >= 1

UNWIND range(1, numTries) AS i
WITH xs, minAnswers, maxAnswers,
     xs[toInteger(rand() * size(xs))] AS x
WHERE x IS NOT NULL

CALL {
  WITH x
  MATCH (x)-[ra]-(a)
  RETURN a, ra
  ORDER BY rand()
  LIMIT 3
}
WITH x, collect({a:a, ra:ra}) AS picks, minAnswers, maxAnswers
WHERE size(picks) = 3
  AND id(picks[0].a) <> id(picks[1].a)
  AND id(picks[0].a) <> id(picks[2].a)
  AND id(picks[1].a) <> id(picks[2].a)

WITH picks[0].a AS a, picks[1].a AS b, picks[2].a AS c,
     type(picks[0].ra) AS relA,
     type(picks[1].ra) AS relB,
     type(picks[2].ra) AS relC,
     minAnswers, maxAnswers

MATCH (a)-[r1]-(y)
MATCH (b)-[r2]-(y)
MATCH (c)-[r3]-(y)

WITH a, b, c, relA, relB, relC,
     count(DISTINCT y) AS answer_count,
  collect(DISTINCT y)[0..50] AS answers
WHERE answer_count >= 1 AND answer_count <= 50

RETURN
  id(a) AS anchorA_id,
  id(b) AS anchorB_id,
  id(c) AS anchorC_id,
  relA AS anchorA_edge_type,
  relB AS anchorB_edge_type,
  relC AS anchorC_edge_type,
  answer_count,
  [n IN answers | id(n)] AS answer_ids_sample
LIMIT 50;
"""


def build_three_anchor_intersection_cypher(
    *,
    node_label: str | None = "Entity",
    anchor_keep_prob: float = 0.03,
    deg_min: int = 2,
    deg_max: int = 200,
    anchor_pool: int = 800,
    num_tries: int = 8000,
    min_answers: int = 1,
    max_answers: int = 50,
    limit: int = 50,
) -> str:
    lbl = f":{node_label}" if node_label else ""
    return f"""
CALL {{
  MATCH (x{lbl})
  WITH x, COUNT {{ (x)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
    AND rand() < {anchor_keep_prob}
  RETURN x
  LIMIT {anchor_pool}
}}

WITH collect(x) AS xs
WHERE size(xs) >= 1

UNWIND range(1, {num_tries}) AS i
WITH xs,
     xs[toInteger(rand() * size(xs))] AS x
WHERE x IS NOT NULL

CALL {{
  WITH x
  MATCH (x)-[ra]-(a{lbl})
  RETURN a, ra
  ORDER BY rand()
  LIMIT 3
}}
WITH x, collect({{a:a, ra:ra}}) AS picks
WHERE size(picks) = 3
  AND elementId(picks[0].a) <> elementId(picks[1].a)
  AND elementId(picks[0].a) <> elementId(picks[2].a)
  AND elementId(picks[1].a) <> elementId(picks[2].a)

WITH picks[0].a AS a, picks[1].a AS b, picks[2].a AS c,
     type(picks[0].ra) AS relA,
     type(picks[1].ra) AS relB,
     type(picks[2].ra) AS relC

MATCH (a)-[r1]-(y{lbl})
MATCH (b)-[r2]-(y{lbl})
MATCH (c)-[r3]-(y{lbl})

WITH a, b, c, relA, relB, relC,
     count(DISTINCT y) AS answer_count,
     collect(DISTINCT y)[0..{max_answers}] AS answers
WHERE answer_count >= {min_answers} AND answer_count <= {max_answers}

RETURN
  elementId(a) AS anchorA_id,
  a.name       AS anchorA_name,
  coalesce(a.type, labels(a)) AS anchorA_types,
  elementId(b) AS anchorB_id,
  b.name       AS anchorB_name,
  coalesce(b.type, labels(b)) AS anchorB_types,
  elementId(c) AS anchorC_id,
  c.name       AS anchorC_name,
  coalesce(c.type, labels(c)) AS anchorC_types,
  relA         AS anchorA_edge_type,
  relB         AS anchorB_edge_type,
  relC         AS anchorC_edge_type,
  answer_count,
  [n IN answers | elementId(n)] AS answer_ids_sample,
  [n IN answers | {{id: elementId(n), name: n.name, types: coalesce(n.type, labels(n))}}] AS answer_nodes_sample
LIMIT {limit};
"""


# Backward-compatible defaults (small samples)
one_hop_chain_cypher = build_one_hop_chain_cypher()
two_hop_chain_cypher = build_two_hop_chain_cypher()
two_anchor_intersection_cypher = build_two_anchor_intersection_cypher()
theree_anchor_intersection_cypher = build_three_anchor_intersection_cypher()


# ---------------------------------------------------------------------------
# Simple (fast) queries for quick sampling
# - No ORDER BY rand(), no large COLLECT, no tries loops.
# - Returns one answer per row; answer_count is set to 1.
# ---------------------------------------------------------------------------

simple_one_hop_chain_cypher = """
CALL {
  MATCH (a)
  WHERE rand() < $keep_prob
  RETURN a
  LIMIT $anchor_pool
}
WITH collect(a) AS anchors
WHERE size(anchors) > 0
UNWIND range(1, $limit) AS i
WITH anchors, apoc.coll.randomItem(anchors) AS a
CALL {
  WITH a
  MATCH (a)-[r]-(x)
  RETURN r, x
  ORDER BY rand()
  LIMIT 1
}
RETURN
  elementId(a) AS anchor_id,
  a.name       AS anchor_name,
  coalesce(a.type, labels(a)) AS anchor_types,
  type(r) AS rel_type,
  1 AS answer_count,
  [elementId(x)] AS answer_ids_sample,
  [{id: elementId(x), name: x.name, types: coalesce(x.type, labels(x))}] AS answer_nodes_sample;
"""


simple_two_hop_chain_cypher = """
CALL {
  MATCH (a)
  WHERE rand() < $keep_prob
  RETURN a
  LIMIT $anchor_pool
}
WITH collect(a) AS anchors
WHERE size(anchors) > 0
UNWIND range(1, $limit) AS i
WITH anchors, apoc.coll.randomItem(anchors) AS a
CALL {
  WITH a
  MATCH (a)-[r1]-(z)
  RETURN r1, z
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH a, z
  MATCH (z)-[r2]-(x)
  WHERE x <> a
  RETURN r2, x
  ORDER BY rand()
  LIMIT 1
}
RETURN
  elementId(a) AS anchor_id,
  a.name       AS anchor_name,
  coalesce(a.type, labels(a)) AS anchor_types,
  type(r1) AS rel1,
  type(r2) AS rel2,
  1 AS answer_count,
  [elementId(x)] AS answer_ids_sample,
  [{id: elementId(x), name: x.name, types: coalesce(x.type, labels(x))}] AS answer_nodes_sample;
"""


simple_two_anchor_intersection_cypher = """
CALL {
  MATCH (a)
  WHERE rand() < $keep_prob
  RETURN a
  LIMIT $anchor_pool
}
WITH collect(a) AS anchors
WHERE size(anchors) > 0
UNWIND range(1, $limit) AS i
WITH anchors, apoc.coll.randomItem(anchors) AS a
CALL {
  WITH a
  MATCH (a)-[ra]-(y)
  RETURN ra, y
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH a, y
  MATCH (y)-[rb]-(b)
  WHERE b <> a
  RETURN rb, b
  ORDER BY rand()
  LIMIT 1
}
RETURN
  elementId(a) AS anchorA_id,
  a.name       AS anchorA_name,
  coalesce(a.type, labels(a)) AS anchorA_types,
  elementId(b) AS anchorB_id,
  b.name       AS anchorB_name,
  coalesce(b.type, labels(b)) AS anchorB_types,
  type(ra) AS anchorA_edge_type,
  type(rb) AS anchorB_edge_type,
  1 AS answer_count,
  [elementId(y)] AS answer_ids_sample,
  [{id: elementId(y), name: y.name, types: coalesce(y.type, labels(y))}] AS answer_nodes_sample;
"""


simple_three_anchor_intersection_cypher = """
CALL {
  MATCH (a)
  WHERE rand() < $keep_prob
  RETURN a
  LIMIT $anchor_pool
}
WITH collect(a) AS anchors
WHERE size(anchors) > 0
UNWIND range(1, $limit) AS i
WITH anchors, apoc.coll.randomItem(anchors) AS a
CALL {
  WITH a
  MATCH (a)-[r1]-(y)
  RETURN r1, y
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH a, y
  MATCH (y)-[r2]-(b)
  WHERE b <> a
  RETURN r2, b
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH a, y, b
  MATCH (y)-[r3]-(c)
  WHERE c <> a AND c <> b
  RETURN r3, c
  ORDER BY rand()
  LIMIT 1
}
RETURN
  elementId(a) AS anchorA_id,
  a.name       AS anchorA_name,
  coalesce(a.type, labels(a)) AS anchorA_types,
  elementId(b) AS anchorB_id,
  b.name       AS anchorB_name,
  coalesce(b.type, labels(b)) AS anchorB_types,
  elementId(c) AS anchorC_id,
  c.name       AS anchorC_name,
  coalesce(c.type, labels(c)) AS anchorC_types,
  type(r1) AS anchorA_edge_type,
  type(r2) AS anchorB_edge_type,
  type(r3) AS anchorC_edge_type,
  1 AS answer_count,
  [elementId(y)] AS answer_ids_sample,
  [{id: elementId(y), name: y.name, types: coalesce(y.type, labels(y))}] AS answer_nodes_sample;
"""
