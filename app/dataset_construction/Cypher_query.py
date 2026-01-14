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
// Reverse lookup (answer-first)
// 1) Sample candidate answers x (optionally degree-constrained to avoid hubs)
CALL {{
  // IMPORTANT: avoid full label scans with rand(); sample by internal id.
  WITH toInteger($max_node_id) AS maxId
  UNWIND range(1, {anchor_pool} * 10) AS i
  WITH maxId, toInteger(rand() * (maxId + 1)) AS rid
  MATCH (x{lbl})
  WHERE id(x) = rid
  WITH DISTINCT x
  LIMIT {anchor_pool}
  WITH x, COUNT {{ (x)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
  RETURN x
}}

// 2) Pick one random neighbor anchor a and relation r for each x
CALL {{
  WITH x
  MATCH (a{lbl})-[r]-(x)
  WITH a, r
  ORDER BY rand()
  LIMIT 1
  RETURN a, r
}}

// 3) Validate answer set size for (a, rel)
CALL {{
  WITH a, r
  MATCH (a)-[vr]-(vx{lbl})
  WHERE type(vr) = type(r)
  WITH collect(DISTINCT vx)[0..{max_answers + 1}] AS answers
  RETURN answers, size(answers) AS answer_count
}}
WITH a, r, answers, answer_count
WHERE answer_count >= {min_answers} AND answer_count <= {max_answers}

RETURN
  elementId(a) AS anchor_id,
  a.name       AS anchor_name,
  coalesce(a.type, labels(a)) AS anchor_types,
  type(r)      AS rel_type,
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
// Reverse lookup (answer-first) for 2-hop: a -[r1]- z -[r2]- x
// 1) Sample candidate answers x (degree-constrained to avoid hubs)
CALL {{
  // IMPORTANT: avoid full label scans with rand(); sample by internal id.
  WITH toInteger($max_node_id) AS maxId
  UNWIND range(1, {anchor_pool} * 10) AS i
  WITH maxId, toInteger(rand() * (maxId + 1)) AS rid
  MATCH (x{lbl})
  WHERE id(x) = rid
  WITH DISTINCT x
  LIMIT {anchor_pool}
  WITH x, COUNT {{ (x)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
  RETURN x
}}

// 2) From each x, pick one random (z, r2, a, r1) path backwards.
//    IMPORTANT: avoid scanning/ordering large neighbor sets.
//    We use a small retry loop: repeatedly sample a random neighbor, then
//    keep it only if it satisfies degree constraints.
CALL {{
  WITH x

  // 2-a) pick z among neighbors of x (bounded retries)
  CALL {{
    WITH x
    UNWIND range(1, 20) AS i
    MATCH (x)-[r2]-(z{lbl})
    WHERE x <> z
    WITH x, z, r2
    ORDER BY rand()
    LIMIT 1
    WITH x, z, r2, COUNT {{ (z)--() }} AS zdeg
    WHERE {deg_min} <= zdeg AND zdeg <= {deg_max}
    RETURN x AS x2, z, r2
    LIMIT 1
  }}

  // 2-b) pick a among neighbors of z (bounded retries)
  CALL {{
    WITH x2, z, r2
    UNWIND range(1, 20) AS j
    MATCH (z)-[r1]-(a{lbl})
    WHERE a <> z AND a <> x2
    WITH a, r1
    ORDER BY rand()
    LIMIT 1
    WITH a, r1, COUNT {{ (a)--() }} AS adeg
    WHERE {deg_min} <= adeg AND adeg <= {deg_max}
    RETURN a, r1
    LIMIT 1
  }}

  RETURN a, z, r1, r2
}}

// 3) Validate answer set size for (a, rel1, rel2)
CALL {{
  WITH a, r1, r2
  MATCH (a)-[vr1]-(vz{lbl})-[vr2]-(vx{lbl})
  WHERE type(vr1) = type(r1) AND type(vr2) = type(r2)
    AND a <> vz AND vz <> vx AND a <> vx
  WITH DISTINCT vx
  LIMIT {max_answers + 1}
  WITH collect(vx) AS answers
  RETURN answers, size(answers) AS answer_count
}}
WITH a, z, r1, r2, answers, answer_count
WHERE answer_count >= {min_answers} AND answer_count <= {max_answers}

RETURN
  elementId(a) AS anchor_id,
  a.name       AS anchor_name,
  coalesce(a.type, labels(a)) AS anchor_types,
  type(r1)     AS rel1,
  type(r2)     AS rel2,
  answer_count,
  [elementId(z)] AS mid_ids_sample,
  [{{id: elementId(z), name: z.name, types: coalesce(z.type, labels(z))}}] AS mid_nodes_sample,
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
// Middle-out optimized 2-anchor intersection
// 1) Sample center nodes x with degree constraints to avoid hubs
CALL {{
  // IMPORTANT: avoid full label scans with rand(); sample by internal id.
  WITH toInteger($max_node_id) AS maxId
  UNWIND range(1, {anchor_pool} * 10) AS i
  WITH maxId, toInteger(rand() * (maxId + 1)) AS rid
  MATCH (x{lbl})
  WHERE id(x) = rid
  WITH DISTINCT x
  LIMIT {anchor_pool}
  WITH x, COUNT {{ (x)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
  RETURN x
}}

// 2) For each x, pick two random neighbors a,b (x is guaranteed to be a common answer)
CALL {{
  WITH x
  MATCH (x)-[ra]-(a{lbl})
  WITH x, a, ra
  ORDER BY rand()
  LIMIT 2
  RETURN collect({{a:a, ra:ra}}) AS picks
}}
WITH x, picks
WHERE size(picks) = 2 AND elementId(picks[0].a) <> elementId(picks[1].a)

WITH picks[0].a AS a, picks[1].a AS b,
     picks[0].ra AS ra, picks[1].ra AS rb

// 3) Validate answer set size for (a,b). Use capped collect to early-stop when too large.
MATCH (a)-[rA]-(y{lbl}), (b)-[rB]-(y{lbl})
WITH a, b, ra, rb,
     collect(DISTINCT y)[0..{max_answers + 1}] AS answers
WITH a, b, ra, rb, answers, size(answers) AS answer_count
WHERE answer_count >= {min_answers} AND answer_count <= {max_answers}

RETURN
  elementId(a) AS anchorA_id,
  a.name       AS anchorA_name,
  coalesce(a.type, labels(a)) AS anchorA_types,
  elementId(b) AS anchorB_id,
  b.name       AS anchorB_name,
  coalesce(b.type, labels(b)) AS anchorB_types,
  type(ra)     AS anchorA_edge_type,
  type(rb)     AS anchorB_edge_type,
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
// Middle-out optimized 3-anchor intersection
CALL {{
  // IMPORTANT: avoid full label scans with rand(); sample by internal id.
  WITH toInteger($max_node_id) AS maxId
  UNWIND range(1, {anchor_pool} * 10) AS i
  WITH maxId, toInteger(rand() * (maxId + 1)) AS rid
  MATCH (x{lbl})
  WHERE id(x) = rid
  WITH DISTINCT x
  LIMIT {anchor_pool}
  WITH x, COUNT {{ (x)--() }} AS deg
  WHERE {deg_min} <= deg AND deg <= {deg_max}
  RETURN x
}}

// 2) Pick three distinct neighbors a,b,c of x without scanning huge neighbor sets.
//    We use bounded retry loops and degree constraints to avoid hubs.
CALL {{
  WITH x
  UNWIND range(1, 30) AS i
  MATCH (x)-[ra]-(a{lbl})
  WHERE a <> x
  WITH a, ra
  ORDER BY rand()
  LIMIT 1
  WITH a, ra, COUNT {{ (a)--() }} AS adeg
  WHERE {deg_min} <= adeg AND adeg <= {deg_max}
  RETURN a, ra
  LIMIT 1
}}
CALL {{
  WITH x, a
  UNWIND range(1, 30) AS j
  MATCH (x)-[rb]-(b{lbl})
  WHERE b <> x AND b <> a
  WITH b, rb
  ORDER BY rand()
  LIMIT 1
  WITH b, rb, COUNT {{ (b)--() }} AS bdeg
  WHERE {deg_min} <= bdeg AND bdeg <= {deg_max}
  RETURN b, rb
  LIMIT 1
}}
CALL {{
  WITH x, a, b
  UNWIND range(1, 30) AS k
  MATCH (x)-[rc]-(c{lbl})
  WHERE c <> x AND c <> a AND c <> b
  WITH c, rc
  ORDER BY rand()
  LIMIT 1
  WITH c, rc, COUNT {{ (c)--() }} AS cdeg
  WHERE {deg_min} <= cdeg AND cdeg <= {deg_max}
  RETURN c, rc
  LIMIT 1
}}

// 3) Validate answer set size for (a,b,c).
//    Intersect stepwise and early-stop when too large.
MATCH (a)-[r1]-(y{lbl})
WITH a, b, c, ra, rb, rc, y
MATCH (b)-[r2]-(y{lbl})
WITH a, b, c, ra, rb, rc, y
MATCH (c)-[r3]-(y{lbl})
WITH DISTINCT a, b, c, ra, rb, rc, y
LIMIT {max_answers + 1}
WITH a, b, c, ra, rb, rc, collect(y) AS answers
WITH a, b, c, ra, rb, rc, answers, size(answers) AS answer_count
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
  type(ra)     AS anchorA_edge_type,
  type(rb)     AS anchorB_edge_type,
  type(rc)     AS anchorC_edge_type,
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
  // Answer-first sampling
  MATCH (x)
  WHERE rand() < $keep_prob
  RETURN x
  LIMIT $anchor_pool
}
WITH collect(x) AS answers
WHERE size(answers) > 0
UNWIND range(1, $limit) AS i
WITH answers, apoc.coll.randomItem(answers) AS x
CALL {
  WITH x
  MATCH (a)-[r]-(x)
  RETURN a, r
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
  // Answer-first sampling
  MATCH (x)
  WHERE rand() < $keep_prob
  RETURN x
  LIMIT $anchor_pool
}
WITH collect(x) AS answers
WHERE size(answers) > 0
UNWIND range(1, $limit) AS i
WITH answers, apoc.coll.randomItem(answers) AS x
CALL {
  WITH x
  MATCH (x)-[r2]-(z)
  RETURN r2, z
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH x, z
  MATCH (z)-[r1]-(a)
  WHERE a <> x
  RETURN r1, a
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
  [elementId(z)] AS mid_ids_sample,
  [{id: elementId(z), name: z.name, types: coalesce(z.type, labels(z))}] AS mid_nodes_sample,
  [elementId(x)] AS answer_ids_sample,
  [{id: elementId(x), name: x.name, types: coalesce(x.type, labels(x))}] AS answer_nodes_sample;
"""


simple_two_anchor_intersection_cypher = """
CALL {
  // Middle-out: sample center y first
  MATCH (y)
  WHERE rand() < $keep_prob
  RETURN y
  LIMIT $anchor_pool
}
WITH collect(y) AS centers
WHERE size(centers) > 0
UNWIND range(1, $limit) AS i
WITH centers, apoc.coll.randomItem(centers) AS y
CALL {
  WITH y
  MATCH (y)-[ra]-(a)
  RETURN ra, a
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH y, a
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
  // Middle-out: sample center y first
  MATCH (y)
  WHERE rand() < $keep_prob
  RETURN y
  LIMIT $anchor_pool
}
WITH collect(y) AS centers
WHERE size(centers) > 0
UNWIND range(1, $limit) AS i
WITH centers, apoc.coll.randomItem(centers) AS y
CALL {
  WITH y
  MATCH (y)-[r1]-(a)
  RETURN r1, a
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH y, a
  MATCH (y)-[r2]-(b)
  WHERE b <> a
  RETURN r2, b
  ORDER BY rand()
  LIMIT 1
}
CALL {
  WITH y, a, b
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
