"""
MetaQA Schema Definition

MetaQA映画ドメインのスキーマグラフ定義
"""

# MetaQAのスキーマグラフ
# (src_type, relation, tgt_type, direction)
# direction: "->" for forward, "<-" for reverse, "<->" for bidirectional
# Note: MetaQAは双方向クエリが多いため、逆方向も追加
SCHEMA_GRAPH = [
    # Forward: Movie -> X
    ("Movie", "DIRECTED_BY", "Organization", "->"),
    ("Movie", "DIRECTED_BY", "Person", "->"),
    ("Movie", "HAS_GENRE", "Text", "->"),
    ("Movie", "HAS_IMDB_RATING", "Number", "->"),
    ("Movie", "HAS_IMDB_VOTES", "Number", "->"),
    ("Movie", "HAS_TAGS", "Text", "->"),
    ("Movie", "IN_LANGUAGE", "Language", "->"),
    ("Movie", "IN_LANGUAGE", "Text", "->"),
    ("Movie", "RELEASE_YEAR", "Date", "->"),
    ("Movie", "STARRED_ACTORS", "Person", "->"),
    ("Movie", "STARRED_ACTORS", "Organization", "->"),
    ("Movie", "WRITTEN_BY", "Organization", "->"),
    ("Movie", "WRITTEN_BY", "Person", "->"),
    # Reverse: X -> Movie (for queries like "What movies did X star in?")
    ("Organization", "DIRECTED_BY", "Movie", "<-"),
    ("Person", "DIRECTED_BY", "Movie", "<-"),
    ("Text", "HAS_GENRE", "Movie", "<-"),
    ("Text", "HAS_TAGS", "Movie", "<-"),
    ("Language", "IN_LANGUAGE", "Movie", "<-"),
    ("Text", "IN_LANGUAGE", "Movie", "<-"),
    ("Date", "RELEASE_YEAR", "Movie", "<-"),
    ("Person", "STARRED_ACTORS", "Movie", "<-"),
    ("Organization", "STARRED_ACTORS", "Movie", "<-"),
    ("Organization", "WRITTEN_BY", "Movie", "<-"),
    ("Person", "WRITTEN_BY", "Movie", "<-"),
]

# MetaQAの全エンティティタイプ
ENTITY_TYPES = [
    "Movie",
    "Person",
    "Organization",
    "Text",
    "Date",
    "Language",
    "Number",
]
