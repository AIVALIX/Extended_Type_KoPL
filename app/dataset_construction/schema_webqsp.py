"""
WebQSP Schema Definition

WebQSP (Web Questions Semantic Parses) dataset uses a subset of Freebase as
its knowledge graph. Freebase contains Compound Value Type (CVT) nodes that
act as intermediate join nodes (e.g., film.performance links a film, actor,
and character). These CVT nodes should be collapsed into direct edges for
compatibility with the pipeline's schema graph.

CVT Handling Strategy:
  Freebase uses CVT (Compound Value Type) nodes as n-ary relation mediators.
  For example, instead of (Film)-[actor]->(Person), Freebase has:
    (Film)-[film.film.starring]->(film.performance)-[film.performance.actor]->(Person)
  We collapse these into direct edges:
    (Film)-[film.film.starring..film.performance.actor]->(Person)
  This simplifies the schema and aligns with the hop-based traversal model
  used by all pipelines.

References:
  - WebQSP dataset: https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling/
  - Freebase setup: https://github.com/dki-lab/Freebase-Setup
  - Freebase CVT documentation: https://developers.google.com/freebase/guide/basic_concepts
"""

# WebQSP Schema Graph (Freebase subset)
# (src_type, relation, tgt_type, direction)
# direction: "->" for forward, "<-" for reverse, "<->" for bidirectional
#
# TODO: Populate this list after importing the Freebase subgraph into Neo4j.
#       Run `GraphPathFinder("webqsp").get_all_relations()` to discover the
#       actual schema edges present in the imported graph.
#
# Common Freebase relation types expected in WebQSP:
#   - people.person.nationality          (Person -> Country)
#   - people.person.place_of_birth       (Person -> Location)
#   - people.person.profession           (Person -> Profession)
#   - people.person.spouse_s             (Person -> Person, via CVT)
#   - film.film.directed_by              (Film -> Person)
#   - film.film.starring                 (Film -> Person, via CVT film.performance)
#   - film.film.genre                    (Film -> Genre)
#   - location.location.containedby      (Location -> Location)
#   - location.country.capital           (Country -> City)
#   - organization.organization.founded_by (Organization -> Person)
#   - music.artist.genre                 (Artist -> Genre)
#   - sports.sports_team.roster          (Team -> Person, via CVT)
#   - government.governmental_body.members (GovBody -> Person, via CVT)
#   - education.educational_institution.students_graduates (Institution -> Person, via CVT)
#
# After CVT collapse, these will become direct edges in the schema.
SCHEMA_GRAPH = [
    # TODO: Add edges after Freebase subgraph import and CVT collapse.
    # Example format (uncomment and adjust after import):
    # ("Film", "film.film.directed_by", "Person", "->"),
    # ("Person", "film.film.directed_by", "Film", "<-"),
    # ("Film", "film.film.starring..film.performance.actor", "Person", "->"),
    # ("Person", "people.person.nationality", "Country", "->"),
    # ("Location", "location.location.containedby", "Location", "->"),
]

# WebQSP Entity Types (Freebase types)
# TODO: Populate after importing the subgraph. Use
#       `GraphPathFinder("webqsp").get_all_types()` to discover types.
#
# Expected common types in WebQSP:
#   Person, Film, Location, Country, City, Organization,
#   Genre, Language, Sport, SportsTeam, etc.
ENTITY_TYPES = [
    # TODO: Add entity types after Freebase subgraph import.
]
