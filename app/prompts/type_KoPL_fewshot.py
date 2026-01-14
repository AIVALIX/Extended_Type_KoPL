FEW_SHOT_MetaQA = """
【Schema Context】
Type: Movie, Person, Genre, Language
Relation: starred_actors, directed_by, written_by, in_language, has_genre

Examples:
Question: "what languages is [Dark Horse] in"
Code:
exp1 = Findanchor(anchor_type='Movie', entity_name='Dark Horse')
res = FindTypeRelate(exp1, anchor_type='Movie', target_type='Language', relation_name='in_language', delta=1)
final = Stop(res)

Question: "who are the directors of the films starred by [Bob Denver]"
Code:
exp1 = Findanchor(anchor_type='Person', entity_name='Bob Denver')
films = FindTypeRelate(exp1, anchor_type="Person", target_type="Movie", relation_name="starred_in", delta=1)
directors = FindTypeRelate(films, anchor_type="Movie", target_type="Person", relation_name="directed_by", delta=1)
final = Stop(directors)
"""

FEW_SHOT_PrimeKGQA = """
【Schema Context】
Type: Disease, Gene, Drug, Phenotype, AnatomicalStructure
Relation: has_phenotype, related_to, targets, treated_by, interacts_with

Examples:
Question: "Silver-Russell syndromeに表現型として現れ、benign mesotheliomaに関連する遺伝子は何ですか？"
Code:
exp1 = Findanchor(anchor_type='Disease', entity_name='Silver-Russell syndrome')
srs_genes = FindTypeRelate(exp1, anchor_type="Disease", target_type="Gene", relation_name="has_phenotype", delta=1)

exp2 = Findanchor(anchor_type='Disease', entity_name='benign mesothelioma')
bm_genes = FindTypeRelate(exp2, anchor_type="Disease", target_type="Gene", relation_name="related_to", delta=1)

final = Stop(And(srs_genes, bm_genes))

Question: "遺伝子ACE2に関連する疾患の治療に使用される薬は何ですか？"
Code:
exp1 = Findanchor(anchor_type='Gene', entity_name='ACE2')
diseases = FindTypeRelate(exp1, anchor_type="Gene", target_type="Disease", relation_name="related_to", delta=1)
drugs = FindTypeRelate(diseases, anchor_type="Disease", target_type="Drug", relation_name="treated_by", delta=1)
final = Stop(drugs)
"""
