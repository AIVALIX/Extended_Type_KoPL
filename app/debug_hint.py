import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# disease -> drug で利用可能なリレーション
relations = [
    "off-label use",
    "side effect", 
    "contraindication",
    "indication",
    "target",
    "transporter",
    "carrier"
]

# LLMが生成したhint
hint = "treated by"

# ベクトル類似度を計算
hint_vec = np.array(embeddings.embed_query(hint))
rel_vecs = embeddings.embed_documents(relations)

similarities = []
for i, rel in enumerate(relations):
    sim = np.dot(hint_vec, rel_vecs[i]) / (np.linalg.norm(hint_vec) * np.linalg.norm(rel_vecs[i]))
    similarities.append((rel, sim))

similarities.sort(key=lambda x: -x[1])

print(f'Hint: "{hint}"')
print('Relations ranked by similarity:')
for rel, sim in similarities:
    print(f'  {rel}: {sim:.4f}')
