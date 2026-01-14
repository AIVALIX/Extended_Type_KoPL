# 論文用の検証レポジトリ 共有用

## データ投入
1. config/.envを書き込み以下の項目を入力
   ```
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=password
    NEO4J_URI=bolt://neo4j_metaqa-v2:7687
    OPENAI_API_KEY=sk-xxx
    LANGSMITH_API_KEY=xxx(option)

   ```
2. コンテナを立ち上げる
3. python-environment-app-v2のコンテナ内で以下のコマンドを実行
   ```
   python database/seed.py
   ```


## 統一検証
1. app/overall_test.pyにテスト用コードが載っているので参照
2. パイプラインについてのコードはapp/pipeline/graph_pipeline.pyに記載



# 既存のDBを上書きする場合 (DockerやLinux環境の例)
neo4j-admin database import full \
  --nodes=/var/lib/neo4j/data/kg/nodes.csv \
  --relationships=/var/lib/neo4j/data/kg/relationships.csv \
  --delimiter=, \
  --array-delimiter=";" \
  --overwrite-destination=true \
  neo4j

  docker compose run --rm --entrypoint="" neo4j ls -la /import



python - <<'PY'
import json
from pathlib import Path
p=Path('/tmp/with_kopl_sample.jsonl')
with p.open() as f:
    row=json.loads(next(f))
print('query_type',row.get('query_type'))
print('kopl_strict',row.get('kopl_strict')[:300])
print('kopl_type',row.get('kopl_type')[:300])
print('kopl_strict_struct_len',len(row.get('kopl_strict_struct',[])))
print('kopl_type_struct_len',len(row.get('kopl_type_struct',[])))
PY

