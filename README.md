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

