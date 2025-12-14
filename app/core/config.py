# app/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings

BASEMODEL = "gpt-4.1-mini"  # デフォルトの LLM モデル名


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_URI: str = "bolt://neo4j_metaqa-v2:7688"
    LANGSMITH_API_KEY: str

    class Config:
        env_file = "/app/config/.env"  # ルートに置くだけで十分
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # 1 回だけパース
