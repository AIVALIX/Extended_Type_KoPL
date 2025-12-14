# app/services/embedding.py
from openai import OpenAI, RateLimitError, APIError
from tenacity import retry, wait_exponential, stop_after_attempt
from core.config import get_settings


class OpenAIEmbedder:
    def __init__(self):
        settings = get_settings()
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.EMBEDDING_MODEL

    @retry(
        wait=wait_exponential(multiplier=0.5, max=10),
        stop=stop_after_attempt(4),
        reraise=True,
        retry=(lambda e: isinstance(e, (RateLimitError, APIError))),
    )
    def embed_one(self, text: str) -> list[float]:
        res = self._client.embeddings.create(
            input=[text.replace("\n", " ")], model=self._model
        )
        return res.data[0].embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        res = self._client.embeddings.create(
            input=[t.replace("\n", " ") for t in texts],
            model=self._model,
        )
        return [d.embedding for d in res.data]


if __name__ == "__main__":
    embedder = OpenAIEmbedder()
    print(embedder.embed_one("Hello, world!"))
    print(embedder.embed_many(["Hello, world!", "How are you?"]))
