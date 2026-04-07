"""OpenAI embedding provider wrapper.

Usage:
    encoder = OpenAIEncoder()
    emb = encoder.encode("some text")

Requires OPENAI_API_KEY env var and the `openai` package.
"""
from typing import List, Iterable, Union
import os

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None


class OpenAIEncoder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if openai and self.api_key:
            openai.api_key = self.api_key

    def encode(self, text: Union[str, Iterable[str]]) -> List[List[float]]:
        """Encode input text or iterable of texts to embeddings.

        Returns a single embedding for a str, or a list of embeddings for an iterable.
        """
        if openai is None:
            raise RuntimeError("openai package not installed. Add 'openai' to requirements.txt")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")

        if isinstance(text, str):
            payload = [text]
        else:
            payload = list(text)

        resp = openai.Embeddings.create(input=payload, model=self.model)
        embeddings = [item["embedding"] for item in resp["data"]]
        return embeddings if len(embeddings) > 1 else embeddings[0]
