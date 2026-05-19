import os
from typing import Literal

import requests


class Embedder:
    """Wrapper for Ollama and OpenAI-compatible embedding APIs."""

    def __init__(
        self,
        provider: Literal["ollama", "openai"] = "ollama",
        model: str = "nomic-embed-text",
        host: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.host = host or "http://localhost:11434"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "ollama":
            return self._embed_ollama(texts)
        if self.provider == "openai":
            return self._embed_openai(texts)
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
        }
        resp = requests.post(f"{self.host}/api/embed", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings and "embedding" in data:
            embeddings = [data["embedding"]]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Ollama embedding count mismatch: {len(embeddings)} vs {len(texts)}")
        return embeddings

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            import openai
        except ImportError:
            raise ImportError("openai is required for OpenAI embedding. Install with: uv pip install openai")
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.embeddings.create(input=texts, model=self.model)
        return [e.embedding for e in resp.data]
