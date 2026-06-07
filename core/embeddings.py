"""Embedding Service - Text to vector conversion."""

import os
from lib.logger import log

try:
    from langchain_community.embeddings import OllamaEmbeddings
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class EmbeddingService:
    """Embedding service supporting Ollama and OpenAI."""

    def __init__(self, provider=None, model=None, base_url=None, api_key=None):
        self.provider = provider or os.environ.get("EMBED_PROVIDER", "ollama")
        self.model = model or os.environ.get("EMBED_MODEL", "bge-m3")
        self.base_url = base_url or os.environ.get("EMBED_BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.environ.get("EMBED_API_KEY", "")
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "ollama" and HAS_OLLAMA:
            self._client = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url,
            )
            log.info(f"Embedding: Ollama/{self.model}")
        elif self.provider == "openai" and HAS_OPENAI:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            log.info(f"Embedding: OpenAI/{self.model}")
        else:
            log.warning(f"Embedding provider '{self.provider}' not available, using dummy")
            self._client = None

    def embed_documents(self, texts):
        """Embed a list of documents."""
        if not self._client:
            return [[0.0] * 384 for _ in texts]
        if self.provider == "ollama":
            return self._client.embed_documents(texts)
        elif self.provider == "openai":
            resp = self._client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text):
        """Embed a single query."""
        if not self._client:
            return [0.0] * 384
        if self.provider == "ollama":
            return self._client.embed_query(text)
        elif self.provider == "openai":
            resp = self._client.embeddings.create(model=self.model, input=[text])
            return resp.data[0].embedding
        return [0.0] * 384

    def get_langchain_embeddings(self):
        """Return LangChain-compatible embeddings object."""
        if self.provider == "ollama" and HAS_OLLAMA:
            return self._client
        return self
