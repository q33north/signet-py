"""Local embedding generation via sentence-transformers."""
from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger()


class EmbeddingService:
    """Generates text embeddings using a local sentence-transformers model.

    Model is loaded lazily on first use and kept in memory.
    All public methods are async; GPU inference is offloaded via asyncio.to_thread.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            log.info("embeddings.loading_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            else:
                log.warning("embeddings.no_gpu", fallback="cpu")
            log.info("embeddings.model_loaded", device=str(self._model.device))
        return self._model

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns 384d vector."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Offloaded to thread pool."""
        return await asyncio.to_thread(self._encode_sync, texts)

    @property
    def dimension(self) -> int:
        dims = {"all-MiniLM-L6-v2": 384, "all-mpnet-base-v2": 768}
        return dims.get(self._model_name, 384)
