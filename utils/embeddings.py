import asyncio
import hashlib
import pickle
import time
from typing import List, Optional

import numpy as np

from utils.config import CACHE_DIR, EMBEDDING_DIMENSION, EMBEDDING_MODEL, GOOGLE_API_KEY
from utils.utils import get_logger, is_quota_error

logger = get_logger(__name__)
_CACHE_FILE = CACHE_DIR / "embedding_cache.pkl"


def _ensure_event_loop() -> None:
    """Streamlit runs scripts in a worker thread that has no asyncio event loop.
    The Google client needs one when it is created, so make one if it is missing."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype="float32")
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


class SafeEmbeddings:
    """Wraps GoogleGenerativeAIEmbeddings with batching, retries, caching and normalisation."""

    def __init__(self, batch_size: int = 20, max_retries: int = 4):
        _ensure_event_loop()
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.dimension = EMBEDDING_DIMENSION
        self._cache = self._load_cache()

    # ---- cache -------------------------------------------------------------
    def _load_cache(self) -> dict:
        if _CACHE_FILE.exists():
            try:
                return pickle.loads(_CACHE_FILE.read_bytes())
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            _CACHE_FILE.write_bytes(pickle.dumps(self._cache))
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

    @staticmethod
    def _key(text: str, kind: str) -> str:
        raw = f"{EMBEDDING_MODEL}|{EMBEDDING_DIMENSION}|{kind}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ---- API calls with retry ---------------------------------------------
    def _call(self, fn, *args, **kwargs):
        delay = 5
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if is_quota_error(e) and attempt < self.max_retries - 1:
                    logger.warning(f"Rate limited, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing = []
        for i, t in enumerate(texts):
            hit = self._cache.get(self._key(t, "doc"))
            if hit is not None:
                results[i] = hit
            else:
                missing.append(i)

        for start in range(0, len(missing), self.batch_size):
            idxs = missing[start:start + self.batch_size]
            batch = [texts[i] for i in idxs]
            vectors = self._call(
                self._model.embed_documents,
                batch,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimension,
            )
            for i, v in zip(idxs, vectors):
                v = _normalize(v)
                results[i] = v
                self._cache[self._key(texts[i], "doc")] = v
            self._save_cache()
            time.sleep(0.5)  # be gentle with free-tier limits
        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        key = self._key(text, "query")
        if key in self._cache:
            return self._cache[key]
        v = self._call(
            self._model.embed_query,
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.dimension,
        )
        v = _normalize(v)
        self._cache[key] = v
        self._save_cache()
        return v


def get_embeddings() -> Optional[SafeEmbeddings]:
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY missing - vector search disabled (BM25 only).")
        return None
    try:
        return SafeEmbeddings()
    except Exception as e:
        if is_quota_error(e):
            logger.warning("Embedding quota exhausted - vector search disabled.")
        else:
            logger.error(f"Could not create embeddings: {type(e).__name__}: {e}")
        return None