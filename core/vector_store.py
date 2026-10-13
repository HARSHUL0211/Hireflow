import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from utils.config import EMBEDDING_DIMENSION, FAISS_DIR
from utils.embeddings import get_embeddings
from utils.utils import get_logger, is_quota_error

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, persist_dir: Path = FAISS_DIR, dimension: int = EMBEDDING_DIMENSION):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.embeddings = None
        self.indexes: Dict[str, faiss.Index] = {}
        self.records: Dict[str, List[Dict[str, Any]]] = {}  # namespace -> [{id, text, metadata}]
        self.ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self) -> bool:
        self.embeddings = get_embeddings()
        if self.embeddings is None:
            logger.warning("Vector store disabled (no embeddings).")
            self.ready = False
            return False
        for ns in ("resumes", "job_descriptions"):
            self._load_or_create(ns)
        self.ready = True
        return True

    def _paths(self, ns: str):
        return self.persist_dir / f"{ns}.faiss", self.persist_dir / f"{ns}.json"

    def _load_or_create(self, ns: str) -> None:
        idx_path, meta_path = self._paths(ns)
        if idx_path.exists() and meta_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                if index.d != self.dimension:
                    raise ValueError(f"dimension mismatch ({index.d} != {self.dimension})")
                self.indexes[ns] = index
                self.records[ns] = json.loads(meta_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded FAISS namespace '{ns}' with {index.ntotal} vectors")
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not load '{ns}' ({e}); recreating empty index")
        self.indexes[ns] = faiss.IndexFlatIP(self.dimension)
        self.records[ns] = []

    def save(self, ns: Optional[str] = None) -> None:
        for name in ([ns] if ns else list(self.indexes)):
            idx_path, meta_path = self._paths(name)
            faiss.write_index(self.indexes[name], str(idx_path))
            meta_path.write_text(json.dumps(self.records[name], ensure_ascii=False), encoding="utf-8")

    def reset(self, ns: Optional[str] = None) -> None:
        for name in ([ns] if ns else list(self.indexes)):
            self.indexes[name] = faiss.IndexFlatIP(self.dimension)
            self.records[name] = []
            self.save(name)

    # --------------------------------------------------------------- add docs
    def add_documents(self, documents: List[Document], namespace: str = "resumes") -> bool:
        """Embed and add documents. Chunks already present (same id) are skipped."""
        if not self.ready:
            return False
        id_key = "candidate_id" if namespace == "resumes" else "jd_id"
        existing = {r["id"] for r in self.records[namespace]}
        new_docs, new_ids = [], []
        for d in documents:
            rec_id = f"{d.metadata.get(id_key, 'doc')}::{d.metadata.get('chunk_index', 0)}"
            if rec_id not in existing:
                new_docs.append(d)
                new_ids.append(rec_id)
        if not new_docs:
            logger.info(f"[{namespace}] nothing new to add")
            return True
        try:
            vectors = self.embeddings.embed_documents([d.page_content for d in new_docs])
        except Exception as e:  # noqa: BLE001
            if is_quota_error(e):
                logger.warning("Embedding quota hit - re-run later; cached chunks won't be re-billed.")
            else:
                logger.error(f"Embedding failed: {e}")
            return False
        arr = np.asarray(vectors, dtype="float32")
        self.indexes[namespace].add(arr)
        for rec_id, d in zip(new_ids, new_docs):
            self.records[namespace].append({"id": rec_id, "text": d.page_content, "metadata": d.metadata})
        self.save(namespace)
        logger.info(f"[{namespace}] added {len(new_docs)} chunks (total {self.indexes[namespace].ntotal})")
        return True

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5, namespace: str = "resumes",
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Return chunk-level hits: [{id, text, metadata, score}], best first."""
        if not self.ready or self.indexes[namespace].ntotal == 0:
            return []
        try:
            qvec = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Query embedding failed: {e}")
            return []
        pred = self._build_filter(filters)
        k = self.indexes[namespace].ntotal if pred else min(top_k, self.indexes[namespace].ntotal)
        scores, ids = self.indexes[namespace].search(qvec, k)
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            rec = self.records[namespace][i]
            if pred and not pred(rec["metadata"]):
                continue
            hits.append({"id": rec["id"], "text": rec["text"], "metadata": rec["metadata"], "score": float(score)})
            if len(hits) >= top_k:
                break
        return hits

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Callable[[dict], bool]]:
        """dict -> predicate. Supported: exact match, list = any-of, {'$gte': n}, {'$lte': n}, {'$contains': s}."""
        if not filters:
            return None

        def check(meta: dict) -> bool:
            for key, cond in filters.items():
                val = meta.get(key)
                if isinstance(cond, dict):
                    if "$gte" in cond and (val is None or val < cond["$gte"]):
                        return False
                    if "$lte" in cond and (val is None or val > cond["$lte"]):
                        return False
                    if "$contains" in cond and cond["$contains"].lower() not in str(val or "").lower():
                        return False
                elif isinstance(cond, list):
                    if val not in cond:
                        return False
                elif val != cond:
                    return False
            return True

        return check

    # ------------------------------------------------------------------ stats
    def get_stats(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "backend": "FAISS (IndexFlatIP, cosine)",
            "dimension": self.dimension,
            "namespaces": {ns: self.indexes[ns].ntotal for ns in self.indexes},
            "total_vector_count": sum(i.ntotal for i in self.indexes.values()),
        }