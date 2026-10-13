import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.ingestion import DocumentProcessor
from core.vector_store import VectorStore
from utils.config import BM25_WEIGHT, HYBRID_INDEX_DIR, RRF_K, VECTOR_WEIGHT
from utils.utils import get_logger

logger = get_logger(__name__)

STOPWORDS = {"a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "at", "by", "is", "are",
             "was", "were", "be", "as", "from", "that", "this", "it", "its", "we", "our", "you", "your"}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: (1.0 if hi > 0 else 0.0) for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class HybridIndexer:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.vector_store.initialize()
        self.processor = DocumentProcessor()
        self.bm25: Dict[str, Optional[BM25Okapi]] = {"resumes": None, "job_descriptions": None}
        self.docs: Dict[str, List[Dict[str, Any]]] = {"resumes": [], "job_descriptions": []}  # full docs
        self._load_bm25_docs()

    # ------------------------------------------------------------ persistence
    def _path(self, ns: str):
        return HYBRID_INDEX_DIR / f"{ns}_docs.json"

    def _load_bm25_docs(self) -> None:
        for ns in self.docs:
            p = self._path(ns)
            if p.exists():
                try:
                    self.docs[ns] = json.loads(p.read_text(encoding="utf-8"))
                    self._build_bm25(ns)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Could not load BM25 docs for {ns}: {e}")

    def _save_bm25_docs(self, ns: str) -> None:
        self._path(ns).write_text(json.dumps(self.docs[ns], ensure_ascii=False), encoding="utf-8")

    def _build_bm25(self, ns: str) -> None:
        corpus = [tokenize(d["text"]) for d in self.docs[ns]]
        self.bm25[ns] = BM25Okapi(corpus) if corpus else None

    # --------------------------------------------------------------- indexing
    def _index(self, documents: List[Document], ns: str, id_key: str) -> bool:
        if not documents:
            return False
        by_id = {d[id_key]: d for d in (x["metadata"] | {"text": x["text"]} for x in self.docs[ns])}
        for doc in documents:  # upsert full documents for BM25
            by_id[doc.metadata[id_key]] = doc.metadata | {"text": doc.page_content}
        self.docs[ns] = [{"text": v["text"], "metadata": {k: x for k, x in v.items() if k != "text"}}
                         for v in by_id.values()]
        self._build_bm25(ns)
        self._save_bm25_docs(ns)
        # vector side: chunks
        if self.vector_store.ready:
            self.vector_store.add_documents(self.processor.chunk(documents), namespace=ns)
        return True

    def index_resumes(self, resumes: List[Document]) -> bool:
        return self._index(resumes, "resumes", "candidate_id")

    def index_job_descriptions(self, jds: List[Document]) -> bool:
        return self._index(jds, "job_descriptions", "jd_id")

    # ----------------------------------------------------------------- search
    def search_resumes(self, query: str, top_k: int = 5, method: str = "weighted",
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._search(query, "resumes", "candidate_id", top_k, method, filters)

    def search_job_descriptions(self, query: str, top_k: int = 5, method: str = "weighted") -> List[Dict[str, Any]]:
        return self._search(query, "job_descriptions", "jd_id", top_k, method, None)

    def _search(self, query: str, ns: str, id_key: str, top_k: int, method: str,
                filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        docs = self.docs[ns]
        if not docs:
            return []
        by_id = {d["metadata"][id_key]: d for d in docs}

        # --- BM25 (document level)
        bm25_scores: Dict[str, float] = {}
        if self.bm25[ns] is not None:
            raw = self.bm25[ns].get_scores(tokenize(query))
            bm25_scores = {d["metadata"][id_key]: float(s) for d, s in zip(docs, raw)}

        # --- Vector (chunk level -> best chunk per document)
        vector_scores: Dict[str, float] = {}
        best_chunk: Dict[str, str] = {}
        if self.vector_store.ready:
            for hit in self.vector_store.search(query, top_k=max(top_k * 6, 30), namespace=ns, filters=filters):
                cid = hit["metadata"].get(id_key)
                if cid and hit["score"] > vector_scores.get(cid, -1):
                    vector_scores[cid] = hit["score"]
                    best_chunk[cid] = hit["text"]

        return self.combine_results(bm25_scores, vector_scores, by_id, best_chunk, top_k, id_key, method, filters)

    def combine_results(self, bm25_scores, vector_scores, by_id, best_chunk, top_k, id_key,
                        method: str = "weighted", filters=None) -> List[Dict[str, Any]]:
        ids = set(bm25_scores) | set(vector_scores)
        if filters:  # apply metadata filters to BM25-only hits too
            pred = VectorStore._build_filter(filters)
            ids = {i for i in ids if pred(by_id[i]["metadata"])}
        bm25_scores = {i: bm25_scores.get(i, 0.0) for i in ids}
        vector_scores = {i: vector_scores.get(i, 0.0) for i in ids}

        if method == "rrf":
            combined = {i: 0.0 for i in ids}
            for scores in (bm25_scores, vector_scores):
                if not any(scores.values()):
                    continue
                for rank, (i, _) in enumerate(sorted(scores.items(), key=lambda x: -x[1]), start=1):
                    combined[i] += 1.0 / (RRF_K + rank)
        else:
            nb, nv = _minmax(bm25_scores), _minmax(vector_scores)
            if not self.vector_store.ready or not any(vector_scores.values()):
                w_b, w_v = 1.0, 0.0  # BM25-only fallback
            else:
                w_b, w_v = BM25_WEIGHT, VECTOR_WEIGHT
            combined = {i: w_b * nb[i] + w_v * nv[i] for i in ids}

        results = []
        for i, score in sorted(combined.items(), key=lambda x: -x[1])[:top_k]:
            doc = by_id[i]
            results.append({
                id_key: i,
                "name": doc["metadata"].get("name") or doc["metadata"].get("title", ""),
                "combined_score": round(score, 4),
                "bm25_score": round(bm25_scores[i], 4),
                "vector_score": round(vector_scores[i], 4),
                "text": doc["text"],
                "snippet": best_chunk.get(i, doc["text"][:300]),
                "metadata": doc["metadata"],
            })
        return results

    # ------------------------------------------------------------------ stats
    def get_index_stats(self) -> Dict[str, Any]:
        return {
            "bm25_ready": self.bm25["resumes"] is not None,
            "bm25_resumes": len(self.docs["resumes"]),
            "bm25_job_descriptions": len(self.docs["job_descriptions"]),
            "vector_store_ready": self.vector_store.ready,
            "vector_stats": self.vector_store.get_stats(),
            "hybrid_ready": self.bm25["resumes"] is not None and self.vector_store.ready,
        }