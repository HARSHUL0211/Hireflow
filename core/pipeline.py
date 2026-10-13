import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from core.evaluator import RAGEvaluator
from core.hybrid_indexer import HybridIndexer
from core.ingestion import DocumentProcessor
from core.memory_rag import MemoryRAG
from core.parsing import JobParser, ResumeParser
from core.re_ranker import ReRanker
from core.search_router import SearchRouter
from core.vector_store import VectorStore
from utils.config import CACHE_DIR, HYBRID_INDEX_DIR, RESUMES_DIR, USE_LLM_PARSING
from utils.schemas import CandidateEvaluation, JobDescription
from utils.utils import get_logger

logger = get_logger(__name__)
PARSED_CACHE = CACHE_DIR / "parsed_resumes.json"


class RecruitmentPipeline:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.resume_parser = ResumeParser(use_llm=USE_LLM_PARSING)
        self.job_parser = JobParser(use_llm=True)
        self.vector_store = VectorStore()
        self.hybrid = HybridIndexer(self.vector_store)
        self.router = SearchRouter(self.hybrid)
        self.reranker = ReRanker()
        self.memory = MemoryRAG()
        self.evaluator = RAGEvaluator(self.router)

    # ----------------------------------------------------------------- ingest
    def _load_parsed_cache(self) -> Dict[str, Any]:
        if PARSED_CACHE.exists():
            try:
                return json.loads(PARSED_CACHE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _enrich(self, docs: List[Document]) -> List[Document]:
        """Parse each resume (results are cached) and copy the fields into the document metadata."""
        cache = self._load_parsed_cache()
        for doc in docs:
            cid = doc.metadata["candidate_id"]
            if cid not in cache:
                r = self.resume_parser.parse_resume(
                    doc.page_content, cid, doc.metadata.get("name", "")
                )
                cache[cid] = {
                    "name": r.name,
                    "email": r.email,
                    "phone": r.phone,
                    "location": r.location,
                    "title": r.title,
                    "skills": r.skills,
                    "experience": r.experience,
                }
            doc.metadata.update({k: v for k, v in cache[cid].items() if v is not None})
        PARSED_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return docs

    def ingest_resumes(self, directory: Path = RESUMES_DIR) -> int:
        """PDF -> text -> parse -> BM25 + FAISS index. Safe to re-run: indexed chunks are skipped."""
        docs = self._enrich(self.processor.load_resumes(directory))
        self.hybrid.index_resumes(docs)
        return len(docs)

    def add_uploaded_resume(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Save an uploaded PDF into data/resumes, then parse and index it. Returns its metadata."""
        dest = RESUMES_DIR / Path(filename).name
        dest.write_bytes(file_bytes)
        doc = self.processor.process_pdf(dest)
        if doc is None:
            return None
        doc = self._enrich([doc])[0]
        self.hybrid.index_resumes([doc])
        return doc.metadata

    def is_indexed(self) -> bool:
        return bool(self.hybrid.docs["resumes"])

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "auto",
        required_skills: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        min_experience: Optional[float] = None,
        jd_context: str = "",
    ) -> List[Dict[str, Any]]:
        raw_filters = {
            "required_skills": required_skills,
            "target_locations": locations,
            "min_experience": min_experience,
        }
        filters = {k: v for k, v in raw_filters.items() if v}
        results = self.router.search(
            query,
            top_k=top_k,
            filters=filters or None,
            search_mode=search_mode,
            jd_context=jd_context,
        )
        self.memory.record_search(query, len(results))
        return results

    def evaluate_candidates(
        self, results: List[Dict[str, Any]], jd: JobDescription
    ) -> List[CandidateEvaluation]:
        return self.reranker.re_rank_candidates(results, jd)

    def parse_job(self, text: str, title: str = "") -> JobDescription:
        return self.job_parser.parse_job_description(text, jd_id="job", title=title)

    # ----------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        stats = self.hybrid.get_index_stats()
        stats["llm_available"] = self.reranker.is_available()
        stats["memory"] = self.memory.get_memory_stats()
        return stats

    def reset_indexes(self) -> None:
        """Wipe the FAISS index, BM25 documents and parsed-resume cache. The PDFs are kept."""
        self.vector_store.reset()
        for path in list(HYBRID_INDEX_DIR.glob("*.json")) + [PARSED_CACHE]:
            path.unlink(missing_ok=True)
        self.hybrid.docs = {"resumes": [], "job_descriptions": []}
        self.hybrid.bm25 = {"resumes": None, "job_descriptions": None}