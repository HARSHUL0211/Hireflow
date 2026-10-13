from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

from core.filters import apply_filters
from core.hybrid_indexer import HybridIndexer
from utils.config import GOOGLE_API_KEY, LLM_MODEL, RRF_K
from utils.multi_query import MultiQueryGenerator
from utils.utils import get_logger

logger = get_logger(__name__)


class RankedIds(BaseModel):
    ordered_candidate_ids: List[str] = Field(description="candidate ids, best match first")


class SearchRouter:
    def __init__(self, hybrid_indexer: HybridIndexer):
        self.hybrid = hybrid_indexer
        self.vector_store = hybrid_indexer.vector_store
        self.multi_query = MultiQueryGenerator()
        self.llm = None
        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0
                )
            except Exception as e:
                logger.warning(f"Router LLM unavailable: {e}")
        self.chain = self._build_search_chain()

    # ------------------------------------------------------------- strategies
    def shallow_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Vector-only. Falls back to deep search if the vector store is not usable."""
        if not self.vector_store.ready:
            logger.info("Vector store not ready - shallow search falling back to deep search")
            return self.deep_search(inputs)

        pool = inputs["top_k"] * (4 if inputs.get("filters") else 1)
        best: Dict[str, Dict[str, Any]] = {}
        hits = self.vector_store.search(inputs["query"], top_k=pool * 6, namespace="resumes")
        for hit in hits:
            cid = hit["metadata"]["candidate_id"]
            if cid not in best:  # hits are sorted, so the first one is the best chunk
                best[cid] = {
                    "candidate_id": cid,
                    "name": hit["metadata"].get("name", ""),
                    "combined_score": round(hit["score"], 4),
                    "vector_score": round(hit["score"], 4),
                    "bm25_score": 0.0,
                    "snippet": hit["text"],
                    "metadata": hit["metadata"],
                    "text": self._full_text(cid),
                }
        results = list(best.values())[:pool]
        return {"results": results, "mode": "shallow", "queries": [inputs["query"]]}

    def deep_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query, top_k = inputs["query"], inputs["top_k"]
        pool = top_k * (4 if inputs.get("filters") else 2)

        # 1) several rephrasings of the query
        queries = self.multi_query.generate_queries(query, num_queries=3)

        # 2) hybrid search for each one, fused across queries with RRF
        fused: Dict[str, float] = defaultdict(float)
        seen: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            for rank, r in enumerate(self.hybrid.search_resumes(q, top_k=pool), start=1):
                cid = r["candidate_id"]
                fused[cid] += 1.0 / (RRF_K + rank)
                if cid not in seen or r["combined_score"] > seen[cid]["combined_score"]:
                    seen[cid] = r

        best_possible = len(queries) / (RRF_K + 1)
        ordered = sorted(fused, key=fused.get, reverse=True)
        results = []
        for cid in ordered:
            item = dict(seen[cid])
            item["combined_score"] = round(100 * fused[cid] / best_possible, 1)
            results.append(item)

        # 3) metadata filters, then LLM re-rank
        if inputs.get("filters"):
            results = apply_filters(results, **inputs["filters"])
        results = results[:pool]
        results = self._llm_rerank(results, query, inputs.get("jd_context", ""))
        return {"results": results, "mode": "deep", "queries": queries}

    # ---------------------------------------------------------------- routing
    def route_search(self, inputs: Dict[str, Any]) -> str:
        mode = inputs.get("search_mode", "auto")
        if mode in ("shallow", "deep"):
            return mode
        n_words = len(inputs["query"].split())
        if n_words <= 3 and not inputs.get("jd_context"):
            return "shallow"
        return "deep"

    def _build_search_chain(self):
        return RunnableBranch(
            (lambda x: self.route_search(x) == "shallow", RunnableLambda(self.shallow_search)),
            (lambda x: self.route_search(x) == "deep", RunnableLambda(self.deep_search)),
            RunnableLambda(self.deep_search),
        )

    # ------------------------------------------------------------- LLM rerank
    def _llm_rerank(self, results: List[Dict[str, Any]], query: str, jd_context: str) -> List[Dict[str, Any]]:
        if self.llm is None or len(results) < 2:
            return results
        try:
            lines = []
            for r in results:
                m = r["metadata"]
                skills = ", ".join(m.get("skills", [])[:12])
                lines.append(
                    f"- id={r['candidate_id']} | {m.get('title', '')} | "
                    f"{m.get('experience', '?')} yrs | skills: {skills}"
                )
            prompt = (
                "Rank these candidates for the hiring need below, best first. "
                "Return every id exactly once.\n\n"
                f"Need: {query}\n{jd_context[:1500]}\n\nCandidates:\n" + "\n".join(lines)
            )
            ranked = self.llm.with_structured_output(RankedIds).invoke(prompt).ordered_candidate_ids
            by_id = {r["candidate_id"]: r for r in results}
            ordered = [by_id[i] for i in ranked if i in by_id]
            used = {o["candidate_id"] for o in ordered}
            ordered += [r for r in results if r["candidate_id"] not in used]
            for pos, r in enumerate(ordered, start=1):
                r["llm_reranked"] = True
                r["llm_rank"] = pos
            return ordered
        except Exception as e:
            logger.warning(f"LLM rerank failed ({e}); keeping hybrid order")
            return results

    def _full_text(self, candidate_id: str) -> str:
        for d in self.hybrid.docs["resumes"]:
            if d["metadata"].get("candidate_id") == candidate_id:
                return d["text"]
        return ""

    # ------------------------------------------------------------- public API
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        search_mode: str = "auto",
        jd_context: str = "",
    ) -> List[Dict[str, Any]]:
        """filters = {"required_skills": [...], "target_locations": [...], "min_experience": 3}"""
        inputs = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "search_mode": search_mode,
            "jd_context": jd_context,
        }
        out = self.chain.invoke(inputs)
        results = out["results"]
        if filters and out["mode"] == "shallow":
            results = apply_filters(results, **filters)
        results = results[:top_k]
        for r in results:
            r["search_mode"] = out["mode"]
            r["queries_used"] = out["queries"]
        return results

    def get_search_stats(self) -> Dict[str, Any]:
        return {
            "llm_available": self.llm is not None,
            "vector_store_ready": self.vector_store.ready,
            "hybrid_stats": self.hybrid.get_index_stats(),
        }