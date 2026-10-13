import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.search_router import SearchRouter
from utils.config import DATA_DIR, EMBEDDING_MODEL, GOOGLE_API_KEY, LLM_MODEL
from utils.utils import get_logger

logger = get_logger(__name__)

WEIGHTS = {
    "answer_relevancy": 0.3,
    "context_precision": 0.3,
    "faithfulness": 0.2,
    "answer_correctness": 0.2,
}


@dataclass
class RAGEvaluationMetrics:
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_correctness: float = 0.0
    overall_score: float = 0.0
    method: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGEvaluator:
    def __init__(self, router: SearchRouter):
        self.router = router
        self.evaluation_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ public API
    def evaluate_search_quality(
        self,
        query: str,
        expected_skills: List[str],
        search_mode: str = "deep",
        top_k: int = 3,
        use_ragas: bool = True,
    ) -> RAGEvaluationMetrics:
        results = self.router.search(query, top_k=top_k, search_mode=search_mode)
        if not results:
            return RAGEvaluationMetrics(method="no_results")

        metrics: Optional[RAGEvaluationMetrics] = None
        if use_ragas and GOOGLE_API_KEY:
            data = self.prepare_ragas_data(query, results, expected_skills)
            metrics = self.evaluate_with_ragas(data)
        if metrics is None:
            metrics = self.heuristic_metrics(query, results, expected_skills)

        self.evaluation_history.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "expected_skills": ", ".join(expected_skills),
                "search_mode": search_mode,
                **metrics.to_dict(),
            }
        )
        return metrics

    # ------------------------------------------------------------------ RAGAS
    def prepare_ragas_data(
        self, query: str, results: List[Dict[str, Any]], expected_skills: List[str]
    ) -> List[Dict[str, Any]]:
        """One row: contexts = candidate profiles, response = short summary, reference = ideal answer."""
        contexts, summaries = [], []
        for r in results:
            m = r["metadata"]
            contexts.append(
                f"{r['name']} - {m.get('title', '')}. Experience: {m.get('experience', '?')} years. "
                f"Location: {m.get('location', '')}. Skills: {', '.join(m.get('skills', []))}"
            )
            summaries.append(f"{r['name']} ({m.get('title', '')}, {m.get('experience', '?')} yrs)")
        return [
            {
                "user_input": query,
                "retrieved_contexts": contexts,
                "response": "Top candidates: " + "; ".join(summaries),
                "reference": f"Candidates whose profiles show these skills: {', '.join(expected_skills)}.",
            }
        ]

    def evaluate_with_ragas(self, data: List[Dict[str, Any]]) -> Optional[RAGEvaluationMetrics]:
        """Returns metrics, or None if RAGAS cannot run (not installed, quota, API change)."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            from ragas import EvaluationDataset, evaluate
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import answer_correctness, answer_relevancy, context_precision, faithfulness

            llm = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)
            )
            emb = LangchainEmbeddingsWrapper(
                GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
            )
            result = evaluate(
                EvaluationDataset.from_list(data),
                metrics=[answer_relevancy, context_precision, faithfulness, answer_correctness],
                llm=llm,
                embeddings=emb,
            )
            df = result.to_pandas()
            vals = {}
            for key in WEIGHTS:
                ok = key in df and not df[key].isna().all()
                vals[key] = float(df[key].mean()) if ok else 0.0
            overall = sum(vals[k] * w for k, w in WEIGHTS.items())
            return RAGEvaluationMetrics(**vals, overall_score=overall, method="ragas")
        except Exception as e:
            logger.warning(f"RAGAS unavailable ({type(e).__name__}: {e}) - using heuristic metrics")
            return None

    # -------------------------------------------------------------- heuristic
    def heuristic_metrics(
        self, query: str, results: List[Dict[str, Any]], expected: List[str]
    ) -> RAGEvaluationMetrics:
        """No-LLM fallback. This is NOT RAGAS, so the method is labelled 'heuristic'."""
        wanted = [s.lower() for s in expected]

        def coverage(r: Dict[str, Any]) -> float:
            if not wanted:
                return 0.0
            blob = (r.get("text", "") + " " + " ".join(r["metadata"].get("skills", []))).lower()
            return sum(1 for s in wanted if s in blob) / len(wanted)

        per_result = [coverage(r) for r in results]
        precision = sum(1 for c in per_result if c >= 0.5) / len(results)
        correctness = sum(per_result) / len(per_result)

        q_tokens = set(query.lower().split())
        overlaps = [
            len(q_tokens & set(r.get("text", "").lower().split())) / max(len(q_tokens), 1)
            for r in results
        ]
        relevancy = min(sum(overlaps) / len(overlaps), 1.0)

        vals = {
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "faithfulness": 1.0,  # the response is built from the retrieved contexts by construction
            "answer_correctness": correctness,
        }
        overall = sum(vals[k] * w for k, w in WEIGHTS.items())
        return RAGEvaluationMetrics(**vals, overall_score=overall, method="heuristic")

    # -------------------------------------------------------------- reporting
    def get_evaluation_summary(self) -> Dict[str, Any]:
        if not self.evaluation_history:
            return {"count": 0}
        n = len(self.evaluation_history)
        summary: Dict[str, Any] = {"count": n}
        for key in list(WEIGHTS) + ["overall_score"]:
            summary[f"avg_{key}"] = round(sum(h[key] for h in self.evaluation_history) / n, 3)
        return summary

    def export_evaluations(self, path=None) -> str:
        path = path or (DATA_DIR / "evaluations.csv")
        if not self.evaluation_history:
            return ""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.evaluation_history[0].keys()))
            writer.writeheader()
            writer.writerows(self.evaluation_history)
        return str(path)