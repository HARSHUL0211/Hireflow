from typing import List

from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.utils import get_logger, llm_text

logger = get_logger(__name__)


class MultiQueryGenerator:
    def __init__(self, temperature: float = 0.3):
        self.llm = None
        self.temperature = temperature
        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY,
                                                  temperature=temperature)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Multi-query LLM unavailable: {e}")

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature
        if self.llm is not None:
            self.llm.temperature = temperature

    def generate_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        if self.llm is None:
            return self.generate_fallback(original_query, num_queries)
        try:
            return self.generate_with_llm(original_query, num_queries)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Multi-query LLM failed ({e}); using fallback")
            return self.generate_fallback(original_query, num_queries)

    def generate_with_llm(self, original_query: str, num_queries: int) -> List[str]:
        prompt = (
            f"Generate {num_queries} different search queries for finding resumes that match this "
            f"hiring need. Each query should focus on a different angle (job title, key skills, "
            f"tools, responsibilities) and use different wording. One query per line, no numbering, "
            f"no extra text.\n\nHiring need: {original_query}"
        )
        text = llm_text(self.llm.invoke(prompt))
        queries = [ln.strip(" -*0123456789.)\t") for ln in text.splitlines() if ln.strip()]
        queries = [q for q in queries if q][:num_queries]
        if len(queries) < num_queries:
            queries += self.generate_fallback(original_query, num_queries - len(queries))
        return [original_query] + [q for q in queries if q.lower() != original_query.lower()][:num_queries]

    def generate_fallback(self, original_query: str, num_queries: int) -> List[str]:
        words = original_query.split()
        variants = [original_query]
        if len(words) > 4:
            variants.append(" ".join(words[:max(3, len(words) // 2)]))
            variants.append(" ".join(words[-max(3, len(words) // 2):]))
        if len(words) > 1:
            variants.append(f"{words[0]} {words[-1]}")
        out: List[str] = []
        for v in variants:
            if v and v not in out:
                out.append(v)
        return out[:max(1, num_queries)]