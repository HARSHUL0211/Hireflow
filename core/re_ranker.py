import time
from typing import Any, Dict, List, Union

from core.hybrid_indexer import tokenize
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import CandidateEvaluation, JobDescription, LLMEvaluation, Resume
from utils.utils import get_logger, is_quota_error, truncate_text

logger = get_logger(__name__)

PROMPT = """You are a senior technical recruiter. Evaluate how well the candidate fits the job.

JOB DESCRIPTION
Title: {title}
Required skills: {required}
Minimum experience: {min_exp} years
Details:
{jd_text}

CANDIDATE RESUME
{resume_text}

Return:
- fit_score: integer 0-100 (be strict; 90+ only for near-perfect matches)
- strengths: 2-4 short bullet strings grounded in the resume
- gaps: missing skills/experience relative to the job
- risks: possible red flags (job hopping, gaps, seniority mismatch); empty list if none
- summary: 2 sentences a recruiter can read quickly"""


def to_resume(c: Union[Resume, Dict[str, Any]]) -> Resume:
    if isinstance(c, Resume):
        return c
    meta = c.get("metadata", {})
    try:
        return Resume(
            candidate_id=meta.get("candidate_id", c.get("candidate_id", "")),
            name=meta.get("name", c.get("name", "")),
            email=meta.get("email"),
            phone=meta.get("phone"),
            location=meta.get("location"),
            title=meta.get("title"),
            text=c.get("text", ""),
            skills=meta.get("skills", []),
            experience=meta.get("experience"),
        )
    except Exception:
        # most likely an invalid email: build the resume without it
        return Resume(
            candidate_id=meta.get("candidate_id", ""),
            name=meta.get("name", ""),
            text=c.get("text", ""),
            skills=meta.get("skills", []),
            experience=meta.get("experience"),
        )


class ReRanker:
    def __init__(self):
        self.llm = None
        self.chain = None
        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1
                )
                self.chain = self.llm.with_structured_output(LLMEvaluation)
            except Exception as e:
                logger.warning(f"ReRanker LLM unavailable: {e}")

    def is_available(self) -> bool:
        return self.chain is not None

    def evaluate_candidate(self, resume: Resume, jd: JobDescription) -> CandidateEvaluation:
        if self.chain is not None:
            try:
                prompt = PROMPT.format(
                    title=jd.title,
                    required=", ".join(jd.required_skills) or "not specified",
                    min_exp=jd.min_experience if jd.min_experience is not None else "not specified",
                    jd_text=truncate_text(jd.text, 3000),
                    resume_text=truncate_text(resume.text, 6000),
                )
                out = self.chain.invoke(prompt)
                return CandidateEvaluation(
                    candidate_id=resume.candidate_id,
                    candidate_name=resume.name,
                    fit_score=out.fit_score,
                    strengths=out.strengths,
                    gaps=out.gaps,
                    risks=out.risks,
                    summary=out.summary,
                    evidence={"method": "llm", "model": LLM_MODEL},
                )
            except Exception as e:
                logger.warning(f"LLM evaluation failed for {resume.name}: {e}")
                if is_quota_error(e):
                    self.chain = None  # stop hammering a rate-limited API in this run
        return self.simple_evaluation(resume, jd)

    def simple_evaluation(self, resume: Resume, jd: JobDescription) -> CandidateEvaluation:
        """Rule-based: 60% skills/keyword overlap + 40% experience."""
        resume_text = resume.text.lower()
        resume_skills = [s.lower() for s in resume.skills]
        if jd.required_skills:
            matched = [
                s for s in jd.required_skills
                if s.lower() in resume_text or any(s.lower() in rs for rs in resume_skills)
            ]
            missing = [s for s in jd.required_skills if s not in matched]
            skill_ratio = len(matched) / len(jd.required_skills)
        else:
            # no explicit skills -> keyword overlap with the JD text
            jd_tokens = set(tokenize(jd.text or jd.title))
            res_tokens = set(tokenize(resume.text))
            matched = sorted(jd_tokens & res_tokens)[:8]
            missing = []
            skill_ratio = len(jd_tokens & res_tokens) / max(len(jd_tokens), 1)

        exp_ratio = 1.0
        gaps, strengths = [], []
        if jd.min_experience:
            exp_ratio = min((resume.experience or 0) / jd.min_experience, 1.0)
            if exp_ratio < 1:
                gaps.append(f"Has {resume.experience or 0} yrs, needs {jd.min_experience}+")
            else:
                strengths.append(f"Meets experience requirement ({resume.experience} yrs)")
        if matched:
            strengths.append("Matches: " + ", ".join(matched[:6]))
        gaps += [f"Missing: {s}" for s in missing]

        score = int(round(100 * (0.6 * skill_ratio + 0.4 * exp_ratio)))
        summary = (
            f"Rule-based estimate: {int(skill_ratio * 100)}% skill match, "
            f"{resume.experience or 0} yrs experience."
        )
        return CandidateEvaluation(
            candidate_id=resume.candidate_id,
            candidate_name=resume.name,
            fit_score=max(0, min(100, score)),
            strengths=strengths,
            gaps=gaps,
            risks=[],
            summary=summary,
            evidence={"method": "rule_based"},
        )

    def re_rank_candidates(
        self,
        candidates: List[Union[Resume, Dict[str, Any]]],
        jd: JobDescription,
        delay: float = 1.0,
    ) -> List[CandidateEvaluation]:
        evals = []
        for c in candidates:
            evals.append(self.evaluate_candidate(to_resume(c), jd))
            if self.chain is not None:
                time.sleep(delay)  # stay under free-tier rate limits
        return sorted(evals, key=lambda e: e.fit_score, reverse=True)