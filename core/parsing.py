import re
from datetime import datetime
from typing import Dict, List, Optional

from utils.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, USE_LLM_PARSING
from utils.schemas import JobDescription, LLMJobFields, LLMResumeFields, Resume
from utils.utils import clean_text, get_logger, truncate_text

logger = get_logger(__name__)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", re.IGNORECASE)
SECTION_HEADERS = ["TECHNICAL SKILLS", "SOFTWARE PROFICIENCY", "SKILLS", "CERTIFICATIONS",
                   "PROFESSIONAL EXPERIENCE", "PROFESSIONAL SUMMARY", "EDUCATION", "CONTACT INFORMATION"]


def _section(text: str, header: str) -> str:
    """Text between `header` and the next ALL-CAPS section header."""
    pattern = rf"^{re.escape(header)}\s*$(.*?)(?=^\s*[A-Z][A-Z &/]{{5,}}\s*$|\Z)"
    m = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    return m.group(1) if m else ""


def _bullets(block: str) -> List[str]:
    return [re.sub(r"^[•\-*\x7f]\s*", "", ln).strip() for ln in block.splitlines()
            if ln.strip().startswith(("•", "-", "*", "\x7f")) and len(ln.strip()) > 2]


def _estimate_experience(text: str) -> int:
    """Prefer an explicit '5+ years' statement; otherwise derive it from job date ranges."""
    stated = [int(y) for y in YEARS_RE.findall(text)]
    if stated:
        return max(stated)
    this_year = datetime.now().year
    starts, ends = [], []
    for a, b in re.findall(r"(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|19\d{2}|Present)", text, flags=re.IGNORECASE):
        starts.append(int(a))
        ends.append(this_year if b.lower() == "present" else int(b))
    return max(ends) - min(starts) if starts else 0


def regex_parse_resume(text: str, candidate_id: str, fallback_name: str = "") -> Resume:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    name = lines[0].title() if lines else fallback_name
    title = lines[1] if len(lines) > 1 and "@" not in lines[1] and ":" not in lines[1] else None

    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    loc = re.search(r"Location:\s*(.+)", text)

    skills: List[str] = []
    for header in ("TECHNICAL SKILLS", "SOFTWARE PROFICIENCY", "SKILLS"):
        skills += _bullets(_section(text, header))
    skills = list(dict.fromkeys(s for s in skills if s))  # de-dup, keep order

    experience = _estimate_experience(text)

    return Resume(
        candidate_id=candidate_id, name=name or fallback_name,
        email=email.group(0) if email else None,
        phone=phone.group(0) if phone else None,
        location=loc.group(1).strip() if loc else None,
        title=title, text=text, skills=skills, experience=experience,
    )


def regex_parse_jd(text: str, jd_id: str, title: str = "") -> JobDescription:
    m_exp = YEARS_RE.search(text)
    m_loc = re.search(r"Location:\s*(.+)", text)
    req = re.search(r"(?:required|must[- ]have)[^\n:]*:\s*(.+)", text, re.IGNORECASE)
    required = [s.strip() for s in re.split(r",|;", req.group(1)) if s.strip()] if req else []
    return JobDescription(jd_id=jd_id, title=title or text.splitlines()[0][:80] if text else title,
                          text=text, required_skills=required,
                          min_experience=int(m_exp.group(1)) if m_exp else None,
                          location=m_loc.group(1).strip() if m_loc else None)


class _LLMBase:
    def __init__(self):
        self.llm = None
        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY,
                                                  temperature=LLM_TEMPERATURE)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LLM unavailable for parsing: {e}")


class ResumeParser(_LLMBase):
    PROMPT = (
        "You are an expert resume parser. Extract structured information from the resume below.\n"
        "- skills: every technical skill and software tool mentioned\n"
        "- experience: total years of professional experience as an integer (null if unknown)\n"
        "- title: the candidate's current/most recent job title\n\n"
        "Resume Text:\n{resume_text}"
    )

    def __init__(self, use_llm: bool = USE_LLM_PARSING):
        super().__init__()
        self.use_llm = use_llm and self.llm is not None
        self.chain = self.llm.with_structured_output(LLMResumeFields) if self.use_llm else None

    def parse_resume(self, text: str, candidate_id: str, fallback_name: str = "") -> Resume:
        base = regex_parse_resume(text, candidate_id, fallback_name)  # always available
        if not self.use_llm:
            return base
        try:
            prompt = self.PROMPT.format(resume_text=truncate_text(clean_text_keep_lines(text), 8000))
            out: LLMResumeFields = self.chain.invoke(prompt)
            return base.model_copy(update={
                "name": out.name or base.name,
                "title": out.title or base.title,
                "location": out.location or base.location,
                "skills": out.skills or base.skills,
                "experience": out.experience if out.experience is not None else base.experience,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM parsing failed for {candidate_id} ({e}); using regex result")
            return base


class JobParser(_LLMBase):
    PROMPT = (
        "You are an expert recruiter. Extract structured data from this job description.\n"
        "required_skills = must-haves, optional_skills = nice-to-haves, "
        "min_experience = minimum years (integer or null).\n\nJob Description:\n{jd_text}"
    )

    def __init__(self, use_llm: bool = True):
        super().__init__()
        self.use_llm = use_llm and self.llm is not None
        self.chain = self.llm.with_structured_output(LLMJobFields) if self.use_llm else None

    def parse_job_description(self, text: str, jd_id: str, title: str = "") -> JobDescription:
        base = regex_parse_jd(text, jd_id, title)
        if not self.use_llm:
            return base
        try:
            out: LLMJobFields = self.chain.invoke(self.PROMPT.format(jd_text=truncate_text(text, 8000)))
            return base.model_copy(update={
                "title": out.title or base.title,
                "required_skills": out.required_skills or base.required_skills,
                "optional_skills": out.optional_skills,
                "min_experience": out.min_experience if out.min_experience is not None else base.min_experience,
                "location": out.location or base.location,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM JD parsing failed ({e}); using regex result")
            return base


def clean_text_keep_lines(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()