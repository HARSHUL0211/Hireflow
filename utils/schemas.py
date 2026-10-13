from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class Resume(BaseModel):
    candidate_id: str
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    title: Optional[str] = None
    text: str = ""
    skills: List[str] = Field(default_factory=list)
    experience: Optional[int] = Field(default=None, description="Total years of experience")


class JobDescription(BaseModel):
    jd_id: str
    title: str
    text: str = ""
    required_skills: List[str] = Field(default_factory=list)
    optional_skills: List[str] = Field(default_factory=list)
    min_experience: Optional[int] = None
    location: Optional[str] = None


class CandidateEvaluation(BaseModel):
    candidate_id: str
    candidate_name: str = ""
    fit_score: int = Field(ge=0, le=100)
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    summary: str = ""
    evidence: Optional[Dict[str, Any]] = None


class LLMEvaluation(BaseModel):
    """Schema the LLM must fill in (used with structured output)."""
    fit_score: int = Field(ge=0, le=100, description="Overall fit 0-100")
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    summary: str = ""


class LLMResumeFields(BaseModel):
    """Schema the LLM must fill in when parsing a resume."""
    name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: Optional[int] = None


class LLMJobFields(BaseModel):
    title: str = ""
    required_skills: List[str] = Field(default_factory=list)
    optional_skills: List[str] = Field(default_factory=list)
    min_experience: Optional[int] = None
    location: Optional[str] = None